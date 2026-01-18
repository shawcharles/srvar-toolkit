from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.dataset import Dataset
from .elb import ElbSpec
from .spec import ModelSpec, MuSSVSSpec, PriorSpec, SamplerConfig, SteadyStateSpec
from .sv import VolatilitySpec


class ConfigError(ValueError):
    pass


def _prepare_from_config(
    cfg: dict[str, Any],
    *,
    emit: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[
    Dataset, ModelSpec, PriorSpec, SamplerConfig, np.random.Generator, dict[str, Any] | None
]:
    ds = load_dataset_from_csv(cfg)
    start = None
    end = None
    try:
        if isinstance(ds.time_index, pd.DatetimeIndex) and len(ds.time_index) > 0:
            start = ds.time_index[0]
            end = ds.time_index[-1]
    except Exception:
        start = None
        end = None

    if emit is not None:
        emit(
            "summary",
            {
                "kind": "dataset",
                "T": ds.T,
                "N": ds.N,
                "variables": list(ds.variables),
                "start": str(start) if start is not None else None,
                "end": str(end) if end is not None else None,
            },
        )

    model = build_model(cfg, dataset=ds)
    if emit is not None:
        emit(
            "summary",
            {
                "kind": "model",
                "p": model.p,
                "include_intercept": model.include_intercept,
                "steady_state": bool(model.steady_state is not None),
                "elb": bool(model.elb is not None and model.elb.enabled),
                "sv": bool(model.volatility is not None and model.volatility.enabled),
            },
        )

    if model.elb is not None and model.elb.enabled:
        missing = [v for v in model.elb.applies_to if v not in ds.variables]
        if missing:
            raise ConfigError(f"model.elb.applies_to not found in dataset.variables: {missing}")

    prior = build_prior(cfg, dataset=ds, model=model)
    prior_cfg = cfg.get("prior", {})
    if emit is not None and isinstance(prior_cfg, dict):
        family = prior_cfg.get("family")
        method = prior_cfg.get("method")
        emit(
            "summary",
            {
                "kind": "prior",
                "family": str(family) if family is not None else None,
                "method": str(method) if method is not None else None,
            },
        )

    sampler, rng = build_sampler(cfg)
    if emit is not None:
        emit(
            "summary",
            {
                "kind": "sampler",
                "draws": sampler.draws,
                "burn_in": sampler.burn_in,
                "thin": sampler.thin,
            },
        )

    fc_cfg = build_forecast_config(cfg)
    if emit is not None and fc_cfg is not None:
        emit(
            "summary",
            {
                "kind": "forecast",
                "horizons": list(fc_cfg["horizons"]),
                "draws": int(fc_cfg["draws"]),
                "quantile_levels": list(fc_cfg["quantile_levels"]),
            },
        )

    return ds, model, prior, sampler, rng, fc_cfg


def _require_pyyaml() -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for the config-driven CLI. Install with 'srvar-toolkit[cli]'."
        ) from exc
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    yaml = _require_pyyaml()
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return raw


def _get(cfg: dict[str, Any], key: str, *, default: Any = None, required: bool = False) -> Any:
    if key in cfg:
        return cfg[key]
    if required:
        raise ConfigError(f"missing required key: {key}")
    return default


def _as_bool(x: Any, *, key: str) -> bool:
    if isinstance(x, bool):
        return bool(x)
    raise ConfigError(f"{key} must be a boolean")


def _as_int(x: Any, *, key: str, min_value: int | None = None) -> int:
    if not isinstance(x, (int, np.integer)) or isinstance(x, bool):
        raise ConfigError(f"{key} must be an integer")
    xi = int(x)
    if min_value is not None and xi < min_value:
        raise ConfigError(f"{key} must be >= {min_value}")
    return xi


def _as_float(x: Any, *, key: str) -> float:
    if not isinstance(x, (float, int, np.floating, np.integer)) or isinstance(x, bool):
        raise ConfigError(f"{key} must be a number")
    return float(x)


def _as_str_list(x: Any, *, key: str) -> list[str]:
    if not isinstance(x, list) or not all(isinstance(v, str) for v in x):
        raise ConfigError(f"{key} must be a list[str]")
    return list(x)


def load_dataset_from_csv(cfg: dict[str, Any]) -> Dataset:
    data_cfg = _get(cfg, "data", required=True)
    if not isinstance(data_cfg, dict):
        raise ConfigError("data must be a mapping")

    csv_path = Path(_get(data_cfg, "csv_path", required=True))
    if not csv_path.exists():
        raise ConfigError(f"data.csv_path not found: {csv_path}")

    date_column = _get(data_cfg, "date_column", required=True)
    if not isinstance(date_column, str) or not date_column:
        raise ConfigError("data.date_column must be a non-empty string")

    variables = _as_str_list(_get(data_cfg, "variables", required=True), key="data.variables")

    date_format = _get(data_cfg, "date_format", default=None)
    if date_format is not None and (not isinstance(date_format, str) or not date_format):
        raise ConfigError("data.date_format must be a non-empty string when provided")

    dropna = _as_bool(_get(data_cfg, "dropna", default=True), key="data.dropna")

    df = pd.read_csv(csv_path)
    if date_column not in df.columns:
        raise ConfigError(f"data.date_column not in CSV columns: {date_column}")

    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ConfigError(f"data.variables missing from CSV columns: {missing}")

    dt = pd.to_datetime(df[date_column], format=date_format, errors="raise")
    df = df.drop(columns=[date_column])
    df.index = pd.DatetimeIndex(dt, name=date_column)
    df = df.sort_index()

    x = df.loc[:, variables]
    if dropna:
        x = x.dropna(axis=0, how="any")

    values = x.to_numpy(dtype=float, copy=True)
    return Dataset.from_arrays(values=values, variables=variables, time_index=x.index)


def build_model(cfg: dict[str, Any], *, dataset: Dataset) -> ModelSpec:
    model_cfg = _get(cfg, "model", required=True)
    if not isinstance(model_cfg, dict):
        raise ConfigError("model must be a mapping")

    p = _as_int(_get(model_cfg, "p", required=True), key="model.p", min_value=1)
    include_intercept = _as_bool(
        _get(model_cfg, "include_intercept", default=True), key="model.include_intercept"
    )

    elb_spec: ElbSpec | None = None
    elb_cfg = _get(model_cfg, "elb", default=None)
    if elb_cfg is not None:
        if not isinstance(elb_cfg, dict):
            raise ConfigError("model.elb must be a mapping")
        enabled = _as_bool(_get(elb_cfg, "enabled", default=True), key="model.elb.enabled")
        if enabled:
            bound = _as_float(_get(elb_cfg, "bound", required=True), key="model.elb.bound")
            applies_to = _as_str_list(
                _get(elb_cfg, "applies_to", required=True), key="model.elb.applies_to"
            )
            tol = _as_float(_get(elb_cfg, "tol", default=1e-8), key="model.elb.tol")
            init_offset = _as_float(
                _get(elb_cfg, "init_offset", default=0.05), key="model.elb.init_offset"
            )
            elb_spec = ElbSpec(
                bound=bound, applies_to=applies_to, tol=tol, init_offset=init_offset, enabled=True
            )

    vol_spec: VolatilitySpec | None = None
    vol_cfg = _get(model_cfg, "volatility", default=None)
    if vol_cfg is not None:
        if not isinstance(vol_cfg, dict):
            raise ConfigError("model.volatility must be a mapping")
        enabled = _as_bool(_get(vol_cfg, "enabled", default=True), key="model.volatility.enabled")
        if enabled:
            dynamics = _get(vol_cfg, "dynamics", default="rw")
            if not isinstance(dynamics, str) or not dynamics:
                raise ConfigError("model.volatility.dynamics must be a non-empty string")
            dynamics_l = dynamics.lower()
            if dynamics_l not in {"rw", "ar1"}:
                raise ConfigError("model.volatility.dynamics must be one of {'rw','ar1'}")

            covariance = _get(vol_cfg, "covariance", default="diagonal")
            if not isinstance(covariance, str) or not covariance:
                raise ConfigError("model.volatility.covariance must be a non-empty string")
            covariance_l = covariance.lower()
            if covariance_l not in {"diagonal", "triangular"}:
                raise ConfigError(
                    "model.volatility.covariance must be one of {'diagonal','triangular'}"
                )

            q_prior_var_raw = _get(vol_cfg, "q_prior_var", default=1.0)
            q_prior_var = _as_float(q_prior_var_raw, key="model.volatility.q_prior_var")

            phi_prior_mean = _as_float(
                _get(vol_cfg, "phi_prior_mean", default=0.95), key="model.volatility.phi_prior_mean"
            )
            phi_prior_var = _as_float(
                _get(vol_cfg, "phi_prior_var", default=0.1), key="model.volatility.phi_prior_var"
            )
            gamma0_prior_mean = _as_float(
                _get(vol_cfg, "gamma0_prior_mean", default=0.0),
                key="model.volatility.gamma0_prior_mean",
            )
            gamma0_prior_var = _as_float(
                _get(vol_cfg, "gamma0_prior_var", default=10.0),
                key="model.volatility.gamma0_prior_var",
            )
            vol_spec = VolatilitySpec(
                enabled=True,
                dynamics=dynamics_l,  # type: ignore[arg-type]
                covariance=covariance_l,  # type: ignore[arg-type]
                q_prior_var=q_prior_var,
                epsilon=_as_float(
                    _get(vol_cfg, "epsilon", default=1e-4), key="model.volatility.epsilon"
                ),
                h0_prior_mean=_as_float(
                    _get(vol_cfg, "h0_prior_mean", default=1e-6),
                    key="model.volatility.h0_prior_mean",
                ),
                h0_prior_var=_as_float(
                    _get(vol_cfg, "h0_prior_var", default=10.0),
                    key="model.volatility.h0_prior_var",
                ),
                sigma_eta_prior_nu0=_as_float(
                    _get(vol_cfg, "sigma_eta_prior_nu0", default=1.0),
                    key="model.volatility.sigma_eta_prior_nu0",
                ),
                sigma_eta_prior_s0=_as_float(
                    _get(vol_cfg, "sigma_eta_prior_s0", default=0.01),
                    key="model.volatility.sigma_eta_prior_s0",
                ),
                phi_prior_mean=phi_prior_mean,
                phi_prior_var=phi_prior_var,
                gamma0_prior_mean=gamma0_prior_mean,
                gamma0_prior_var=gamma0_prior_var,
            )

    ss_spec: SteadyStateSpec | None = None
    ss_cfg = _get(model_cfg, "steady_state", default=None)
    if ss_cfg is not None:
        if not isinstance(ss_cfg, dict):
            raise ConfigError("model.steady_state must be a mapping")

        ss_enabled = _as_bool(
            _get(ss_cfg, "enabled", default=True), key="model.steady_state.enabled"
        )
        if ss_enabled:
            mu0_raw = _get(ss_cfg, "mu0", required=True)
            if not isinstance(mu0_raw, list) or not all(
                isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool)
                for v in mu0_raw
            ):
                raise ConfigError("model.steady_state.mu0 must be a list of numbers")
            mu0 = np.asarray([float(v) for v in mu0_raw], dtype=float)
            if mu0.shape != (dataset.N,):
                raise ConfigError("model.steady_state.mu0 must have length N")

            v0_mu_raw = _get(ss_cfg, "v0_mu", required=True)
            if isinstance(v0_mu_raw, list):
                if not all(
                    isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool)
                    for v in v0_mu_raw
                ):
                    raise ConfigError(
                        "model.steady_state.v0_mu must be a number or list of numbers"
                    )
                v0_mu: float | np.ndarray = np.asarray([float(v) for v in v0_mu_raw], dtype=float)
            else:
                v0_mu = _as_float(v0_mu_raw, key="model.steady_state.v0_mu")

            mu_ssvs_spec: MuSSVSSpec | None = None
            mu_ssvs_cfg = _get(ss_cfg, "ssvs", default=None)
            if mu_ssvs_cfg is not None:
                if not isinstance(mu_ssvs_cfg, dict):
                    raise ConfigError("model.steady_state.ssvs must be a mapping")
                mu_ssvs_enabled = _as_bool(
                    _get(mu_ssvs_cfg, "enabled", default=True),
                    key="model.steady_state.ssvs.enabled",
                )
                if mu_ssvs_enabled:
                    spike_var = _as_float(
                        _get(mu_ssvs_cfg, "spike_var", default=1e-4),
                        key="model.steady_state.ssvs.spike_var",
                    )
                    slab_var = _as_float(
                        _get(mu_ssvs_cfg, "slab_var", default=100.0),
                        key="model.steady_state.ssvs.slab_var",
                    )
                    inclusion_prob = _as_float(
                        _get(mu_ssvs_cfg, "inclusion_prob", default=0.5),
                        key="model.steady_state.ssvs.inclusion_prob",
                    )
                    try:
                        mu_ssvs_spec = MuSSVSSpec(
                            spike_var=float(spike_var),
                            slab_var=float(slab_var),
                            inclusion_prob=float(inclusion_prob),
                        )
                    except ValueError as e:
                        raise ConfigError(str(e)) from e

            try:
                ss_spec = SteadyStateSpec(mu0=mu0, v0_mu=v0_mu, ssvs=mu_ssvs_spec)
            except ValueError as e:
                raise ConfigError(str(e)) from e

    return ModelSpec(
        p=p,
        include_intercept=include_intercept,
        steady_state=ss_spec,
        elb=elb_spec,
        volatility=vol_spec,
    )


def build_prior(cfg: dict[str, Any], *, dataset: Dataset, model: ModelSpec) -> PriorSpec:
    prior_cfg = _get(cfg, "prior", required=True)
    if not isinstance(prior_cfg, dict):
        raise ConfigError("prior must be a mapping")

    family = _get(prior_cfg, "family", required=True)
    if not isinstance(family, str) or not family:
        raise ConfigError("prior.family must be a non-empty string")

    family_l = family.lower()
    k = (1 if model.include_intercept else 0) + dataset.N * model.p

    method = _get(prior_cfg, "method", default=None)
    if method is not None and (not isinstance(method, str) or not method):
        raise ConfigError("prior.method must be a non-empty string when provided")

    if family_l == "niw":
        method_l = (method or "minnesota").lower()
        if method_l == "default":
            return PriorSpec.niw_default(k=k, n=dataset.N)
        if method_l == "minnesota":
            hyp = _get(prior_cfg, "minnesota", default={})
            if not isinstance(hyp, dict):
                raise ConfigError("prior.minnesota must be a mapping")
            kwargs: dict[str, Any] = {}
            for name in ["lambda1", "lambda2", "lambda3", "lambda4", "own_lag_mean", "min_sigma2"]:
                if name in hyp:
                    kwargs[name] = hyp[name]
            if "own_lag_means" in hyp:
                kwargs["own_lag_means"] = hyp["own_lag_means"]
            return PriorSpec.niw_minnesota(
                p=model.p,
                y=dataset.values,
                n=dataset.N,
                include_intercept=model.include_intercept,
                **kwargs,
            )
        raise ConfigError("prior.method for family='niw' must be one of: default, minnesota")

    if family_l == "ssvs":
        hyp = _get(prior_cfg, "ssvs", default={})
        if not isinstance(hyp, dict):
            raise ConfigError("prior.ssvs must be a mapping")
        kwargs2: dict[str, Any] = {}
        for name in [
            "spike_var",
            "slab_var",
            "inclusion_prob",
            "intercept_slab_var",
            "fix_intercept",
        ]:
            if name in hyp:
                kwargs2[name] = hyp[name]
        return PriorSpec.from_ssvs(
            k=k, n=dataset.N, include_intercept=model.include_intercept, **kwargs2
        )

    if family_l == "blasso":
        hyp = _get(prior_cfg, "blasso", default={})
        if not isinstance(hyp, dict):
            raise ConfigError("prior.blasso must be a mapping")

        kwargs3: dict[str, Any] = {}
        if "mode" in hyp:
            kwargs3["mode"] = hyp["mode"]
        for name in [
            "a0_global",
            "b0_global",
            "a0_c",
            "b0_c",
            "a0_L",
            "b0_L",
            "tau_init",
            "lambda_init",
        ]:
            if name in hyp:
                kwargs3[name] = hyp[name]

        return PriorSpec.from_blasso(
            k=k, n=dataset.N, include_intercept=model.include_intercept, **kwargs3
        )

    if family_l == "dl":
        hyp = _get(prior_cfg, "dl", default={})
        if not isinstance(hyp, dict):
            raise ConfigError("prior.dl must be a mapping")

        kwargs4: dict[str, Any] = {}
        for name in ["abeta", "dl_scaler"]:
            if name in hyp:
                kwargs4[name] = hyp[name]

        return PriorSpec.from_dl(
            k=k, n=dataset.N, include_intercept=model.include_intercept, **kwargs4
        )

    raise ConfigError("prior.family must be one of: niw, ssvs, blasso, dl")


def build_sampler(cfg: dict[str, Any]) -> tuple[SamplerConfig, np.random.Generator]:
    sampler_cfg = _get(cfg, "sampler", required=True)
    if not isinstance(sampler_cfg, dict):
        raise ConfigError("sampler must be a mapping")

    draws = _as_int(_get(sampler_cfg, "draws", default=2000), key="sampler.draws", min_value=1)
    burn_in = _as_int(_get(sampler_cfg, "burn_in", default=500), key="sampler.burn_in", min_value=0)
    thin = _as_int(_get(sampler_cfg, "thin", default=1), key="sampler.thin", min_value=1)

    seed = _get(sampler_cfg, "seed", default=None)
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(_as_int(seed, key="sampler.seed", min_value=0))

    return SamplerConfig(draws=draws, burn_in=burn_in, thin=thin), rng


def build_forecast_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    fc_cfg = _get(cfg, "forecast", default=None)
    if fc_cfg is None:
        return None
    if not isinstance(fc_cfg, dict):
        raise ConfigError("forecast must be a mapping")

    enabled = _as_bool(_get(fc_cfg, "enabled", default=True), key="forecast.enabled")
    if not enabled:
        return None

    horizons = _get(fc_cfg, "horizons", required=True)
    if not isinstance(horizons, list) or not all(
        isinstance(v, (int, np.integer)) for v in horizons
    ):
        raise ConfigError("forecast.horizons must be a list[int]")
    horizons_i = [int(v) for v in horizons]
    if not horizons_i or any(h < 1 for h in horizons_i):
        raise ConfigError("forecast.horizons must contain positive integers")

    draws = _as_int(_get(fc_cfg, "draws", default=1000), key="forecast.draws", min_value=1)

    q = _get(fc_cfg, "quantile_levels", default=[0.1, 0.5, 0.9])
    if not isinstance(q, list) or not all(
        isinstance(v, (float, int, np.floating, np.integer)) for v in q
    ):
        raise ConfigError("forecast.quantile_levels must be a list[float]")
    qf = [float(v) for v in q]

    stationarity = _get(fc_cfg, "stationarity", default="allow")
    if stationarity is None:
        stationarity = "allow"
    if not isinstance(stationarity, str) or not stationarity:
        raise ConfigError("forecast.stationarity must be a non-empty string when provided")
    stationarity_l = stationarity.lower()
    if stationarity_l not in {"allow", "reject"}:
        raise ConfigError("forecast.stationarity must be one of: allow, reject")

    stationarity_tol = _as_float(
        _get(fc_cfg, "stationarity_tol", default=1e-10), key="forecast.stationarity_tol"
    )
    if stationarity_tol < 0 or not np.isfinite(stationarity_tol):
        raise ConfigError("forecast.stationarity_tol must be finite and >= 0")

    stationarity_max_draws_raw = _get(fc_cfg, "stationarity_max_draws", default=None)
    if stationarity_max_draws_raw is None:
        stationarity_max_draws = None
    else:
        stationarity_max_draws = _as_int(
            stationarity_max_draws_raw, key="forecast.stationarity_max_draws", min_value=1
        )

    return {
        "horizons": horizons_i,
        "draws": draws,
        "quantile_levels": qf,
        "stationarity": stationarity_l,
        "stationarity_tol": float(stationarity_tol),
        "stationarity_max_draws": stationarity_max_draws,
    }


def build_backtest_config(cfg: dict[str, Any], *, model: ModelSpec) -> dict[str, Any]:
    bt_cfg = _get(cfg, "backtest", required=True)
    if not isinstance(bt_cfg, dict):
        raise ConfigError("backtest must be a mapping")

    mode = _get(bt_cfg, "mode", default=None)
    if mode is not None and (not isinstance(mode, str) or not mode):
        raise ConfigError("backtest.mode must be a non-empty string when provided")
    mode_l = (mode or "expanding").lower()
    if mode_l not in {"expanding", "rolling"}:
        raise ConfigError("backtest.mode must be one of: expanding, rolling")

    window = _get(bt_cfg, "window", default=None)
    if mode_l == "rolling":
        if window is None:
            raise ConfigError("backtest.window is required for mode='rolling'")
        window_i = _as_int(window, key="backtest.window", min_value=2)
    else:
        window_i = None

    min_obs = _as_int(_get(bt_cfg, "min_obs", required=True), key="backtest.min_obs", min_value=2)
    step = _as_int(_get(bt_cfg, "step", default=1), key="backtest.step", min_value=1)

    horizons = _get(bt_cfg, "horizons", required=True)
    if not isinstance(horizons, list) or not all(
        isinstance(v, (int, np.integer)) for v in horizons
    ):
        raise ConfigError("backtest.horizons must be a list[int]")
    horizons_i = [int(v) for v in horizons]
    if not horizons_i or any(h < 1 for h in horizons_i):
        raise ConfigError("backtest.horizons must contain positive integers")

    draws = _as_int(_get(bt_cfg, "draws", default=500), key="backtest.draws", min_value=1)

    q = _get(bt_cfg, "quantile_levels", default=[0.1, 0.5, 0.9])
    if not isinstance(q, list) or not all(
        isinstance(v, (float, int, np.floating, np.integer)) for v in q
    ):
        raise ConfigError("backtest.quantile_levels must be a list[float]")
    qf = [float(v) for v in q]

    origin_start = _get(bt_cfg, "origin_start", default=None)
    origin_end = _get(bt_cfg, "origin_end", default=None)
    if origin_start is not None and not isinstance(origin_start, str):
        raise ConfigError("backtest.origin_start must be a string when provided")
    if origin_end is not None and not isinstance(origin_end, str):
        raise ConfigError("backtest.origin_end must be a string when provided")

    stationarity = _get(bt_cfg, "stationarity", default="allow")
    if stationarity is None:
        stationarity = "allow"
    if not isinstance(stationarity, str) or not stationarity:
        raise ConfigError("backtest.stationarity must be a non-empty string when provided")
    stationarity_l = stationarity.lower()
    if stationarity_l not in {"allow", "reject"}:
        raise ConfigError("backtest.stationarity must be one of: allow, reject")

    stationarity_tol = _as_float(
        _get(bt_cfg, "stationarity_tol", default=1e-10), key="backtest.stationarity_tol"
    )
    if stationarity_tol < 0 or not np.isfinite(stationarity_tol):
        raise ConfigError("backtest.stationarity_tol must be finite and >= 0")

    stationarity_max_draws_raw = _get(bt_cfg, "stationarity_max_draws", default=None)
    if stationarity_max_draws_raw is None:
        stationarity_max_draws = None
    else:
        stationarity_max_draws = _as_int(
            stationarity_max_draws_raw, key="backtest.stationarity_max_draws", min_value=1
        )

    return {
        "mode": mode_l,
        "window": window_i,
        "min_obs": min_obs,
        "step": step,
        "horizons": horizons_i,
        "draws": draws,
        "quantile_levels": qf,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "stationarity": stationarity_l,
        "stationarity_tol": float(stationarity_tol),
        "stationarity_max_draws": stationarity_max_draws,
    }


def build_evaluation_config(
    cfg: dict[str, Any], *, variables: list[str], horizons: list[int]
) -> dict[str, Any]:
    ev_cfg = _get(cfg, "evaluation", default={})
    if not isinstance(ev_cfg, dict):
        raise ConfigError("evaluation must be a mapping")

    elb_cfg = _get(ev_cfg, "elb_censor", default={})
    if elb_cfg is None:
        elb_cfg = {}
    if not isinstance(elb_cfg, dict):
        raise ConfigError("evaluation.elb_censor must be a mapping")
    elb_enabled = _as_bool(
        _get(elb_cfg, "enabled", default=False), key="evaluation.elb_censor.enabled"
    )
    elb_bound: float | None = None
    elb_vars: list[str] = []
    elb_censor_realized = True
    elb_censor_forecasts = False
    if elb_enabled:
        elb_bound = _as_float(
            _get(elb_cfg, "bound", required=True), key="evaluation.elb_censor.bound"
        )
        if not np.isfinite(elb_bound):
            raise ConfigError("evaluation.elb_censor.bound must be finite")

        elb_vars = _as_str_list(
            _get(elb_cfg, "variables", required=True), key="evaluation.elb_censor.variables"
        )
        if len(elb_vars) < 1:
            raise ConfigError("evaluation.elb_censor.variables must be non-empty when enabled")
        for v in elb_vars:
            if v not in variables:
                raise ConfigError(f"evaluation.elb_censor.variables contains unknown variable: {v}")

        elb_censor_realized = _as_bool(
            _get(elb_cfg, "censor_realized", default=True),
            key="evaluation.elb_censor.censor_realized",
        )
        elb_censor_forecasts = _as_bool(
            _get(elb_cfg, "censor_forecasts", default=False),
            key="evaluation.elb_censor.censor_forecasts",
        )

    cov_cfg = _get(ev_cfg, "coverage", default={})
    if not isinstance(cov_cfg, dict):
        raise ConfigError("evaluation.coverage must be a mapping")
    cov_enabled = _as_bool(
        _get(cov_cfg, "enabled", default=True), key="evaluation.coverage.enabled"
    )
    cov_intervals = _get(cov_cfg, "intervals", default=[0.5, 0.8, 0.9])
    if not isinstance(cov_intervals, list) or not all(
        isinstance(v, (float, int, np.floating, np.integer)) for v in cov_intervals
    ):
        raise ConfigError("evaluation.coverage.intervals must be a list[float]")
    cov_intervals_f = [float(v) for v in cov_intervals]
    cov_use_latent = _as_bool(
        _get(cov_cfg, "use_latent", default=False), key="evaluation.coverage.use_latent"
    )

    pit_cfg = _get(ev_cfg, "pit", default={})
    if not isinstance(pit_cfg, dict):
        raise ConfigError("evaluation.pit must be a mapping")
    pit_enabled = _as_bool(_get(pit_cfg, "enabled", default=False), key="evaluation.pit.enabled")
    pit_bins = _as_int(_get(pit_cfg, "bins", default=10), key="evaluation.pit.bins", min_value=2)
    pit_use_latent = _as_bool(
        _get(pit_cfg, "use_latent", default=False), key="evaluation.pit.use_latent"
    )
    pit_vars = _get(pit_cfg, "variables", default=[variables[0]] if variables else [])
    if not isinstance(pit_vars, list) or not all(isinstance(v, str) for v in pit_vars):
        raise ConfigError("evaluation.pit.variables must be a list[str]")
    for v in pit_vars:
        if v not in variables:
            raise ConfigError(f"evaluation.pit.variables contains unknown variable: {v}")
    pit_h = _get(pit_cfg, "horizons", default=[1])
    if not isinstance(pit_h, list) or not all(isinstance(v, (int, np.integer)) for v in pit_h):
        raise ConfigError("evaluation.pit.horizons must be a list[int]")
    pit_h_i = [int(v) for v in pit_h]
    for h in pit_h_i:
        if h not in horizons:
            raise ConfigError(
                f"evaluation.pit.horizons contains horizon not in backtest.horizons: {h}"
            )

    crps_cfg = _get(ev_cfg, "crps", default={})
    if not isinstance(crps_cfg, dict):
        raise ConfigError("evaluation.crps must be a mapping")
    crps_enabled = _as_bool(_get(crps_cfg, "enabled", default=True), key="evaluation.crps.enabled")
    crps_use_latent = _as_bool(
        _get(crps_cfg, "use_latent", default=False), key="evaluation.crps.use_latent"
    )

    metrics_table = _as_bool(
        _get(ev_cfg, "metrics_table", default=True), key="evaluation.metrics_table"
    )

    return {
        "coverage": {
            "enabled": cov_enabled,
            "intervals": cov_intervals_f,
            "use_latent": cov_use_latent,
        },
        "pit": {
            "enabled": pit_enabled,
            "bins": pit_bins,
            "variables": list(pit_vars),
            "horizons": pit_h_i,
            "use_latent": pit_use_latent,
        },
        "crps": {"enabled": crps_enabled, "use_latent": crps_use_latent},
        "elb_censor": {
            "enabled": elb_enabled,
            "bound": elb_bound,
            "variables": list(elb_vars),
            "censor_realized": elb_censor_realized,
            "censor_forecasts": elb_censor_forecasts,
        },
        "metrics_table": metrics_table,
    }


def validate_config(cfg: dict[str, Any]) -> None:
    ds, model, _prior, _sampler, _rng, _fc_cfg = _prepare_from_config(cfg, emit=None)

    if "backtest" in cfg:
        bt = build_backtest_config(cfg, model=model)
        build_evaluation_config(cfg, variables=list(ds.variables), horizons=list(bt["horizons"]))
