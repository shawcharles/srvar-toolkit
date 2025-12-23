from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import time

import numpy as np
import pandas as pd

from .api import fit, forecast
from .data.dataset import Dataset
from .elb import ElbSpec
from .metrics import crps_draws, mae, rmse
from .results import FitResult, ForecastResult
from .spec import ModelSpec, MuSSVSSpec, PriorSpec, SamplerConfig, SteadyStateSpec
from .sv import VolatilitySpec


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    fit_result: FitResult
    forecast_result: ForecastResult | None


def _prepare_from_config(
    cfg: dict[str, Any],
    *,
    emit: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Dataset, ModelSpec, PriorSpec, SamplerConfig, np.random.Generator, dict[str, Any] | None]:
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
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for the config-driven CLI. Install with 'srvar-toolkit[cli]'."
        ) from e
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
        return x
    raise ConfigError(f"{key} must be a boolean")


def _as_int(x: Any, *, key: str, min_value: int | None = None) -> int:
    if not isinstance(x, (int, np.integer)) or isinstance(x, bool):
        raise ConfigError(f"{key} must be an integer")
    v = int(x)
    if min_value is not None and v < min_value:
        raise ConfigError(f"{key} must be >= {min_value}")
    return v


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
    include_intercept = _as_bool(_get(model_cfg, "include_intercept", default=True), key="model.include_intercept")

    elb_spec: ElbSpec | None = None
    elb_cfg = _get(model_cfg, "elb", default=None)
    if elb_cfg is not None:
        if not isinstance(elb_cfg, dict):
            raise ConfigError("model.elb must be a mapping")
        enabled = _as_bool(_get(elb_cfg, "enabled", default=True), key="model.elb.enabled")
        if enabled:
            bound = _as_float(_get(elb_cfg, "bound", required=True), key="model.elb.bound")
            applies_to = _as_str_list(_get(elb_cfg, "applies_to", required=True), key="model.elb.applies_to")
            tol = _as_float(_get(elb_cfg, "tol", default=1e-8), key="model.elb.tol")
            init_offset = _as_float(_get(elb_cfg, "init_offset", default=0.05), key="model.elb.init_offset")
            elb_spec = ElbSpec(bound=bound, applies_to=applies_to, tol=tol, init_offset=init_offset, enabled=True)

    vol_spec: VolatilitySpec | None = None
    vol_cfg = _get(model_cfg, "volatility", default=None)
    if vol_cfg is not None:
        if not isinstance(vol_cfg, dict):
            raise ConfigError("model.volatility must be a mapping")
        enabled = _as_bool(_get(vol_cfg, "enabled", default=True), key="model.volatility.enabled")
        if enabled:
            vol_spec = VolatilitySpec(
                enabled=True,
                epsilon=_as_float(_get(vol_cfg, "epsilon", default=1e-4), key="model.volatility.epsilon"),
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
            )

    ss_spec: SteadyStateSpec | None = None
    ss_cfg = _get(model_cfg, "steady_state", default=None)
    if ss_cfg is not None:
        if not isinstance(ss_cfg, dict):
            raise ConfigError("model.steady_state must be a mapping")

        ss_enabled = _as_bool(_get(ss_cfg, "enabled", default=True), key="model.steady_state.enabled")
        if ss_enabled:
            mu0_raw = _get(ss_cfg, "mu0", required=True)
            if not isinstance(mu0_raw, list) or not all(
                isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) for v in mu0_raw
            ):
                raise ConfigError("model.steady_state.mu0 must be a list of numbers")
            mu0 = np.asarray([float(v) for v in mu0_raw], dtype=float)
            if mu0.shape != (dataset.N,):
                raise ConfigError("model.steady_state.mu0 must have length N")

            v0_mu_raw = _get(ss_cfg, "v0_mu", required=True)
            if isinstance(v0_mu_raw, list):
                if not all(
                    isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) for v in v0_mu_raw
                ):
                    raise ConfigError("model.steady_state.v0_mu must be a number or list of numbers")
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
        for name in ["spike_var", "slab_var", "inclusion_prob", "intercept_slab_var", "fix_intercept"]:
            if name in hyp:
                kwargs2[name] = hyp[name]
        return PriorSpec.from_ssvs(k=k, n=dataset.N, include_intercept=model.include_intercept, **kwargs2)

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

        return PriorSpec.from_blasso(k=k, n=dataset.N, include_intercept=model.include_intercept, **kwargs3)

    if family_l == "dl":
        hyp = _get(prior_cfg, "dl", default={})
        if not isinstance(hyp, dict):
            raise ConfigError("prior.dl must be a mapping")

        kwargs4: dict[str, Any] = {}
        for name in ["abeta", "dl_scaler"]:
            if name in hyp:
                kwargs4[name] = hyp[name]

        return PriorSpec.from_dl(k=k, n=dataset.N, include_intercept=model.include_intercept, **kwargs4)

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
    if not isinstance(horizons, list) or not all(isinstance(v, (int, np.integer)) for v in horizons):
        raise ConfigError("forecast.horizons must be a list[int]")
    horizons_i = [int(v) for v in horizons]
    if not horizons_i or any(h < 1 for h in horizons_i):
        raise ConfigError("forecast.horizons must contain positive integers")

    draws = _as_int(_get(fc_cfg, "draws", default=1000), key="forecast.draws", min_value=1)

    q = _get(fc_cfg, "quantile_levels", default=[0.1, 0.5, 0.9])
    if not isinstance(q, list) or not all(isinstance(v, (float, int, np.floating, np.integer)) for v in q):
        raise ConfigError("forecast.quantile_levels must be a list[float]")
    qf = [float(v) for v in q]

    return {"horizons": horizons_i, "draws": draws, "quantile_levels": qf}


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
    if window is None:
        window_i: int | None = None
    else:
        window_i = _as_int(window, key="backtest.window", min_value=1)

    if mode_l == "rolling" and window_i is None:
        raise ConfigError("backtest.window is required when backtest.mode='rolling'")
    if mode_l == "expanding" and window_i is not None:
        raise ConfigError("backtest.window must be null/omitted when backtest.mode='expanding'")

    min_obs = _as_int(_get(bt_cfg, "min_obs", default=max(20, model.p + 1)), key="backtest.min_obs", min_value=1)
    if min_obs < model.p + 1:
        raise ConfigError("backtest.min_obs must be >= model.p + 1")

    if window_i is not None and window_i < min_obs:
        raise ConfigError("backtest.window must be >= backtest.min_obs")

    step = _as_int(_get(bt_cfg, "step", default=1), key="backtest.step", min_value=1)

    horizons = _get(bt_cfg, "horizons", required=True)
    if not isinstance(horizons, list) or not all(isinstance(v, (int, np.integer)) for v in horizons):
        raise ConfigError("backtest.horizons must be a list[int]")
    horizons_i = [int(v) for v in horizons]
    if not horizons_i or any(h < 1 for h in horizons_i):
        raise ConfigError("backtest.horizons must contain positive integers")

    draws = _as_int(_get(bt_cfg, "draws", default=500), key="backtest.draws", min_value=1)

    q = _get(bt_cfg, "quantile_levels", default=[0.1, 0.5, 0.9])
    if not isinstance(q, list) or not all(isinstance(v, (float, int, np.floating, np.integer)) for v in q):
        raise ConfigError("backtest.quantile_levels must be a list[float]")
    qf = [float(v) for v in q]

    return {
        "mode": mode_l,
        "window": window_i,
        "min_obs": min_obs,
        "step": step,
        "horizons": horizons_i,
        "draws": draws,
        "quantile_levels": qf,
    }


def build_evaluation_config(cfg: dict[str, Any], *, variables: list[str], horizons: list[int]) -> dict[str, Any]:
    ev_cfg = _get(cfg, "evaluation", default={})
    if not isinstance(ev_cfg, dict):
        raise ConfigError("evaluation must be a mapping")

    cov_cfg = _get(ev_cfg, "coverage", default={})
    if not isinstance(cov_cfg, dict):
        raise ConfigError("evaluation.coverage must be a mapping")
    cov_enabled = _as_bool(_get(cov_cfg, "enabled", default=True), key="evaluation.coverage.enabled")
    cov_intervals = _get(cov_cfg, "intervals", default=[0.5, 0.8, 0.9])
    if not isinstance(cov_intervals, list) or not all(isinstance(v, (float, int, np.floating, np.integer)) for v in cov_intervals):
        raise ConfigError("evaluation.coverage.intervals must be a list[float]")
    cov_intervals_f = [float(v) for v in cov_intervals]
    cov_use_latent = _as_bool(_get(cov_cfg, "use_latent", default=False), key="evaluation.coverage.use_latent")

    pit_cfg = _get(ev_cfg, "pit", default={})
    if not isinstance(pit_cfg, dict):
        raise ConfigError("evaluation.pit must be a mapping")
    pit_enabled = _as_bool(_get(pit_cfg, "enabled", default=False), key="evaluation.pit.enabled")
    pit_bins = _as_int(_get(pit_cfg, "bins", default=10), key="evaluation.pit.bins", min_value=2)
    pit_use_latent = _as_bool(_get(pit_cfg, "use_latent", default=False), key="evaluation.pit.use_latent")
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
            raise ConfigError(f"evaluation.pit.horizons contains horizon not in backtest.horizons: {h}")

    crps_cfg = _get(ev_cfg, "crps", default={})
    if not isinstance(crps_cfg, dict):
        raise ConfigError("evaluation.crps must be a mapping")
    crps_enabled = _as_bool(_get(crps_cfg, "enabled", default=True), key="evaluation.crps.enabled")
    crps_use_latent = _as_bool(_get(crps_cfg, "use_latent", default=False), key="evaluation.crps.use_latent")

    metrics_table = _as_bool(_get(ev_cfg, "metrics_table", default=True), key="evaluation.metrics_table")

    return {
        "coverage": {"enabled": cov_enabled, "intervals": cov_intervals_f, "use_latent": cov_use_latent},
        "pit": {
            "enabled": pit_enabled,
            "bins": pit_bins,
            "variables": list(pit_vars),
            "horizons": pit_h_i,
            "use_latent": pit_use_latent,
        },
        "crps": {"enabled": crps_enabled, "use_latent": crps_use_latent},
        "metrics_table": metrics_table,
    }


def validate_config(cfg: dict[str, Any]) -> None:
    ds, model, _prior, _sampler, _rng, _fc_cfg = _prepare_from_config(cfg, emit=None)

    if "backtest" in cfg:
        bt = build_backtest_config(cfg, model=model)
        build_evaluation_config(cfg, variables=list(ds.variables), horizons=list(bt["horizons"]))


def _save_fit_npz(path: Path, fit_res: FitResult) -> None:
    np.savez_compressed(
        path,
        variables=np.asarray(fit_res.dataset.variables, dtype=object),
        time_index=np.asarray(fit_res.dataset.time_index.to_numpy(), dtype="datetime64[ns]"),
        values=fit_res.dataset.values,
        beta_draws=fit_res.beta_draws,
        sigma_draws=fit_res.sigma_draws,
        latent_draws=fit_res.latent_draws,
        h_draws=fit_res.h_draws,
        h0_draws=fit_res.h0_draws,
        sigma_eta2_draws=fit_res.sigma_eta2_draws,
        gamma_draws=fit_res.gamma_draws,
    )


def _save_forecast_npz(path: Path, fc: ForecastResult) -> None:
    keys = {f"q_{q}": arr for q, arr in fc.quantiles.items()}
    np.savez_compressed(
        path,
        variables=np.asarray(fc.variables, dtype=object),
        horizons=np.asarray(fc.horizons, dtype=int),
        draws=fc.draws,
        mean=fc.mean,
        latent_draws=fc.latent_draws,
        **keys,
    )


def run_from_config(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    validate_only: bool = False,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> RunArtifacts | None:
    t0_total = time.perf_counter()

    def emit(event: str, payload: dict[str, Any]) -> None:
        if progress is not None:
            progress(event, payload)

    emit("stage_start", {"name": "load_config"})
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    emit("stage_end", {"name": "load_config", "elapsed_s": time.perf_counter() - t0})

    emit("stage_start", {"name": "validate_config"})
    t0 = time.perf_counter()
    ds, model, prior, sampler, rng, fc_cfg = _prepare_from_config(cfg, emit=emit)
    emit("stage_end", {"name": "validate_config", "elapsed_s": time.perf_counter() - t0})
    if validate_only:
        emit("validate_end", {"elapsed_s": time.perf_counter() - t0_total})
        return None

    emit("stage_start", {"name": "fit"})
    t0 = time.perf_counter()
    fit_res = fit(ds, model, prior, sampler, rng=rng)
    emit("stage_end", {"name": "fit", "elapsed_s": time.perf_counter() - t0})

    fc_res: ForecastResult | None = None
    if fc_cfg is not None:
        emit("stage_start", {"name": "forecast"})
        t0 = time.perf_counter()
        fc_res = forecast(
            fit_res,
            horizons=fc_cfg["horizons"],
            draws=fc_cfg["draws"],
            quantile_levels=fc_cfg["quantile_levels"],
            rng=rng,
        )
        emit("stage_end", {"name": "forecast", "elapsed_s": time.perf_counter() - t0})

    output_cfg = _get(cfg, "output", default={})
    if not isinstance(output_cfg, dict):
        raise ConfigError("output must be a mapping")

    emit("stage_start", {"name": "prepare_output"})
    t0 = time.perf_counter()
    out = Path(out_dir) if out_dir is not None else Path(_get(output_cfg, "out_dir", default="outputs"))
    out.mkdir(parents=True, exist_ok=True)
    emit(
        "summary",
        {
            "kind": "output",
            "out_dir": str(out),
            "save_fit": bool(_get(output_cfg, "save_fit", default=True)),
            "save_forecast": bool(_get(output_cfg, "save_forecast", default=True)),
            "save_plots": bool(_get(output_cfg, "save_plots", default=False)),
        },
    )
    emit("stage_end", {"name": "prepare_output", "elapsed_s": time.perf_counter() - t0})

    emit("stage_start", {"name": "write_artifacts"})
    t0_write = time.perf_counter()

    cfg_out = Path(out / "config.yml")
    cfg_out.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    emit(
        "artifact",
        {"path": str(cfg_out), "bytes": int(cfg_out.stat().st_size), "kind": "config"},
    )

    if _as_bool(_get(output_cfg, "save_fit", default=True), key="output.save_fit"):
        fit_path = out / "fit_result.npz"
        _save_fit_npz(fit_path, fit_res)
        emit(
            "artifact",
            {"path": str(fit_path), "bytes": int(fit_path.stat().st_size), "kind": "fit"},
        )

    if fc_res is not None and _as_bool(_get(output_cfg, "save_forecast", default=True), key="output.save_forecast"):
        fc_path = out / "forecast_result.npz"
        _save_forecast_npz(fc_path, fc_res)
        emit(
            "artifact",
            {"path": str(fc_path), "bytes": int(fc_path.stat().st_size), "kind": "forecast"},
        )

    if _as_bool(_get(output_cfg, "save_plots", default=False), key="output.save_plots"):
        plots_cfg = _get(cfg, "plots", default={})
        if not isinstance(plots_cfg, dict):
            raise ConfigError("plots must be a mapping")
        vars_to_plot = _as_str_list(_get(plots_cfg, "variables", default=ds.variables), key="plots.variables")
        bands = _get(plots_cfg, "bands", default=[0.1, 0.9])
        if not isinstance(bands, list) or len(bands) != 2:
            raise ConfigError("plots.bands must be a list of two floats")
        bands_t = (float(bands[0]), float(bands[1]))

        from .plotting import plot_forecast_fanchart, plot_shadow_rate, plot_volatility

        for v in vars_to_plot:
            fig, _ax = plot_shadow_rate(fit_res, var=v, bands=bands_t)
            p_shadow = out / f"shadow_rate_{v}.png"
            fig.savefig(p_shadow, dpi=200, bbox_inches="tight")
            emit(
                "artifact",
                {"path": str(p_shadow), "bytes": int(p_shadow.stat().st_size), "kind": "plot"},
            )

            if fit_res.h_draws is not None:
                fig, _ax = plot_volatility(fit_res, var=v, bands=bands_t)
                p_vol = out / f"volatility_{v}.png"
                fig.savefig(p_vol, dpi=200, bbox_inches="tight")
                emit(
                    "artifact",
                    {"path": str(p_vol), "bytes": int(p_vol.stat().st_size), "kind": "plot"},
                )

            if fc_res is not None:
                fig, _ax = plot_forecast_fanchart(fc_res, var=v, bands=bands_t, use_latent=False)
                p_fc_obs = out / f"forecast_fan_{v}_observed.png"
                fig.savefig(p_fc_obs, dpi=200, bbox_inches="tight")
                emit(
                    "artifact",
                    {"path": str(p_fc_obs), "bytes": int(p_fc_obs.stat().st_size), "kind": "plot"},
                )

                if fc_res.latent_draws is not None:
                    fig, _ax = plot_forecast_fanchart(fc_res, var=v, bands=bands_t, use_latent=True)
                    p_fc_sh = out / f"forecast_fan_{v}_shadow.png"
                    fig.savefig(p_fc_sh, dpi=200, bbox_inches="tight")
                    emit(
                        "artifact",
                        {"path": str(p_fc_sh), "bytes": int(p_fc_sh.stat().st_size), "kind": "plot"},
                    )

    emit("stage_end", {"name": "write_artifacts", "elapsed_s": time.perf_counter() - t0_write})
    emit("run_end", {"elapsed_s": time.perf_counter() - t0_total})
    return RunArtifacts(fit_result=fit_res, forecast_result=fc_res)


def backtest_from_config(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    t0_total = time.perf_counter()

    def emit(event: str, payload: dict[str, Any]) -> None:
        if progress is not None:
            progress(event, payload)

    emit("stage_start", {"name": "load_config"})
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    emit("stage_end", {"name": "load_config", "elapsed_s": time.perf_counter() - t0})

    emit("stage_start", {"name": "validate_config"})
    t0 = time.perf_counter()
    ds_full, model, _prior0, sampler, rng, _fc_cfg = _prepare_from_config(cfg, emit=emit)
    emit("stage_end", {"name": "validate_config", "elapsed_s": time.perf_counter() - t0})

    bt = build_backtest_config(cfg, model=model)
    ev = build_evaluation_config(cfg, variables=list(ds_full.variables), horizons=list(bt["horizons"]))

    output_cfg = _get(cfg, "output", default={})
    if not isinstance(output_cfg, dict):
        raise ConfigError("output must be a mapping")
    save_plots = bool(_get(output_cfg, "save_plots", default=True))
    save_forecasts = bool(_get(output_cfg, "save_forecasts", default=False))

    emit("stage_start", {"name": "prepare_output"})
    t0 = time.perf_counter()
    out = Path(out_dir) if out_dir is not None else Path(_get(output_cfg, "out_dir", default="outputs/backtest"))
    out.mkdir(parents=True, exist_ok=True)
    emit(
        "summary",
        {
            "kind": "output",
            "out_dir": str(out),
            "save_fit": False,
            "save_forecast": False,
            "save_plots": bool(save_plots),
            "save_forecasts": bool(save_forecasts),
        },
    )
    emit("stage_end", {"name": "prepare_output", "elapsed_s": time.perf_counter() - t0})

    emit("stage_start", {"name": "write_artifacts"})
    t0_write = time.perf_counter()

    cfg_out = Path(out / "config.yml")
    cfg_out.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    emit("artifact", {"path": str(cfg_out), "bytes": int(cfg_out.stat().st_size), "kind": "config"})

    mode = str(bt["mode"])
    window_i = bt["window"]
    min_obs = int(bt["min_obs"])
    step = int(bt["step"])
    horizons = list(bt["horizons"])
    max_h = int(max(horizons))
    pred_draws = int(bt["draws"])
    q_levels = list(bt["quantile_levels"])

    if ds_full.T <= max_h:
        raise ConfigError("dataset is too short for requested backtest horizons")

    first_origin_end = min_obs - 1
    last_origin_end = ds_full.T - max_h - 1
    if last_origin_end < first_origin_end:
        raise ConfigError("backtest settings imply zero feasible forecast origins")

    origins = list(range(first_origin_end, last_origin_end + 1, step))
    k_orig = int(len(origins))
    n = int(ds_full.N)

    y_true = np.full((k_orig, max_h, n), np.nan, dtype=float)
    forecasts: list[ForecastResult] = []

    emit(
        "summary",
        {
            "kind": "backtest",
            "mode": mode,
            "window": window_i,
            "min_obs": min_obs,
            "step": step,
            "horizons": horizons,
            "origins": k_orig,
            "draws": pred_draws,
        },
    )

    fc_dir = out / "forecasts"
    if save_forecasts:
        fc_dir.mkdir(parents=True, exist_ok=True)

    for i, origin_end in enumerate(origins):
        t0_origin = time.perf_counter()

        if mode == "expanding":
            train_start = 0
        else:
            assert window_i is not None
            train_start = max(0, int(origin_end - window_i + 1))
        train_end_excl = int(origin_end + 1)

        train_values = ds_full.values[train_start:train_end_excl, :]
        train_index = ds_full.time_index[train_start:train_end_excl]
        train_ds = Dataset.from_arrays(values=train_values, variables=ds_full.variables, time_index=train_index)

        prior_i = build_prior(cfg, dataset=train_ds, model=model)
        fit_res = fit(train_ds, model, prior_i, sampler, rng=rng)
        fc_res = forecast(
            fit_res,
            horizons=horizons,
            draws=pred_draws,
            quantile_levels=q_levels,
            rng=rng,
        )
        forecasts.append(fc_res)

        y_true[i, :, :] = ds_full.values[origin_end + 1 : origin_end + 1 + max_h, :]

        if save_forecasts:
            p = fc_dir / f"origin_{origin_end:04d}.npz"
            _save_forecast_npz(p, fc_res)
            emit("artifact", {"path": str(p), "bytes": int(p.stat().st_size), "kind": "forecast"})

        emit(
            "backtest_origin",
            {
                "i": i,
                "k": k_orig,
                "origin_end": int(origin_end),
                "train_start": int(train_start),
                "train_T": int(train_ds.T),
                "elapsed_s": time.perf_counter() - t0_origin,
            },
        )

    intervals = list(ev["coverage"]["intervals"])

    metrics_path = out / "metrics.csv"
    if bool(ev["metrics_table"]):
        rows: list[dict[str, Any]] = []
        for j, vname in enumerate(ds_full.variables):
            for h in range(1, max_h + 1):
                y = y_true[:, h - 1, j]
                mu = np.asarray([fc.mean[h - 1, j] for fc in forecasts], dtype=float)
                e = mu - y

                sims_list = [
                    (fc.latent_draws if (bool(ev["crps"]["use_latent"]) and fc.latent_draws is not None) else fc.draws)[:, h - 1, j]
                    for fc in forecasts
                ]
                crps_vals = np.asarray([float("nan") if np.isnan(y[i2]) else float(crps_draws(y[i2], sims_list[i2])) for i2 in range(k_orig)], dtype=float)

                row: dict[str, Any] = {
                    "variable": vname,
                    "horizon": h,
                    "crps": float(np.nanmean(crps_vals)),
                    "rmse": float(rmse(e, axis=0)),
                    "mae": float(mae(e, axis=0)),
                }

                for c in intervals:
                    qlo = 0.5 - 0.5 * float(c)
                    qhi = 0.5 + 0.5 * float(c)
                    hit = np.empty(k_orig, dtype=float)
                    for i2, fc in enumerate(forecasts):
                        sims = fc.latent_draws if (bool(ev["coverage"]["use_latent"]) and fc.latent_draws is not None) else fc.draws
                        lo = float(np.quantile(sims[:, h - 1, j], q=qlo))
                        hi = float(np.quantile(sims[:, h - 1, j], q=qhi))
                        yi = float(y_true[i2, h - 1, j])
                        hit[i2] = float(lo <= yi <= hi)
                    row[f"coverage_{int(round(100 * float(c)))}"] = float(np.nanmean(hit))
                rows.append(row)

        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else ["variable", "horizon", "crps", "rmse", "mae"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        emit("artifact", {"path": str(metrics_path), "bytes": int(metrics_path.stat().st_size), "kind": "table"})

    if save_plots:
        from .plotting import plot_crps_by_horizon, plot_forecast_coverage, plot_pit_histogram

        cov_all = out / "coverage_all.png"
        fig, _ax = plot_forecast_coverage(
            forecasts,
            y_true,
            intervals=intervals,
            horizons=horizons,
            var=None,
            use_latent=bool(ev["coverage"]["use_latent"]),
        )
        fig.savefig(cov_all, dpi=200, bbox_inches="tight")
        emit("artifact", {"path": str(cov_all), "bytes": int(cov_all.stat().st_size), "kind": "plot"})

        for vname in ds_full.variables:
            p_cov = out / f"coverage_{vname}.png"
            fig, _ax = plot_forecast_coverage(
                forecasts,
                y_true,
                intervals=intervals,
                horizons=horizons,
                var=vname,
                use_latent=bool(ev["coverage"]["use_latent"]),
            )
            fig.savefig(p_cov, dpi=200, bbox_inches="tight")
            emit("artifact", {"path": str(p_cov), "bytes": int(p_cov.stat().st_size), "kind": "plot"})

        if bool(ev["crps"]["enabled"]):
            crps_all = out / "crps_by_horizon.png"
            fig, _ax = plot_crps_by_horizon(
                forecasts,
                y_true,
                horizons=horizons,
                var=None,
                use_latent=bool(ev["crps"]["use_latent"]),
            )
            fig.savefig(crps_all, dpi=200, bbox_inches="tight")
            emit("artifact", {"path": str(crps_all), "bytes": int(crps_all.stat().st_size), "kind": "plot"})

        if bool(ev["pit"]["enabled"]):
            for vname in list(ev["pit"]["variables"]):
                for h in list(ev["pit"]["horizons"]):
                    p_pit = out / f"pit_{vname}_h{int(h)}.png"
                    fig, _ax = plot_pit_histogram(
                        forecasts,
                        y_true,
                        var=str(vname),
                        horizon=int(h),
                        bins=int(ev["pit"]["bins"]),
                        use_latent=bool(ev["pit"]["use_latent"]),
                    )
                    fig.savefig(p_pit, dpi=200, bbox_inches="tight")
                    emit("artifact", {"path": str(p_pit), "bytes": int(p_pit.stat().st_size), "kind": "plot"})

    summary = {
        "mode": mode,
        "window": window_i,
        "min_obs": min_obs,
        "step": step,
        "horizons": horizons,
        "origins": k_orig,
        "dataset_T": int(ds_full.T),
        "dataset_N": int(ds_full.N),
        "elapsed_s": float(time.perf_counter() - t0_total),
    }
    summary_path = out / "backtest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit("artifact", {"path": str(summary_path), "bytes": int(summary_path.stat().st_size), "kind": "meta"})

    emit("stage_end", {"name": "write_artifacts", "elapsed_s": time.perf_counter() - t0_write})
    emit("backtest_end", {"elapsed_s": time.perf_counter() - t0_total})
