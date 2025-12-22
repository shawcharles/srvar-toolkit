from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .api import fit, forecast
from .data.dataset import Dataset
from .elb import ElbSpec
from .results import FitResult, ForecastResult
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .sv import VolatilitySpec


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    fit_result: FitResult
    forecast_result: ForecastResult | None


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


def build_model(cfg: dict[str, Any]) -> ModelSpec:
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
            elb_spec = ElbSpec(bound=bound, applies_to=applies_to, tol=tol, enabled=True)

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

    return ModelSpec(p=p, include_intercept=include_intercept, elb=elb_spec, volatility=vol_spec)


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

    raise ConfigError("prior.family must be one of: niw, ssvs")


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


def validate_config(cfg: dict[str, Any]) -> None:
    ds = load_dataset_from_csv(cfg)
    model = build_model(cfg)

    if model.elb is not None and model.elb.enabled:
        missing = [v for v in model.elb.applies_to if v not in ds.variables]
        if missing:
            raise ConfigError(f"model.elb.applies_to not found in dataset.variables: {missing}")

    build_prior(cfg, dataset=ds, model=model)
    build_sampler(cfg)

    fc = build_forecast_config(cfg)
    if fc is not None:
        max_h = max(fc["horizons"])
        if max_h < 1:
            raise ConfigError("forecast.horizons must include positive integers")


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
) -> RunArtifacts | None:
    cfg = load_config(config_path)
    validate_config(cfg)
    if validate_only:
        return None

    ds = load_dataset_from_csv(cfg)
    model = build_model(cfg)
    prior = build_prior(cfg, dataset=ds, model=model)
    sampler, rng = build_sampler(cfg)

    fit_res = fit(ds, model, prior, sampler, rng=rng)

    fc_cfg = build_forecast_config(cfg)
    fc_res: ForecastResult | None = None
    if fc_cfg is not None:
        fc_res = forecast(
            fit_res,
            horizons=fc_cfg["horizons"],
            draws=fc_cfg["draws"],
            quantile_levels=fc_cfg["quantile_levels"],
            rng=rng,
        )

    output_cfg = _get(cfg, "output", default={})
    if not isinstance(output_cfg, dict):
        raise ConfigError("output must be a mapping")

    out = Path(out_dir) if out_dir is not None else Path(_get(output_cfg, "out_dir", default="outputs"))
    out.mkdir(parents=True, exist_ok=True)

    Path(out / "config.yml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    if _as_bool(_get(output_cfg, "save_fit", default=True), key="output.save_fit"):
        _save_fit_npz(out / "fit_result.npz", fit_res)

    if fc_res is not None and _as_bool(_get(output_cfg, "save_forecast", default=True), key="output.save_forecast"):
        _save_forecast_npz(out / "forecast_result.npz", fc_res)

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
            fig.savefig(out / f"shadow_rate_{v}.png", dpi=200, bbox_inches="tight")

            if fit_res.h_draws is not None:
                fig, _ax = plot_volatility(fit_res, var=v, bands=bands_t)
                fig.savefig(out / f"volatility_{v}.png", dpi=200, bbox_inches="tight")

            if fc_res is not None:
                fig, _ax = plot_forecast_fanchart(fc_res, var=v, bands=bands_t, use_latent=False)
                fig.savefig(out / f"forecast_fan_{v}_observed.png", dpi=200, bbox_inches="tight")

                if fc_res.latent_draws is not None:
                    fig, _ax = plot_forecast_fanchart(fc_res, var=v, bands=bands_t, use_latent=True)
                    fig.savefig(out / f"forecast_fan_{v}_shadow.png", dpi=200, bbox_inches="tight")

    return RunArtifacts(fit_result=fit_res, forecast_result=fc_res)
