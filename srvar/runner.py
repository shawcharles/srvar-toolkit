from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import fit, forecast
from .artifacts import save_fit_npz, save_forecast_npz
from .backtest import backtest_from_config
from .config import (
    ConfigError,
    _as_bool,
    _as_str_list,
    _get,
    _prepare_from_config,
    _validate_backtest_prior_at_first_origin,
    build_backtest_config,
    build_evaluation_config,
    build_forecast_config,
    build_model,
    build_prior,
    build_sampler,
    load_config,
    load_dataset_from_csv,
    validate_config,
)
from .results import FitResult, ForecastResult

__all__ = [
    "ConfigError",
    "RunArtifacts",
    "backtest_from_config",
    "build_backtest_config",
    "build_evaluation_config",
    "build_forecast_config",
    "build_model",
    "build_prior",
    "build_sampler",
    "load_config",
    "load_dataset_from_csv",
    "run_from_config",
    "validate_config",
]


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    fit_result: FitResult
    forecast_result: ForecastResult | None


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
    validate_backtest = validate_only and "backtest" in cfg
    ds, model, prior, sampler, rng, fc_cfg = _prepare_from_config(
        cfg, emit=emit, build_full_prior=not validate_backtest
    )
    emit("stage_end", {"name": "validate_config", "elapsed_s": time.perf_counter() - t0})
    if validate_only:
        if validate_backtest:
            bt = build_backtest_config(cfg, model=model)
            build_evaluation_config(
                cfg, variables=list(ds.variables), horizons=list(bt["horizons"])
            )
            _validate_backtest_prior_at_first_origin(cfg, dataset=ds, model=model, bt=bt)
        emit("validate_end", {"elapsed_s": time.perf_counter() - t0_total})
        return None

    assert prior is not None
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
            stationarity=str(fc_cfg.get("stationarity", "allow")),
            stationarity_tol=float(fc_cfg.get("stationarity_tol", 1e-10)),
            stationarity_max_draws=fc_cfg.get("stationarity_max_draws", None),
            rng=rng,
        )
        emit("stage_end", {"name": "forecast", "elapsed_s": time.perf_counter() - t0})

    output_cfg = _get(cfg, "output", default={})
    if not isinstance(output_cfg, dict):
        raise ConfigError("output must be a mapping")

    emit("stage_start", {"name": "prepare_output"})
    t0 = time.perf_counter()
    out = (
        Path(out_dir)
        if out_dir is not None
        else Path(_get(output_cfg, "out_dir", default="outputs"))
    )
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
    emit("artifact", {"path": str(cfg_out), "bytes": int(cfg_out.stat().st_size), "kind": "config"})

    if _as_bool(_get(output_cfg, "save_fit", default=True), key="output.save_fit"):
        fit_path = out / "fit_result.npz"
        save_fit_npz(fit_path, fit_res)
        emit(
            "artifact",
            {"path": str(fit_path), "bytes": int(fit_path.stat().st_size), "kind": "fit"},
        )

    if fc_res is not None and _as_bool(
        _get(output_cfg, "save_forecast", default=True), key="output.save_forecast"
    ):
        fc_path = out / "forecast_result.npz"
        save_forecast_npz(fc_path, fc_res)
        emit(
            "artifact",
            {"path": str(fc_path), "bytes": int(fc_path.stat().st_size), "kind": "forecast"},
        )

    if _as_bool(_get(output_cfg, "save_plots", default=False), key="output.save_plots"):
        plots_cfg = _get(cfg, "plots", default={})
        if not isinstance(plots_cfg, dict):
            raise ConfigError("plots must be a mapping")
        vars_to_plot = _as_str_list(
            _get(plots_cfg, "variables", default=ds.variables), key="plots.variables"
        )
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
                        {
                            "path": str(p_fc_sh),
                            "bytes": int(p_fc_sh.stat().st_size),
                            "kind": "plot",
                        },
                    )

    emit("stage_end", {"name": "write_artifacts", "elapsed_s": time.perf_counter() - t0_write})
    emit("run_end", {"elapsed_s": time.perf_counter() - t0_total})
    return RunArtifacts(fit_result=fit_res, forecast_result=fc_res)
