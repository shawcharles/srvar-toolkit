from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .api import fit, forecast
from .artifacts import save_forecast_npz
from .config import (
    ConfigError,
    _get,
    _prepare_from_config,
    build_backtest_config,
    build_evaluation_config,
    build_prior,
)
from .data.dataset import Dataset
from .evaluation import MetricsAccumulator, compute_metrics_rows, prepare_evaluation_inputs
from .results import ForecastResult


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
    from .config import load_config

    cfg = load_config(config_path)
    emit("stage_end", {"name": "load_config", "elapsed_s": time.perf_counter() - t0})

    emit("stage_start", {"name": "validate_config"})
    t0 = time.perf_counter()
    ds_full, model, _prior0, sampler, rng, _fc_cfg = _prepare_from_config(cfg, emit=emit)
    emit("stage_end", {"name": "validate_config", "elapsed_s": time.perf_counter() - t0})

    bt = build_backtest_config(cfg, model=model)
    ev = build_evaluation_config(
        cfg, variables=list(ds_full.variables), horizons=list(bt["horizons"])
    )

    output_cfg = _get(cfg, "output", default={})
    if not isinstance(output_cfg, dict):
        raise ConfigError("output must be a mapping")
    save_plots = bool(_get(output_cfg, "save_plots", default=True))
    save_forecasts = bool(_get(output_cfg, "save_forecasts", default=False))
    store_forecasts_in_memory = bool(
        _get(output_cfg, "store_forecasts_in_memory", default=save_plots)
    )
    if save_plots and not store_forecasts_in_memory:
        raise ConfigError(
            "output.store_forecasts_in_memory must be true when output.save_plots is true "
            "(streaming plots are not supported yet)"
        )

    emit("stage_start", {"name": "prepare_output"})
    t0 = time.perf_counter()
    out = (
        Path(out_dir)
        if out_dir is not None
        else Path(_get(output_cfg, "out_dir", default="outputs/backtest"))
    )
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
            "store_forecasts_in_memory": bool(store_forecasts_in_memory),
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
    stationarity = str(bt.get("stationarity", "allow"))
    stationarity_tol = float(bt.get("stationarity_tol", 1e-10))
    stationarity_max_draws = bt.get("stationarity_max_draws", None)

    if ds_full.T <= max_h:
        raise ConfigError("dataset is too short for requested backtest horizons")

    first_origin_end = min_obs - 1
    last_origin_end = ds_full.T - max_h - 1
    if last_origin_end < first_origin_end:
        raise ConfigError("backtest settings imply zero feasible forecast origins")

    origin_start = bt.get("origin_start")
    origin_end = bt.get("origin_end")
    if origin_start is not None or origin_end is not None:
        if not isinstance(ds_full.time_index, pd.DatetimeIndex):
            raise ConfigError(
                "backtest.origin_start/end requires a datetime index (data.date_column parsed as dates)"
            )

        if origin_start is not None:
            ts = pd.to_datetime(origin_start)
            try:
                start_i = int(ds_full.time_index.get_loc(ts))
            except KeyError as e:
                raise ConfigError(
                    f"backtest.origin_start not found in dataset index: {origin_start}"
                ) from e
            first_origin_end = max(first_origin_end, start_i)

        if origin_end is not None:
            ts = pd.to_datetime(origin_end)
            try:
                end_i = int(ds_full.time_index.get_loc(ts))
            except KeyError as e:
                raise ConfigError(
                    f"backtest.origin_end not found in dataset index: {origin_end}"
                ) from e
            last_origin_end = min(last_origin_end, end_i)

        if last_origin_end < first_origin_end:
            raise ConfigError("backtest.origin_start/end implies zero feasible forecast origins")

    origins = list(range(first_origin_end, last_origin_end + 1, step))
    k_orig = int(len(origins))
    n = int(ds_full.N)

    y_true: np.ndarray | None
    forecasts: list[ForecastResult] | None
    acc: MetricsAccumulator | None
    if store_forecasts_in_memory:
        y_true = np.full((k_orig, max_h, n), np.nan, dtype=float)
        forecasts = []
        acc = None
    else:
        y_true = None
        forecasts = None
        acc = (
            MetricsAccumulator(variables=list(ds_full.variables), max_h=max_h, evaluation=ev)
            if bool(ev["metrics_table"])
            else None
        )

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

    for i, origin_end_i in enumerate(origins):
        t0_origin = time.perf_counter()

        if mode == "expanding":
            train_start = 0
        else:
            assert window_i is not None
            train_start = max(0, int(origin_end_i - window_i + 1))
        train_end_excl = int(origin_end_i + 1)

        train_values = ds_full.values[train_start:train_end_excl, :]
        train_index = ds_full.time_index[train_start:train_end_excl]
        train_ds = Dataset.from_arrays(
            values=train_values, variables=ds_full.variables, time_index=train_index
        )

        prior_i = build_prior(cfg, dataset=train_ds, model=model)
        fit_res = fit(train_ds, model, prior_i, sampler, rng=rng)
        fc_res = forecast(
            fit_res,
            horizons=horizons,
            draws=pred_draws,
            quantile_levels=q_levels,
            stationarity=stationarity,
            stationarity_tol=stationarity_tol,
            stationarity_max_draws=stationarity_max_draws,
            rng=rng,
        )
        y_true_i = ds_full.values[origin_end_i + 1 : origin_end_i + 1 + max_h, :]
        if store_forecasts_in_memory:
            assert y_true is not None
            assert forecasts is not None
            forecasts.append(fc_res)
            y_true[i, :, :] = y_true_i
        elif acc is not None:
            acc.update(forecast=fc_res, y_true=y_true_i)

        if save_forecasts:
            p = fc_dir / f"origin_{origin_end_i:04d}.npz"
            save_forecast_npz(p, fc_res)
            emit("artifact", {"path": str(p), "bytes": int(p.stat().st_size), "kind": "forecast"})

        emit(
            "backtest_origin",
            {
                "i": i,
                "k": k_orig,
                "origin_end": int(origin_end_i),
                "train_start": int(train_start),
                "train_T": int(train_ds.T),
                "elapsed_s": time.perf_counter() - t0_origin,
            },
        )

    metrics_path = out / "metrics.csv"
    forecasts_eval: list[ForecastResult]
    y_true_eval: np.ndarray
    if store_forecasts_in_memory:
        assert y_true is not None
        assert forecasts is not None
        y_true_eval, forecasts_eval = prepare_evaluation_inputs(
            y_true=y_true,
            forecasts=list(forecasts),
            variables=list(ds_full.variables),
            evaluation=ev,
        )
        if bool(ev["metrics_table"]):
            rows = compute_metrics_rows(
                forecasts=forecasts_eval,
                y_true=y_true_eval,
                variables=list(ds_full.variables),
                evaluation=ev,
            )

            with metrics_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = (
                    list(rows[0].keys()) if rows else ["variable", "horizon", "crps", "rmse", "mae"]
                )
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            emit(
                "artifact",
                {
                    "path": str(metrics_path),
                    "bytes": int(metrics_path.stat().st_size),
                    "kind": "table",
                },
            )
    else:
        forecasts_eval = []
        y_true_eval = np.empty((0, 0, 0), dtype=float)
        if bool(ev["metrics_table"]) and acc is not None:
            rows = acc.rows()
            with metrics_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = (
                    list(rows[0].keys()) if rows else ["variable", "horizon", "crps", "rmse", "mae"]
                )
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            emit(
                "artifact",
                {
                    "path": str(metrics_path),
                    "bytes": int(metrics_path.stat().st_size),
                    "kind": "table",
                },
            )

    coverage_enabled = bool(ev["coverage"]["enabled"])
    crps_enabled = bool(ev["crps"]["enabled"])
    intervals = list(ev["coverage"]["intervals"]) if coverage_enabled else []

    if save_plots:
        from .plotting import plot_crps_by_horizon, plot_forecast_coverage, plot_pit_histogram

        if coverage_enabled and len(intervals) > 0:
            cov_all = out / "coverage_all.png"
            fig, _ax = plot_forecast_coverage(
                forecasts_eval,
                y_true_eval,
                intervals=intervals,
                horizons=horizons,
                var=None,
                use_latent=bool(ev["coverage"]["use_latent"]),
            )
            fig.savefig(cov_all, dpi=200, bbox_inches="tight")
            emit(
                "artifact",
                {"path": str(cov_all), "bytes": int(cov_all.stat().st_size), "kind": "plot"},
            )

            for vname in ds_full.variables:
                p_cov = out / f"coverage_{vname}.png"
                fig, _ax = plot_forecast_coverage(
                    forecasts_eval,
                    y_true_eval,
                    intervals=intervals,
                    horizons=horizons,
                    var=vname,
                    use_latent=bool(ev["coverage"]["use_latent"]),
                )
                fig.savefig(p_cov, dpi=200, bbox_inches="tight")
                emit(
                    "artifact",
                    {"path": str(p_cov), "bytes": int(p_cov.stat().st_size), "kind": "plot"},
                )

        if crps_enabled:
            crps_all = out / "crps_by_horizon.png"
            fig, _ax = plot_crps_by_horizon(
                forecasts_eval,
                y_true_eval,
                horizons=horizons,
                var=None,
                use_latent=bool(ev["crps"]["use_latent"]),
            )
            fig.savefig(crps_all, dpi=200, bbox_inches="tight")
            emit(
                "artifact",
                {"path": str(crps_all), "bytes": int(crps_all.stat().st_size), "kind": "plot"},
            )

        if bool(ev["pit"]["enabled"]):
            for vname in list(ev["pit"]["variables"]):
                for h in list(ev["pit"]["horizons"]):
                    p_pit = out / f"pit_{vname}_h{int(h)}.png"
                    fig, _ax = plot_pit_histogram(
                        forecasts_eval,
                        y_true_eval,
                        var=str(vname),
                        horizon=int(h),
                        bins=int(ev["pit"]["bins"]),
                        use_latent=bool(ev["pit"]["use_latent"]),
                    )
                    fig.savefig(p_pit, dpi=200, bbox_inches="tight")
                    emit(
                        "artifact",
                        {"path": str(p_pit), "bytes": int(p_pit.stat().st_size), "kind": "plot"},
                    )

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
    emit(
        "artifact",
        {"path": str(summary_path), "bytes": int(summary_path.stat().st_size), "kind": "meta"},
    )

    emit("stage_end", {"name": "write_artifacts", "elapsed_s": time.perf_counter() - t0_write})
    emit("backtest_end", {"elapsed_s": time.perf_counter() - t0_total})
