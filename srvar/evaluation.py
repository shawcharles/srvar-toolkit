from __future__ import annotations

from typing import Any

import numpy as np

from . import elb as elb_mod
from . import metrics
from .results import ForecastResult


class MetricsAccumulator:
    """Incrementally compute backtest metrics without storing all forecasts in memory."""

    def __init__(
        self,
        *,
        variables: list[str],
        max_h: int,
        evaluation: dict[str, Any],
    ) -> None:
        if max_h < 1:
            raise ValueError("max_h must be >= 1")
        if len(variables) < 1:
            raise ValueError("variables must be non-empty")

        self._variables = list(variables)
        self._max_h = int(max_h)
        self._evaluation = evaluation

        self._origins = 0

        n = int(len(self._variables))
        self._sum_sq_err = np.zeros((self._max_h, n), dtype=float)
        self._sum_abs_err = np.zeros((self._max_h, n), dtype=float)
        self._count_err = np.zeros((self._max_h, n), dtype=int)

        self._crps_enabled = bool(evaluation["crps"]["enabled"])
        self._sum_crps = np.zeros((self._max_h, n), dtype=float)
        self._count_crps = np.zeros((self._max_h, n), dtype=int)

        self._coverage_enabled = bool(evaluation["coverage"]["enabled"])
        self._coverage_intervals = (
            [float(c) for c in list(evaluation["coverage"]["intervals"])]
            if self._coverage_enabled
            else []
        )
        self._sum_coverage: dict[float, np.ndarray] = {
            c: np.zeros((self._max_h, n), dtype=float) for c in self._coverage_intervals
        }

        elb_cfg = dict(evaluation.get("elb_censor", {}))
        self._elb_enabled = bool(elb_cfg.get("enabled", False))
        self._elb_bound = float(elb_cfg["bound"]) if self._elb_enabled else float("nan")
        self._elb_indices = (
            [int(self._variables.index(v)) for v in list(elb_cfg.get("variables", []))]
            if self._elb_enabled
            else []
        )
        self._elb_censor_realized = bool(elb_cfg.get("censor_realized", True))
        self._elb_censor_forecasts = bool(elb_cfg.get("censor_forecasts", False))

    @property
    def origins(self) -> int:
        return int(self._origins)

    def update(self, *, forecast: ForecastResult, y_true: np.ndarray) -> None:
        yt = np.asarray(y_true, dtype=float)
        if yt.ndim != 2:
            raise ValueError("y_true must have shape (H, N)")
        if yt.shape[0] != self._max_h:
            raise ValueError("y_true horizon dimension must match accumulator.max_h")
        if yt.shape[1] != len(self._variables):
            raise ValueError("y_true variable dimension must match len(variables)")

        if len(forecast.variables) != len(self._variables):
            raise ValueError("forecast.variables must match variables")

        draws = np.asarray(forecast.draws, dtype=float)
        mean = np.asarray(forecast.mean, dtype=float)
        if draws.ndim != 3 or mean.ndim != 2:
            raise ValueError("forecast draws/mean must have shapes (D, H, N)/(H, N)")
        if draws.shape[1] < self._max_h or mean.shape[0] < self._max_h:
            raise ValueError("forecast does not contain required horizons")
        if draws.shape[2] != len(self._variables) or mean.shape[1] != len(self._variables):
            raise ValueError("forecast variable dimension must match len(variables)")

        if self._elb_enabled:
            if self._elb_censor_realized:
                yt = elb_mod.apply_elb_floor(yt, bound=self._elb_bound, indices=self._elb_indices)
            if self._elb_censor_forecasts:
                draws = elb_mod.apply_elb_floor(
                    draws, bound=self._elb_bound, indices=self._elb_indices
                )
                mean = draws.mean(axis=0)

        errors = mean[: self._max_h, :] - yt
        mask = ~np.isnan(errors)

        self._sum_sq_err += np.where(mask, errors**2, 0.0)
        self._sum_abs_err += np.where(mask, np.abs(errors), 0.0)
        self._count_err += mask.astype(int)

        if self._crps_enabled:
            use_latent = bool(self._evaluation["crps"]["use_latent"])
            sims_all = (
                np.asarray(forecast.latent_draws, dtype=float)
                if (use_latent and forecast.latent_draws is not None)
                else draws
            )
            for h in range(self._max_h):
                for j in range(len(self._variables)):
                    y = float(yt[h, j])
                    if np.isnan(y):
                        continue
                    val = float(metrics.crps_draws(y, sims_all[:, h, j]))
                    if np.isnan(val):
                        continue
                    self._sum_crps[h, j] += val
                    self._count_crps[h, j] += 1

        if self._coverage_enabled and self._coverage_intervals:
            use_latent = bool(self._evaluation["coverage"]["use_latent"])
            sims_all = (
                np.asarray(forecast.latent_draws, dtype=float)
                if (use_latent and forecast.latent_draws is not None)
                else draws
            )
            for c in self._coverage_intervals:
                qlo = 0.5 - 0.5 * float(c)
                qhi = 0.5 + 0.5 * float(c)
                lo = np.quantile(sims_all[:, : self._max_h, :], q=qlo, axis=0)
                hi = np.quantile(sims_all[:, : self._max_h, :], q=qhi, axis=0)
                hit = ((yt >= lo) & (yt <= hi)).astype(float)
                self._sum_coverage[c] += hit

        self._origins += 1

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for j, vname in enumerate(self._variables):
            for h in range(1, self._max_h + 1):
                n_err = int(self._count_err[h - 1, j])
                rmse = (
                    float("nan")
                    if n_err < 1
                    else float(np.sqrt(self._sum_sq_err[h - 1, j] / n_err))
                )
                mae = float("nan") if n_err < 1 else float(self._sum_abs_err[h - 1, j] / n_err)

                if self._crps_enabled:
                    n_crps = int(self._count_crps[h - 1, j])
                    crps = float("nan") if n_crps < 1 else float(self._sum_crps[h - 1, j] / n_crps)
                else:
                    crps = float("nan")

                row: dict[str, Any] = {
                    "variable": vname,
                    "horizon": h,
                    "crps": crps,
                    "rmse": rmse,
                    "mae": mae,
                }

                if self._coverage_enabled and self._coverage_intervals:
                    denom = float(self._origins) if self._origins > 0 else float("nan")
                    for c in self._coverage_intervals:
                        row[f"coverage_{int(round(100 * float(c)))}"] = float(
                            self._sum_coverage[c][h - 1, j] / denom
                        )

                rows.append(row)

        return rows


def prepare_evaluation_inputs(
    *,
    y_true: np.ndarray,
    forecasts: list[ForecastResult],
    variables: list[str],
    evaluation: dict[str, Any],
) -> tuple[np.ndarray, list[ForecastResult]]:
    """Apply evaluation-time transformations (e.g., ELB censoring) consistently.

    Returns the transformed realized array and (optionally) transformed forecasts.
    """
    yt = np.asarray(y_true, dtype=float)
    fc_eval = forecasts

    elb_cfg = dict(evaluation.get("elb_censor", {}))
    if bool(elb_cfg.get("enabled", False)):
        bound = float(elb_cfg["bound"])
        idx = [int(variables.index(v)) for v in list(elb_cfg.get("variables", []))]

        if bool(elb_cfg.get("censor_realized", True)):
            yt = elb_mod.apply_elb_floor(yt, bound=bound, indices=idx)

        if bool(elb_cfg.get("censor_forecasts", False)):
            fc_eval = []
            for fc in forecasts:
                draws_c = elb_mod.apply_elb_floor(fc.draws, bound=bound, indices=idx)
                mean_c = draws_c.mean(axis=0)
                quantiles_c = {
                    q: np.quantile(draws_c, q=float(q), axis=0) for q in fc.quantiles.keys()
                }
                fc_eval.append(
                    ForecastResult(
                        variables=list(fc.variables),
                        horizons=list(fc.horizons),
                        draws=draws_c,
                        mean=mean_c,
                        quantiles=quantiles_c,
                        latent_draws=fc.latent_draws,
                    )
                )

    return yt, fc_eval


def compute_metrics_rows(
    *,
    forecasts: list[ForecastResult],
    y_true: np.ndarray,
    variables: list[str],
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute backtest metrics rows for writing to metrics.csv.

    This intentionally mirrors the existing metrics schema:
    - Always outputs `crps`, `rmse`, `mae` columns (CRPS is NaN if disabled).
    - Coverage columns are only added when enabled.
    """
    yt = np.asarray(y_true, dtype=float)
    if yt.ndim != 3:
        raise ValueError("y_true must have shape (K, H, N)")
    if len(forecasts) != int(yt.shape[0]):
        raise ValueError("len(forecasts) must equal y_true.shape[0]")
    if int(yt.shape[2]) != len(variables):
        raise ValueError("len(variables) must equal y_true.shape[2]")

    k_orig = int(yt.shape[0])
    max_h = int(yt.shape[1])

    coverage_enabled = bool(evaluation["coverage"]["enabled"])
    crps_enabled = bool(evaluation["crps"]["enabled"])
    intervals = list(evaluation["coverage"]["intervals"]) if coverage_enabled else []

    rows: list[dict[str, Any]] = []
    for j, vname in enumerate(variables):
        for h in range(1, max_h + 1):
            y = yt[:, h - 1, j]
            mu = np.asarray([fc.mean[h - 1, j] for fc in forecasts], dtype=float)
            errors = mu - y

            row: dict[str, Any] = {
                "variable": vname,
                "horizon": h,
                "crps": float("nan"),
                "rmse": float(metrics.rmse(errors, axis=0)),
                "mae": float(metrics.mae(errors, axis=0)),
            }

            if crps_enabled:
                sims_list = [
                    (
                        fc.latent_draws
                        if (bool(evaluation["crps"]["use_latent"]) and fc.latent_draws is not None)
                        else fc.draws
                    )[:, h - 1, j]
                    for fc in forecasts
                ]
                crps_vals = np.asarray(
                    [
                        (
                            float("nan")
                            if np.isnan(y[i2])
                            else float(metrics.crps_draws(y[i2], sims))
                        )
                        for i2, sims in enumerate(sims_list)
                    ],
                    dtype=float,
                )
                row["crps"] = float(np.nanmean(crps_vals))

            if coverage_enabled:
                for c in intervals:
                    qlo = 0.5 - 0.5 * float(c)
                    qhi = 0.5 + 0.5 * float(c)
                    hit = np.empty(k_orig, dtype=float)
                    for i2, fc in enumerate(forecasts):
                        sims = (
                            fc.latent_draws
                            if (
                                bool(evaluation["coverage"]["use_latent"])
                                and fc.latent_draws is not None
                            )
                            else fc.draws
                        )
                        lo = float(np.quantile(sims[:, h - 1, j], q=qlo))
                        hi = float(np.quantile(sims[:, h - 1, j], q=qhi))
                        yi = float(yt[i2, h - 1, j])
                        hit[i2] = float(lo <= yi <= hi)
                    row[f"coverage_{int(round(100 * float(c)))}"] = float(np.nanmean(hit))

            rows.append(row)

    return rows
