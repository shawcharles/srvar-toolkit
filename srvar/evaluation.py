from __future__ import annotations

from typing import Any

import numpy as np

from . import metrics
from .evaluation_common import (
    extract_series_draws,
    interval_hit_value,
    interval_hit_vector,
    mean_of_finite,
    prepare_evaluation_pair,
    score_value_from_draws,
    score_vector_from_draws,
    select_forecast_draws,
)
from .evaluation_common import (
    prepare_evaluation_inputs as prepare_evaluation_inputs_common,
)
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

        self._wis_enabled = bool(evaluation["wis"]["enabled"])
        self._wis_intervals = (
            [float(c) for c in list(evaluation["wis"]["intervals"])] if self._wis_enabled else []
        )
        self._sum_wis = np.zeros((self._max_h, n), dtype=float)
        self._count_wis = np.zeros((self._max_h, n), dtype=int)

        self._pinball_enabled = bool(evaluation["pinball"]["enabled"])
        self._pinball_quantiles = (
            [float(q) for q in list(evaluation["pinball"]["quantiles"])]
            if self._pinball_enabled
            else []
        )
        self._sum_pinball = np.zeros((self._max_h, n), dtype=float)
        self._count_pinball = np.zeros((self._max_h, n), dtype=int)

        self._log_score_enabled = bool(evaluation["log_score"]["enabled"])
        self._log_score_var_floor = (
            float(evaluation["log_score"]["variance_floor"]) if self._log_score_enabled else 1e-12
        )
        self._sum_log_score = np.zeros((self._max_h, n), dtype=float)
        self._count_log_score = np.zeros((self._max_h, n), dtype=int)

        self._coverage_enabled = bool(evaluation["coverage"]["enabled"])
        self._coverage_intervals = (
            [float(c) for c in list(evaluation["coverage"]["intervals"])]
            if self._coverage_enabled
            else []
        )
        self._sum_coverage: dict[float, np.ndarray] = {
            c: np.zeros((self._max_h, n), dtype=float) for c in self._coverage_intervals
        }
        self._count_coverage: dict[float, np.ndarray] = {
            c: np.zeros((self._max_h, n), dtype=int) for c in self._coverage_intervals
        }

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

        yt, forecast_eval = prepare_evaluation_pair(
            forecast=forecast,
            y_true=yt,
            variables=self._variables,
            evaluation=self._evaluation,
        )

        draws = np.asarray(forecast_eval.draws, dtype=float)
        mean = np.asarray(forecast_eval.mean, dtype=float)
        if draws.ndim != 3 or mean.ndim != 2:
            raise ValueError("forecast draws/mean must have shapes (D, H, N)/(H, N)")
        if draws.shape[1] < self._max_h or mean.shape[0] < self._max_h:
            raise ValueError("forecast does not contain required horizons")
        if draws.shape[2] != len(self._variables) or mean.shape[1] != len(self._variables):
            raise ValueError("forecast variable dimension must match len(variables)")

        errors = mean[: self._max_h, :] - yt
        mask = ~np.isnan(errors)

        self._sum_sq_err += np.where(mask, errors**2, 0.0)
        self._sum_abs_err += np.where(mask, np.abs(errors), 0.0)
        self._count_err += mask.astype(int)

        if self._crps_enabled:
            sims_all = select_forecast_draws(
                forecast_eval, use_latent=bool(self._evaluation["crps"]["use_latent"])
            )
            for h in range(self._max_h):
                for j in range(len(self._variables)):
                    val = score_value_from_draws(
                        float(yt[h, j]), sims_all[:, h, j], metrics.crps_draws
                    )
                    if not np.isfinite(val):
                        continue
                    self._sum_crps[h, j] += val
                    self._count_crps[h, j] += 1

        if self._wis_enabled:
            sims_all = select_forecast_draws(
                forecast_eval, use_latent=bool(self._evaluation["wis"]["use_latent"])
            )
            for h in range(self._max_h):
                for j in range(len(self._variables)):
                    val = score_value_from_draws(
                        float(yt[h, j]),
                        sims_all[:, h, j],
                        metrics.wis_draws,
                        intervals=self._wis_intervals,
                    )
                    if not np.isfinite(val):
                        continue
                    self._sum_wis[h, j] += val
                    self._count_wis[h, j] += 1

        if self._pinball_enabled:
            sims_all = select_forecast_draws(
                forecast_eval, use_latent=bool(self._evaluation["pinball"]["use_latent"])
            )
            for h in range(self._max_h):
                for j in range(len(self._variables)):
                    val = score_value_from_draws(
                        float(yt[h, j]),
                        sims_all[:, h, j],
                        metrics.pinball_draws,
                        quantiles=self._pinball_quantiles,
                    )
                    if not np.isfinite(val):
                        continue
                    self._sum_pinball[h, j] += val
                    self._count_pinball[h, j] += 1

        if self._log_score_enabled:
            sims_all = select_forecast_draws(
                forecast_eval, use_latent=bool(self._evaluation["log_score"]["use_latent"])
            )
            for h in range(self._max_h):
                for j in range(len(self._variables)):
                    val = score_value_from_draws(
                        float(yt[h, j]),
                        sims_all[:, h, j],
                        metrics.log_score_draws,
                        variance_floor=self._log_score_var_floor,
                    )
                    if not np.isfinite(val):
                        continue
                    self._sum_log_score[h, j] += val
                    self._count_log_score[h, j] += 1

        if self._coverage_enabled and self._coverage_intervals:
            sims_all = select_forecast_draws(
                forecast_eval, use_latent=bool(self._evaluation["coverage"]["use_latent"])
            )
            for c in self._coverage_intervals:
                for h in range(self._max_h):
                    for j in range(len(self._variables)):
                        hit = interval_hit_value(float(yt[h, j]), sims_all[:, h, j], interval=c)
                        if not np.isfinite(hit):
                            continue
                        self._sum_coverage[c][h, j] += hit
                        self._count_coverage[c][h, j] += 1

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

                row: dict[str, Any] = {"variable": vname, "horizon": h, "crps": crps}

                if self._wis_enabled:
                    n_wis = int(self._count_wis[h - 1, j])
                    wis = float("nan") if n_wis < 1 else float(self._sum_wis[h - 1, j] / n_wis)
                    row["wis"] = wis

                if self._pinball_enabled:
                    n_pin = int(self._count_pinball[h - 1, j])
                    pinball = (
                        float("nan") if n_pin < 1 else float(self._sum_pinball[h - 1, j] / n_pin)
                    )
                    row["pinball"] = pinball

                if self._log_score_enabled:
                    n_ls = int(self._count_log_score[h - 1, j])
                    log_score = (
                        float("nan") if n_ls < 1 else float(self._sum_log_score[h - 1, j] / n_ls)
                    )
                    row["log_score"] = log_score

                row["rmse"] = rmse
                row["mae"] = mae

                if self._coverage_enabled and self._coverage_intervals:
                    for c in self._coverage_intervals:
                        n_cov = int(self._count_coverage[c][h - 1, j])
                        row[f"coverage_{int(round(100 * float(c)))}"] = (
                            float("nan")
                            if n_cov < 1
                            else float(self._sum_coverage[c][h - 1, j] / n_cov)
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
    return prepare_evaluation_inputs_common(
        y_true=y_true,
        forecasts=forecasts,
        variables=variables,
        evaluation=evaluation,
    )


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
    - Outputs `wis` only when enabled.
    - Outputs `pinball` only when enabled.
    - Outputs `log_score` only when enabled.
    - Coverage columns are only added when enabled.
    """
    yt = np.asarray(y_true, dtype=float)
    if yt.ndim != 3:
        raise ValueError("y_true must have shape (K, H, N)")
    if len(forecasts) != int(yt.shape[0]):
        raise ValueError("len(forecasts) must equal y_true.shape[0]")
    if int(yt.shape[2]) != len(variables):
        raise ValueError("len(variables) must equal y_true.shape[2]")

    yt, forecasts_eval = prepare_evaluation_inputs_common(
        y_true=yt,
        forecasts=forecasts,
        variables=variables,
        evaluation=evaluation,
    )

    max_h = int(yt.shape[1])

    coverage_enabled = bool(evaluation["coverage"]["enabled"])
    crps_enabled = bool(evaluation["crps"]["enabled"])
    wis_enabled = bool(evaluation["wis"]["enabled"])
    pinball_enabled = bool(evaluation["pinball"]["enabled"])
    log_score_enabled = bool(evaluation["log_score"]["enabled"])
    intervals = list(evaluation["coverage"]["intervals"]) if coverage_enabled else []
    wis_intervals = list(evaluation["wis"]["intervals"]) if wis_enabled else []
    pinball_quantiles = list(evaluation["pinball"]["quantiles"]) if pinball_enabled else []
    log_score_var_floor = (
        float(evaluation["log_score"]["variance_floor"]) if log_score_enabled else 1e-12
    )

    rows: list[dict[str, Any]] = []
    for j, vname in enumerate(variables):
        for h in range(1, max_h + 1):
            y = yt[:, h - 1, j]
            mu = np.asarray([fc.mean[h - 1, j] for fc in forecasts_eval], dtype=float)
            errors = mu - y
            err_mask = np.isfinite(errors)
            n_err = int(err_mask.sum())

            row: dict[str, Any] = {"variable": vname, "horizon": h, "crps": float("nan")}
            if wis_enabled:
                row["wis"] = float("nan")
            if pinball_enabled:
                row["pinball"] = float("nan")
            if log_score_enabled:
                row["log_score"] = float("nan")
            row["rmse"] = (
                float("nan")
                if n_err < 1
                else float(np.sqrt(np.sum((errors[err_mask]) ** 2) / float(n_err)))
            )
            row["mae"] = (
                float("nan")
                if n_err < 1
                else float(np.sum(np.abs(errors[err_mask])) / float(n_err))
            )

            if crps_enabled:
                crps_vals = score_vector_from_draws(
                    y,
                    extract_series_draws(
                        forecasts_eval,
                        horizon_index=h - 1,
                        var_index=j,
                        use_latent=bool(evaluation["crps"]["use_latent"]),
                    ),
                    metrics.crps_draws,
                )
                row["crps"] = mean_of_finite(crps_vals)

            if wis_enabled:
                wis_vals = score_vector_from_draws(
                    y,
                    extract_series_draws(
                        forecasts_eval,
                        horizon_index=h - 1,
                        var_index=j,
                        use_latent=bool(evaluation["wis"]["use_latent"]),
                    ),
                    metrics.wis_draws,
                    intervals=wis_intervals,
                )
                row["wis"] = mean_of_finite(wis_vals)

            if pinball_enabled:
                pin_vals = score_vector_from_draws(
                    y,
                    extract_series_draws(
                        forecasts_eval,
                        horizon_index=h - 1,
                        var_index=j,
                        use_latent=bool(evaluation["pinball"]["use_latent"]),
                    ),
                    metrics.pinball_draws,
                    quantiles=pinball_quantiles,
                )
                row["pinball"] = mean_of_finite(pin_vals)

            if log_score_enabled:
                ls_vals = score_vector_from_draws(
                    y,
                    extract_series_draws(
                        forecasts_eval,
                        horizon_index=h - 1,
                        var_index=j,
                        use_latent=bool(evaluation["log_score"]["use_latent"]),
                    ),
                    metrics.log_score_draws,
                    variance_floor=log_score_var_floor,
                )
                row["log_score"] = mean_of_finite(ls_vals)

            if coverage_enabled:
                for c in intervals:
                    hit = interval_hit_vector(
                        y,
                        extract_series_draws(
                            forecasts_eval,
                            horizon_index=h - 1,
                            var_index=j,
                            use_latent=bool(evaluation["coverage"]["use_latent"]),
                        ),
                        interval=float(c),
                    )
                    row[f"coverage_{int(round(100 * float(c)))}"] = mean_of_finite(hit)

            rows.append(row)

    return rows
