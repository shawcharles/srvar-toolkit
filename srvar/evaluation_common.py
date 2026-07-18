from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import elb as elb_mod
from .results import ForecastResult


def select_forecast_draws(
    forecast: ForecastResult,
    *,
    use_latent: bool = False,
) -> np.ndarray:
    """Return predictive draws for evaluation or plotting.

    When ``use_latent`` is true and latent draws are available, those are returned.
    Otherwise the observed predictive draws are used.
    """
    sims = (
        forecast.latent_draws
        if (use_latent and forecast.latent_draws is not None)
        else forecast.draws
    )
    arr = np.asarray(sims, dtype=float)
    if arr.ndim != 3:
        raise ValueError("forecast draws must have shape (D, H, N)")
    return arr


def _elb_evaluation_spec(
    *,
    variables: list[str],
    evaluation: dict[str, Any],
) -> tuple[bool, float, list[int], bool, bool]:
    elb_cfg = dict(evaluation.get("elb_censor", {}))
    enabled = bool(elb_cfg.get("enabled", False))
    if not enabled:
        return False, float("nan"), [], True, False

    bound = float(elb_cfg["bound"])
    indices = [int(variables.index(v)) for v in list(elb_cfg.get("variables", []))]
    censor_realized = bool(elb_cfg.get("censor_realized", True))
    censor_forecasts = bool(elb_cfg.get("censor_forecasts", False))
    return True, bound, indices, censor_realized, censor_forecasts


def apply_elb_censor_to_forecast(
    forecast: ForecastResult,
    *,
    bound: float,
    indices: list[int],
) -> ForecastResult:
    """Apply ELB censoring to observed predictive draws and derived summaries."""
    draws_c = elb_mod.apply_elb_floor(forecast.draws, bound=bound, indices=indices)
    mean_c = draws_c.mean(axis=0)
    quantiles_c = {q: np.quantile(draws_c, q=float(q), axis=0) for q in forecast.quantiles.keys()}

    return ForecastResult(
        variables=list(forecast.variables),
        horizons=list(forecast.horizons),
        draws=draws_c,
        mean=mean_c,
        quantiles=quantiles_c,
        latent_draws=forecast.latent_draws,
    )


def prepare_evaluation_pair(
    *,
    forecast: ForecastResult,
    y_true: np.ndarray,
    variables: list[str],
    evaluation: dict[str, Any],
) -> tuple[np.ndarray, ForecastResult]:
    """Apply evaluation-time transformations to one realized block and one forecast."""
    yt = np.asarray(y_true, dtype=float)
    if yt.ndim != 2:
        raise ValueError("y_true must have shape (H, N)")

    enabled, bound, indices, censor_realized, censor_forecasts = _elb_evaluation_spec(
        variables=variables, evaluation=evaluation
    )
    if not enabled:
        return yt, forecast

    yt_out = yt
    fc_out = forecast
    if censor_realized:
        yt_out = elb_mod.apply_elb_floor(yt_out, bound=bound, indices=indices)
    if censor_forecasts:
        fc_out = apply_elb_censor_to_forecast(fc_out, bound=bound, indices=indices)
    return yt_out, fc_out


def prepare_evaluation_inputs(
    *,
    y_true: np.ndarray,
    forecasts: list[ForecastResult],
    variables: list[str],
    evaluation: dict[str, Any],
) -> tuple[np.ndarray, list[ForecastResult]]:
    """Apply evaluation-time transformations consistently across all origins."""
    yt = np.asarray(y_true, dtype=float)
    if yt.ndim != 3:
        raise ValueError("y_true must have shape (K, H, N)")
    if len(forecasts) != int(yt.shape[0]):
        raise ValueError("len(forecasts) must equal y_true.shape[0]")

    yt_out = np.empty_like(yt)
    fc_out: list[ForecastResult] = []
    for i, forecast in enumerate(forecasts):
        yi, fi = prepare_evaluation_pair(
            forecast=forecast,
            y_true=yt[i],
            variables=variables,
            evaluation=evaluation,
        )
        yt_out[i] = yi
        fc_out.append(fi)

    return yt_out, fc_out


def extract_series_draws(
    forecasts: list[ForecastResult],
    *,
    horizon_index: int,
    var_index: int,
    use_latent: bool = False,
) -> list[np.ndarray]:
    """Extract one predictive-draw vector per origin for a given horizon and variable."""
    out: list[np.ndarray] = []
    for forecast in forecasts:
        sims = select_forecast_draws(forecast, use_latent=use_latent)
        if horizon_index < 0 or horizon_index >= int(sims.shape[1]):
            raise ValueError("horizon_index out of range")
        if var_index < 0 or var_index >= int(sims.shape[2]):
            raise ValueError("var_index out of range")
        out.append(np.asarray(sims[:, horizon_index, var_index], dtype=float).reshape(-1))
    return out


def score_value_from_draws(
    y: float,
    draws: np.ndarray,
    scorer: Callable[..., float],
    **kwargs: Any,
) -> float:
    """Score one realized scalar against one predictive-draw vector."""
    sims = np.asarray(draws, dtype=float).reshape(-1)
    if sims.size < 1:
        raise ValueError("draws must be non-empty")
    if not np.isfinite(y):
        return float("nan")
    return float(scorer(float(y), sims, **kwargs))


def score_vector_from_draws(
    y: np.ndarray,
    draws_by_origin: Sequence[np.ndarray],
    scorer: Callable[..., float],
    **kwargs: Any,
) -> np.ndarray:
    """Score one realized vector against origin-aligned predictive-draw vectors."""
    yt = np.asarray(y, dtype=float).reshape(-1)
    if yt.shape[0] != len(draws_by_origin):
        raise ValueError("y and draws_by_origin must have the same length")

    out = np.full(yt.shape, np.nan, dtype=float)
    for i, draws in enumerate(draws_by_origin):
        out[i] = score_value_from_draws(float(yt[i]), draws, scorer, **kwargs)
    return out


def interval_hit_value(y: float, draws: np.ndarray, *, interval: float) -> float:
    """Return a coverage hit for one realized scalar and one predictive-draw vector."""
    sims = np.asarray(draws, dtype=float).reshape(-1)
    if sims.size < 1:
        raise ValueError("draws must be non-empty")

    c = float(interval)
    if not np.isfinite(c) or c <= 0.0 or c >= 1.0:
        raise ValueError("interval must be finite and in (0, 1)")
    if not np.isfinite(y):
        return float("nan")
    if np.any(~np.isfinite(sims)):
        return float("nan")

    qlo = 0.5 - 0.5 * c
    qhi = 0.5 + 0.5 * c
    lo = float(np.quantile(sims, q=qlo))
    hi = float(np.quantile(sims, q=qhi))
    return float(lo <= y <= hi)


def interval_hit_vector(
    y: np.ndarray,
    draws_by_origin: Sequence[np.ndarray],
    *,
    interval: float,
) -> np.ndarray:
    """Return origin-aligned coverage hits with NaN for non-evaluable observations."""
    yt = np.asarray(y, dtype=float).reshape(-1)
    if yt.shape[0] != len(draws_by_origin):
        raise ValueError("y and draws_by_origin must have the same length")

    out = np.full(yt.shape, np.nan, dtype=float)
    for i, draws in enumerate(draws_by_origin):
        out[i] = interval_hit_value(float(yt[i]), draws, interval=interval)
    return out


def pit_value(y: float, draws: np.ndarray) -> float:
    """Return the PIT value for one realized scalar and one predictive-draw vector."""
    sims = np.asarray(draws, dtype=float).reshape(-1)
    if sims.size < 1:
        raise ValueError("draws must be non-empty")
    if not np.isfinite(y):
        return float("nan")
    if np.any(~np.isfinite(sims)):
        return float("nan")
    return float(np.mean(sims <= y))


def pit_vector(y: np.ndarray, draws_by_origin: Sequence[np.ndarray]) -> np.ndarray:
    """Return origin-aligned PIT values with NaN for non-evaluable observations."""
    yt = np.asarray(y, dtype=float).reshape(-1)
    if yt.shape[0] != len(draws_by_origin):
        raise ValueError("y and draws_by_origin must have the same length")

    out = np.full(yt.shape, np.nan, dtype=float)
    for i, draws in enumerate(draws_by_origin):
        out[i] = pit_value(float(yt[i]), draws)
    return out


def mean_of_finite(values: np.ndarray) -> float:
    """Average over finite values only, returning NaN when no finite values exist."""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(arr[mask]))
