from __future__ import annotations

import numpy as np

from .results import ForecastResult


def pool_forecasts(
    forecasts: list[ForecastResult],
    *,
    weights: list[float] | None = None,
    draws: int | None = None,
    quantile_levels: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> ForecastResult:
    """Pool predictive distributions across models via a mixture of draws.

    This is a lightweight "forecast combination" utility that merges multiple
    `ForecastResult`s into a single combined predictive distribution.

    Pooling method: sample models according to `weights` and, for each selected
    model, sample one of its predictive draws. This yields a Monte Carlo
    approximation to a weighted mixture distribution.

    Parameters
    ----------
    forecasts:
        List of forecasts to pool. All forecasts must share the same `variables`,
        `horizons`, and draw shapes (H, N). Draw counts may differ.
    weights:
        Non-negative weights per forecast. Defaults to equal weights.
    draws:
        Number of pooled draws to produce. Defaults to `min(D_i) * len(forecasts)`,
        which is a reasonable "equal-weight" default that does not up-sample the
        largest model.
    quantile_levels:
        Quantiles to compute for the pooled forecast. Defaults to the union of
        quantile levels present in the input forecasts; if empty, no quantiles
        are computed.
    rng:
        Random number generator for mixture resampling.
    """
    if not forecasts:
        raise ValueError("forecasts must be non-empty")

    f0 = forecasts[0]
    variables = list(f0.variables)
    horizons = list(f0.horizons)

    draws_list = [np.asarray(fc.draws, dtype=float) for fc in forecasts]
    mean_list = [np.asarray(fc.mean, dtype=float) for fc in forecasts]

    if any(d.ndim != 3 for d in draws_list):
        raise ValueError("each forecast.draws must have shape (D, H, N)")
    if any(m.ndim != 2 for m in mean_list):
        raise ValueError("each forecast.mean must have shape (H, N)")

    h = int(draws_list[0].shape[1])
    n = int(draws_list[0].shape[2])

    for fc, d, m in zip(forecasts, draws_list, mean_list, strict=True):
        if list(fc.variables) != variables:
            raise ValueError("all forecasts must have identical variables ordering")
        if list(fc.horizons) != horizons:
            raise ValueError("all forecasts must have identical horizons list")
        if int(d.shape[1]) != h or int(d.shape[2]) != n:
            raise ValueError("all forecasts must have identical (H, N) draw shapes")
        if int(m.shape[0]) != h or int(m.shape[1]) != n:
            raise ValueError("all forecasts must have identical (H, N) mean shapes")

    m_models = int(len(forecasts))
    d_counts = np.asarray([int(d.shape[0]) for d in draws_list], dtype=int)
    if np.any(d_counts < 1):
        raise ValueError("each forecast must contain at least 1 draw")

    if draws is None:
        draws_out = int(d_counts.min() * m_models)
    else:
        draws_out = int(draws)
    if draws_out < 1:
        raise ValueError("draws must be >= 1")

    if weights is None:
        w = np.full(m_models, 1.0 / float(m_models), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if int(w.size) != m_models:
            raise ValueError("weights must have the same length as forecasts")
        if np.any(~np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        s = float(np.sum(w))
        if s <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w = w / s

    rng_ = np.random.default_rng() if rng is None else rng

    model_idx = rng_.choice(m_models, size=draws_out, replace=True, p=w)

    pooled = np.empty((draws_out, h, n), dtype=float)
    for k in range(m_models):
        pos = np.where(model_idx == k)[0]
        if pos.size < 1:
            continue
        idx = rng_.integers(0, d_counts[k], size=int(pos.size))
        pooled[pos, :, :] = draws_list[k][idx, :, :]

    all_latent = all(fc.latent_draws is not None for fc in forecasts)
    pooled_latent: np.ndarray | None
    if all_latent:
        latent_list = [np.asarray(fc.latent_draws, dtype=float) for fc in forecasts]
        if any(ld.shape[1:] != (h, n) for ld in latent_list):
            raise ValueError("all latent_draws must have identical (H, N) shapes when present")
        pooled_latent = np.empty((draws_out, h, n), dtype=float)
        for k in range(m_models):
            pos = np.where(model_idx == k)[0]
            if pos.size < 1:
                continue
            idx = rng_.integers(0, int(latent_list[k].shape[0]), size=int(pos.size))
            pooled_latent[pos, :, :] = latent_list[k][idx, :, :]
    else:
        pooled_latent = None

    mean = pooled.mean(axis=0)

    if quantile_levels is None:
        qs = sorted({float(q) for fc in forecasts for q in fc.quantiles.keys()})
    else:
        qs = [float(q) for q in quantile_levels]
    for q in qs:
        if not np.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError("quantile_levels must satisfy 0 <= q <= 1")

    quantiles: dict[float, np.ndarray]
    if qs:
        quantiles = {float(q): np.quantile(pooled, q=float(q), axis=0) for q in qs}
    else:
        quantiles = {}

    return ForecastResult(
        variables=variables,
        horizons=horizons,
        draws=pooled,
        mean=mean,
        quantiles=quantiles,
        latent_draws=pooled_latent,
    )
