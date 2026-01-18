from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..results import FEVDResult, FitResult, IRFResult
from .irf import irf_cholesky


def _parse_horizons(horizons: int | Sequence[int]) -> list[int]:
    if isinstance(horizons, (int, np.integer)) and not isinstance(horizons, bool):
        hmax = int(horizons)
        if hmax < 1:
            raise ValueError("horizons must be >= 1")
        return list(range(1, hmax + 1))

    if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be an int or a sequence of ints")

    hs: list[int] = []
    for h in horizons:
        if not isinstance(h, (int, np.integer)) or isinstance(h, bool):
            raise ValueError("horizons must contain only integers")
        hi = int(h)
        if hi < 1:
            raise ValueError("horizons must contain only positive integers")
        hs.append(hi)

    if len(hs) < 1:
        raise ValueError("horizons must be non-empty")
    if len(set(hs)) != len(hs):
        raise ValueError("horizons must not contain duplicates")
    return sorted(hs)


def _parse_quantiles(quantile_levels: list[float] | None) -> list[float]:
    if quantile_levels is None:
        return [0.1, 0.5, 0.9]
    if not isinstance(quantile_levels, list) or len(quantile_levels) < 1:
        raise ValueError("quantile_levels must be a non-empty list")

    qs: list[float] = []
    for q in quantile_levels:
        qf = float(q)
        if not np.isfinite(qf) or not (0.0 < qf < 1.0):
            raise ValueError("quantile_levels must be finite and in (0, 1)")
        qs.append(qf)
    return qs


def fevd_from_irf(
    irf: IRFResult,
    *,
    horizons: int | Sequence[int] | None = None,
    quantile_levels: list[float] | None = None,
) -> FEVDResult:
    """Compute FEVD from a structural IRF.

    Horizon definition
    ------------------
    For forecast horizon ``h`` (steps ahead, ``h >= 1``), FEVD uses the cumulative sum of squared
    structural impulse responses from IRF horizons ``0..h-1``.

    Notes
    -----
    - This requires an orthogonal (structural) IRF. Reduced-form IRFs are not supported.
    - The IRF must include all integer horizons ``0..(max(horizons)-1)``.
    """
    if irf.identification == "reduced_form":
        raise ValueError("FEVD requires a structural IRF; reduced-form identification is not valid")

    if not isinstance(irf.horizons, list) or len(irf.horizons) < 1:
        raise ValueError("irf.horizons must be a non-empty list")

    if horizons is None:
        default_hmax = max(1, int(max(irf.horizons)))
        h_list = _parse_horizons(default_hmax)
    else:
        h_list = _parse_horizons(horizons)
    q_list = _parse_quantiles(quantile_levels)

    hmax = int(max(h_list))
    max_lag = hmax - 1

    irf_h = list(irf.horizons)
    idx = {int(h): i for i, h in enumerate(irf_h)}
    if 0 not in idx:
        raise ValueError("FEVD requires IRF horizons to include 0 (impact response)")
    missing = [k for k in range(max_lag + 1) if k not in idx]
    if missing:
        raise ValueError(
            f"FEVD requires IRF horizons to include all integers 0..{max_lag}; missing: {missing}"
        )

    draws = np.asarray(irf.draws, dtype=float)
    if draws.ndim != 4:
        raise ValueError("irf.draws must have shape (D, H, N, N)")
    if draws.shape[1] != len(irf_h):
        raise ValueError("irf.draws second dimension must match len(irf.horizons)")

    n = len(irf.variables)
    s = len(irf.shocks)
    if draws.shape[2:] != (n, s):
        raise ValueError("irf.draws has wrong shape relative to variables/shocks")

    needed = [idx[k] for k in range(max_lag + 1)]
    theta = draws[:, needed, :, :]  # (D, L, N, S), where L = max_lag+1

    cum_sq = np.cumsum(theta**2, axis=1)  # (D, L, N, S)
    sel = np.asarray([h - 1 for h in h_list], dtype=int)
    num = cum_sq[:, sel, :, :]  # (D, H_out, N, S)
    denom = num.sum(axis=3, keepdims=True)  # (D, H_out, N, 1)

    out = np.full_like(num, np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 0)
    np.divide(num, denom, out=out, where=valid)

    mean = np.nanmean(out, axis=0)
    quants = {q: np.nanquantile(out, q=q, axis=0) for q in q_list}

    metadata: dict[str, Any] = dict(irf.metadata) if irf.metadata is not None else {}
    metadata.update(
        {
            "fevd_horizon_definition": "h steps ahead uses IRF horizons 0..h-1",
            "source": "irf",
        }
    )

    zero_var = int(np.count_nonzero(~valid))
    if zero_var:
        metadata["zero_variance_cells"] = zero_var

    return FEVDResult(
        variables=list(irf.variables),
        shocks=list(irf.shocks),
        horizons=list(h_list),
        draws=out,
        mean=mean,
        quantiles=quants,
        identification=str(irf.identification),
        metadata=metadata,
    )


def fevd_cholesky(
    fit: FitResult,
    *,
    horizons: int | Sequence[int] = 24,
    draws: int | None = None,
    ordering: Sequence[str] | None = None,
    shock_scale: Literal["one_sd", "unit"] = "one_sd",
    quantile_levels: list[float] | None = None,
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> FEVDResult:
    """Convenience wrapper: compute Cholesky FEVD from a fitted model."""
    h_list = _parse_horizons(horizons)
    max_lag = int(max(h_list) - 1)

    irf = irf_cholesky(
        fit,
        horizons=max_lag,
        draws=draws,
        ordering=ordering,
        shock_scale=shock_scale,
        quantile_levels=[0.5],
        stationarity=stationarity,
        stationarity_tol=stationarity_tol,
        stationarity_max_draws=stationarity_max_draws,
        rng=rng,
    )
    return fevd_from_irf(irf, horizons=h_list, quantile_levels=quantile_levels)
