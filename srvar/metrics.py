from __future__ import annotations

import numpy as np


def crps_draws(y: float, draws: np.ndarray) -> float:
    x = np.asarray(draws, dtype=float).reshape(-1)
    if x.size < 1:
        raise ValueError("draws must be non-empty")

    if np.isnan(y) or np.any(np.isnan(x)):
        return float("nan")

    x_ord = np.sort(x)
    n = int(x_ord.size)

    if n == 1:
        return float(np.abs(x_ord[0] - y))

    alpha = np.full(n - 1, np.nan, dtype=float)
    beta = np.full(n - 1, np.nan, dtype=float)

    ndx1 = y < x_ord[:-1]
    ndx2 = (x_ord[:-1] <= y) & (y <= x_ord[1:])
    ndx3 = x_ord[1:] < y

    alpha[ndx1] = 0.0
    alpha[ndx2] = y - x_ord[:-1][ndx2]
    alpha[ndx3] = x_ord[1:][ndx3] - x_ord[:-1][ndx3]

    beta[ndx1] = x_ord[1:][ndx1] - x_ord[:-1][ndx1]
    beta[ndx2] = x_ord[1:][ndx2] - y
    beta[ndx3] = 0.0

    alpha_full = np.zeros(n + 1, dtype=float)
    beta_full = np.zeros(n + 1, dtype=float)
    alpha_full[1:-1] = alpha
    beta_full[1:-1] = beta

    if y >= x_ord[-1]:
        alpha_full[0] = y - x_ord[-1]
        alpha_full[-1] = y - x_ord[-1]
    if y <= x_ord[0]:
        beta_full[0] = x_ord[0] - y
        beta_full[-1] = x_ord[0] - y

    ndx = np.arange(n + 1, dtype=float) / float(n)
    return float(np.sum(alpha_full * ndx**2 + beta_full * (1.0 - ndx) ** 2))


def rmse(errors: np.ndarray, *, axis: int = 0) -> np.ndarray:
    e = np.asarray(errors, dtype=float)
    return np.sqrt(np.nanmean(e**2, axis=axis))


def mae(errors: np.ndarray, *, axis: int = 0) -> np.ndarray:
    e = np.asarray(errors, dtype=float)
    return np.nanmean(np.abs(e), axis=axis)


def wis_draws(y: float, draws: np.ndarray, *, intervals: list[float]) -> float:
    """Weighted interval score (WIS) from predictive draws.

    Parameters
    ----------
    y:
        Realized scalar value.
    draws:
        Predictive draws for the same scalar (any shape; flattened internally).
    intervals:
        Central interval coverages in [0, 1). For example, [0.5, 0.8, 0.9].

    Notes
    -----
    Uses the common WIS definition:

        WIS = (0.5*|y - m| + Σ_k (α_k/2)*IS_{α_k}(y, l_k, u_k)) / (K + 0.5)

    where m is the median, α_k = 1 - c_k, and [l_k, u_k] is the central (1-α_k) interval.
    """
    x = np.asarray(draws, dtype=float).reshape(-1)
    if x.size < 1:
        raise ValueError("draws must be non-empty")

    if np.isnan(y) or np.any(np.isnan(x)):
        return float("nan")

    if any((not np.isfinite(c)) or (c < 0.0) or (c >= 1.0) for c in intervals):
        raise ValueError("intervals must be finite and satisfy 0 <= c < 1")

    med = float(np.quantile(x, q=0.5))
    total = 0.5 * float(np.abs(y - med))

    for c in intervals:
        alpha = 1.0 - float(c)
        qlo = 0.5 - 0.5 * float(c)
        qhi = 0.5 + 0.5 * float(c)
        lo = float(np.quantile(x, q=qlo))
        hi = float(np.quantile(x, q=qhi))

        total += 0.5 * alpha * (hi - lo)
        if y < lo:
            total += lo - y
        elif y > hi:
            total += y - hi

    return float(total / (float(len(intervals)) + 0.5))
