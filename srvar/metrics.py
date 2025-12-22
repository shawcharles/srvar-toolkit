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
