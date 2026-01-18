from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult:
    statistic: float
    pvalue: float
    alternative: Literal["two-sided", "less", "greater"]
    nobs: int
    mean_diff: float
    long_run_variance: float
    max_lag: int
    horizon: int
    small_sample_correction: bool


def newey_west_long_run_variance(x: np.ndarray, *, max_lag: int) -> float:
    """Estimate long-run variance via Newey-West with Bartlett kernel.

    Parameters
    ----------
    x:
        1D array-like time series. NaNs are removed.
    max_lag:
        Maximum lag for HAC estimation. Must be >= 0.
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 1:
        return float("nan")
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    if max_lag == 0:
        return float(np.mean(v * v))

    L = int(min(max_lag, n - 1))
    gamma0 = float(np.mean(v * v))
    lrv = gamma0

    for k in range(1, L + 1):
        w = 1.0 - (k / float(L + 1))
        cov = float(np.mean(v[k:] * v[:-k]))
        lrv += 2.0 * w * cov

    return float(max(lrv, 0.0))


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    max_lag: int | None = None,
    small_sample_correction: bool = True,
) -> DieboldMarianoResult:
    """Diebold–Mariano test for equal predictive accuracy.

    Tests E[d_t] = 0 where d_t = loss_a_t - loss_b_t.

    Parameters
    ----------
    loss_a, loss_b:
        Loss series (same length). NaNs are dropped pairwise.
    horizon:
        Forecast horizon associated with the loss series. Used for the default HAC lag
        choice (`max_lag = horizon - 1`) and for the optional Harvey–Leybourne–Newbold
        small-sample correction.
    alternative:
        Alternative hypothesis for E[d_t]:
        - "two-sided": E[d_t] != 0
        - "less":      E[d_t] < 0
        - "greater":   E[d_t] > 0
    max_lag:
        HAC lag. Defaults to `max(horizon - 1, 0)`.
    small_sample_correction:
        Apply the Harvey–Leybourne–Newbold correction.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    a = np.asarray(loss_a, dtype=float).reshape(-1)
    b = np.asarray(loss_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")

    mask = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[mask]
    n = int(d.size)
    if n < 2:
        raise ValueError("need at least 2 finite observations for DM test")

    L_default = max(int(horizon) - 1, 0)
    L_raw = L_default if max_lag is None else int(max_lag)
    if L_raw < 0:
        raise ValueError("max_lag must be >= 0")
    L = int(min(L_raw, n - 1))

    mean_d = float(np.mean(d))

    d_centered = d - mean_d
    lrv = newey_west_long_run_variance(d_centered, max_lag=L)

    if not np.isfinite(lrv) or lrv < 0.0:
        lrv = float("nan")

    if lrv == 0.0:
        if mean_d == 0.0:
            stat = 0.0
        else:
            stat = float(np.copysign(np.inf, mean_d))
    else:
        stat = float(mean_d / np.sqrt(lrv / float(n)))

    if small_sample_correction and np.isfinite(stat):
        h = int(horizon)
        factor = (n + 1.0 - 2.0 * h + (h * (h - 1)) / float(n)) / float(n)
        if factor <= 0.0:
            raise ValueError(
                f"too few observations for small-sample correction (nobs={n}, horizon={h})"
            )
        stat = float(stat * np.sqrt(factor))

    df = n - 1
    if alternative == "two-sided":
        pvalue = float(2.0 * stats.t.sf(np.abs(stat), df=df))
    elif alternative == "greater":
        pvalue = float(stats.t.sf(stat, df=df))
    elif alternative == "less":
        pvalue = float(stats.t.cdf(stat, df=df))
    else:
        raise ValueError("alternative must be one of: two-sided, less, greater")

    return DieboldMarianoResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        alternative=alternative,
        nobs=n,
        mean_diff=mean_d,
        long_run_variance=float(lrv),
        max_lag=L,
        horizon=int(horizon),
        small_sample_correction=bool(small_sample_correction),
    )
