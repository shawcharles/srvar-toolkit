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


def newey_west_covariance_matrix(z: np.ndarray, *, nlags: int) -> np.ndarray:
    """Newey–West covariance matrix estimator (Bartlett kernel).

    This matches the MATLAB reference implementation used in the original replication code
    (`functions/NeweyWest.m`): it demeans `z` column-wise and uses zero-padded lag matrices.
    """
    x = np.asarray(z, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("z must be a 1D or 2D array")
    if nlags < 0:
        raise ValueError("nlags must be >= 0")

    n, k = x.shape
    if n < 1:
        return np.full((k, k), np.nan, dtype=float)

    x0 = x - x.mean(axis=0, keepdims=True)
    omegahat = (x0.T @ x0) / float(n)

    L = int(min(int(nlags), n - 1))
    if L < 1:
        return omegahat

    for lag in range(1, L + 1):
        xlag = np.zeros_like(x0)
        xlag[lag:, :] = x0[: n - lag, :]

        gamma = (x0.T @ xlag + xlag.T @ x0) / float(n)
        w = 1.0 - (lag / float(L + 1))
        omegahat = omegahat + w * gamma

    return omegahat


@dataclass(frozen=True, slots=True)
class GiacominiWhiteResult:
    statistic: float
    pvalue: float
    critical_value: float
    df: int
    nobs: int
    horizon: int
    choice: Literal["unconditional", "conditional"]
    mean_diff: float
    significance_code: int


def giacomini_white_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    *,
    horizon: int,
    alpha: float = 0.05,
    choice: Literal["unconditional", "conditional"] = "unconditional",
) -> GiacominiWhiteResult:
    """Giacomini–White conditional predictive ability (CPA) test.

    This implements the asymptotic CPA test from Giacomini & White (2006) and matches the
    MATLAB reference routine `functions/CPAtest.m` used in the Carriero et al. replication code.

    Parameters
    ----------
    loss1, loss2:
        Loss series over the out-of-sample period. NaNs are dropped pairwise.
    horizon:
        Forecast horizon (tau). Used to define the instrument set in the conditional test and
        the HAC lag length (`nlags = horizon - 1`) when `horizon > 1`.
    alpha:
        Nominal risk level for the chi-square critical value.
    choice:
        - "unconditional": instruments are a constant.
        - "conditional": instruments are a constant and lagged loss differential `d_{t-h}`.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    a = np.asarray(loss1, dtype=float).reshape(-1)
    b = np.asarray(loss2, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("loss1 and loss2 must have the same shape")

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    d1 = a - b

    tau = int(horizon)
    tt = int(d1.size)
    if tt < max(2, tau + 1):
        raise ValueError("not enough observations for GW test")

    if choice == "unconditional":
        instruments = np.ones((tt, 1), dtype=float)
        d = d1
        t = tt
    elif choice == "conditional":
        instruments = np.column_stack([np.ones(tt - tau, dtype=float), d1[: tt - tau]])
        d = d1[tau:]
        t = tt - tau
    else:
        raise ValueError("choice must be one of: unconditional, conditional")

    reg2 = instruments * d.reshape(-1, 1)
    q = int(reg2.shape[1])

    if tau == 1:
        y = np.ones(t, dtype=float)
        beta, *_ = np.linalg.lstsq(reg2, y, rcond=None)
        err = y - reg2 @ beta
        r2 = 1.0 - float(np.mean(err**2))
        teststat = float(t * r2)
    else:
        zbar = reg2.mean(axis=0, keepdims=True).T  # (q, 1)
        omega = newey_west_covariance_matrix(reg2, nlags=tau - 1)
        if np.allclose(zbar, 0.0):
            teststat = 0.0
        else:
            if np.allclose(omega, 0.0):
                teststat = float("inf")
            else:
                omega_pinv = np.linalg.pinv(omega)
                teststat = float(t * (zbar.T @ omega_pinv @ zbar).item())

    crit = float(stats.chi2.ppf(1.0 - float(alpha), df=q))
    pval = float(1.0 - stats.chi2.cdf(np.abs(teststat), df=q))

    if pval < 0.01:
        sig = 3
    elif pval < 0.05:
        sig = 2
    elif pval < 0.1:
        sig = 1
    else:
        sig = 0

    return GiacominiWhiteResult(
        statistic=float(teststat),
        pvalue=float(pval),
        critical_value=crit,
        df=q,
        nobs=t,
        horizon=tau,
        choice=choice,
        mean_diff=float(np.mean(a - b)),
        significance_code=int(sig),
    )


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
