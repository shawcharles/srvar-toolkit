from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import scipy.stats

from .linalg import solve_psd


@dataclass(frozen=True, slots=True)
class ElbSpec:
    bound: float
    applies_to: list[str] = field(default_factory=list)
    tol: float = 1e-8
    enabled: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.bound):
            raise ValueError("elb.bound must be finite")
        if self.tol < 0 or not np.isfinite(self.tol):
            raise ValueError("elb.tol must be finite and >= 0")
        if self.enabled and len(self.applies_to) < 1:
            raise ValueError("elb.applies_to must not be empty when enabled")


def truncnorm_rvs_upper(*, mean: float, sd: float, upper: float, rng: np.random.Generator) -> float:
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("sd must be positive")

    a = -np.inf
    b = (upper - mean) / sd
    return float(scipy.stats.truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, random_state=rng))


def apply_elb_floor(values: np.ndarray, *, bound: float, indices: list[int]) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    if y.ndim < 2:
        raise ValueError("values must have ndim >= 2")

    if len(indices) == 0:
        return y

    y[..., indices] = np.maximum(y[..., indices], bound)
    return y


def _x_row(y: np.ndarray, *, t: int, p: int, include_intercept: bool) -> np.ndarray:
    n = y.shape[1]
    parts: list[np.ndarray] = []
    if include_intercept:
        parts.append(np.array([1.0], dtype=float))
    for lag in range(1, p + 1):
        parts.append(y[t - lag])
    return np.concatenate(parts)


def sample_shadow_value(
    *,
    y: np.ndarray,
    t: int,
    j: int,
    p: int,
    beta: np.ndarray,
    sigma: np.ndarray,
    upper: float,
    include_intercept: bool,
    rng: np.random.Generator,
) -> float:
    """Sample y[t, j] from its full conditional (truncated above at `upper`).

    This conditional accounts for the fact that y[t, j] enters as a regressor in the next p likelihood terms.

    Assumptions:
    - Standard VAR(p): y_t ~ N(x_t beta, Sigma)
    - The only constraint is: y[t, j] <= upper

    Returns:
        new value for y[t, j]
    """
    y = np.asarray(y, dtype=float)
    beta = np.asarray(beta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    t_max, n = y.shape
    if not (0 <= j < n):
        raise ValueError("j out of range")
    if not (0 <= t < t_max):
        raise ValueError("t out of range")
    if p < 1:
        raise ValueError("p must be >= 1")

    e_j = np.zeros(n, dtype=float)
    e_j[j] = 1.0

    sinv_ej = solve_psd(sigma, e_j)
    sinv_jj = float(sinv_ej[j])

    a = 0.0
    b = 0.0

    s_start = max(p, t)
    s_end = min(t_max - 1, t + p)

    # cache current y[t, j]
    y_curr = float(y[t, j])

    for s in range(s_start, s_end + 1):
        x_s = _x_row(y, t=s, p=p, include_intercept=include_intercept)
        mu_s = x_s @ beta

        if s == t:
            # Likelihood term for time t itself (only exists if t >= p)
            # mu_t does not depend on y[t, j] in a VAR, since regressors are lagged.
            u = y[s].copy()
            u[j] = 0.0
            u = u - mu_s
            sinv_u = solve_psd(sigma, u)
            a += sinv_jj
            b += -float(sinv_u[j])
            continue

        # future terms where y[t, j] appears as a lagged regressor
        lag = s - t
        if not (1 <= lag <= p):
            continue

        k0 = 1 if include_intercept else 0
        idx = k0 + (lag - 1) * n + j
        d = beta[idx, :]

        r_curr = y[s] - mu_s
        r_base = r_curr + d * y_curr

        sinv_d = solve_psd(sigma, d)
        sinv_r = solve_psd(sigma, r_base)

        a += float(d @ sinv_d)
        b += float(d @ sinv_r)

    if a <= 0.0 or not np.isfinite(a):
        raise RuntimeError("non-positive or invalid conditional precision")

    var = 1.0 / a
    mean = b / a

    return truncnorm_rvs_upper(mean=mean, sd=float(np.sqrt(var)), upper=upper, rng=rng)


def sample_shadow_value_svrw(
    *,
    y: np.ndarray,
    h: np.ndarray,
    t: int,
    j: int,
    p: int,
    beta: np.ndarray,
    upper: float,
    include_intercept: bool,
    rng: np.random.Generator,
) -> float:
    y = np.asarray(y, dtype=float)
    ht = np.asarray(h, dtype=float)
    beta = np.asarray(beta, dtype=float)

    t_max, n = y.shape
    if ht.shape != y.shape:
        raise ValueError("h must have the same shape as y")
    if not (0 <= j < n):
        raise ValueError("j out of range")
    if not (0 <= t < t_max):
        raise ValueError("t out of range")
    if p < 1:
        raise ValueError("p must be >= 1")

    a = 0.0
    b = 0.0

    s_start = max(p, t)
    s_end = min(t_max - 1, t + p)

    y_curr = float(y[t, j])

    for s in range(s_start, s_end + 1):
        x_s = _x_row(y, t=s, p=p, include_intercept=include_intercept)
        mu_s = x_s @ beta

        if s == t:
            w = float(np.exp(-ht[s, j]))
            a += w
            b += w * float(mu_s[j])
            continue

        lag = s - t
        if not (1 <= lag <= p):
            continue

        k0 = 1 if include_intercept else 0
        idx = k0 + (lag - 1) * n + j

        for i in range(n):
            d = float(beta[idx, i])
            if d == 0.0:
                continue

            w = float(np.exp(-ht[s, i]))
            r_base = float(y[s, i] - mu_s[i] + d * y_curr)
            a += w * d * d
            b += w * d * r_base

    if a <= 0.0 or not np.isfinite(a):
        raise RuntimeError("non-positive or invalid conditional precision")

    var = 1.0 / a
    mean = b / a
    return truncnorm_rvs_upper(mean=mean, sd=float(np.sqrt(var)), upper=upper, rng=rng)
