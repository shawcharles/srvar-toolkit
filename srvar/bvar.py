from __future__ import annotations

import numpy as np
import scipy.stats

from .linalg import cholesky_jitter, solve_psd, symmetrize


def posterior_niw(
    *,
    x: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    v0: np.ndarray,
    s0: np.ndarray,
    nu0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute NIW posterior parameters for VAR coefficients.

    Model:
        Y | B, Sigma ~ MN(X B, I, Sigma)
        B | Sigma ~ MN(M0, V0, Sigma)
        Sigma ~ InvWishart(nu0, S0)

    Returns:
        Mn, Vn, Sn, nun
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    s0 = np.asarray(s0, dtype=float)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays")

    t, k = x.shape
    if y.shape[0] != t:
        raise ValueError("x and y must have the same number of rows")

    n = y.shape[1]
    if m0.shape != (k, n):
        raise ValueError("m0 must have shape (K, N)")
    if v0.shape != (k, k):
        raise ValueError("v0 must have shape (K, K)")
    if s0.shape != (n, n):
        raise ValueError("s0 must have shape (N, N)")

    inv_v0 = solve_psd(v0, np.eye(k, dtype=float))

    xtx = x.T @ x
    inv_vn = symmetrize(inv_v0 + xtx)
    vn = solve_psd(inv_vn, np.eye(k, dtype=float))

    rhs = inv_v0 @ m0 + x.T @ y
    mn = vn @ rhs

    resid = y - x @ mn
    sn = s0 + resid.T @ resid + (mn - m0).T @ inv_v0 @ (mn - m0)
    sn = symmetrize(sn)

    nun = float(nu0 + t)

    return mn, vn, sn, nun


def sample_posterior_niw(
    *,
    mn: np.ndarray,
    vn: np.ndarray,
    sn: np.ndarray,
    nun: float,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (B, Sigma) from a matrix-normal inverse-Wishart posterior.

    Returns:
        beta_draws: (D, K, N)
        sigma_draws: (D, N, N)
    """
    if draws < 1:
        raise ValueError("draws must be >= 1")

    mn = np.asarray(mn, dtype=float)
    vn = np.asarray(vn, dtype=float)
    sn = np.asarray(sn, dtype=float)

    k, n = mn.shape

    lv = cholesky_jitter(vn)

    beta_draws = np.empty((draws, k, n), dtype=float)
    sigma_draws = np.empty((draws, n, n), dtype=float)

    iw = scipy.stats.invwishart(df=nun, scale=sn)

    for d in range(draws):
        sigma = iw.rvs(random_state=rng)
        sigma = symmetrize(np.asarray(sigma, dtype=float))

        ls = cholesky_jitter(sigma)
        z = rng.standard_normal((k, n))
        beta = mn + lv @ z @ ls.T

        beta_draws[d] = beta
        sigma_draws[d] = sigma

    return beta_draws, sigma_draws


def simulate_var_forecast(
    *,
    y_last: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
    include_intercept: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one forecast path for a VAR(p).

    Args:
        y_last: (p, N) last p observations in chronological order.
            Convention: y_last[0] is the oldest lag and y_last[-1] is the most recent.
        beta: (K, N)
        sigma: (N, N)
        horizon: number of steps to simulate

    Returns:
        path: (horizon, N)
    """
    y_last = np.asarray(y_last, dtype=float)
    if y_last.ndim != 2:
        raise ValueError("y_last must be 2D (p, N)")

    p, n = y_last.shape

    beta = np.asarray(beta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    k_expected = (1 if include_intercept else 0) + n * p
    if beta.shape != (k_expected, n):
        raise ValueError("beta has wrong shape for given p and include_intercept")

    if sigma.shape != (n, n):
        raise ValueError("sigma must have shape (N, N)")

    path = np.empty((horizon, n), dtype=float)
    lags = y_last.copy()

    for h in range(horizon):
        x_parts = []
        if include_intercept:
            x_parts.append(np.array([1.0], dtype=float))
        # lag 1 first, then lag 2, ...
        for lag in range(1, p + 1):
            x_parts.append(lags[-lag, :])
        x = np.concatenate(x_parts)

        mean = x @ beta
        eps = rng.multivariate_normal(mean=np.zeros(n, dtype=float), cov=sigma)
        y_next = mean + eps

        path[h] = y_next
        lags = np.vstack([lags[1:, :], y_next]) if p > 1 else y_next.reshape(1, -1)

    return path
