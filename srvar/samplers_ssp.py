from __future__ import annotations

import numpy as np

from .linalg import cholesky_jitter, solve_psd, symmetrize
from .var import design_matrix

def _strip_intercept_niw_blocks(*, m0: np.ndarray, v0: np.ndarray, k_no_intercept: int) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(m0, dtype=float)
    v = np.asarray(v0, dtype=float)
    if m.ndim != 2 or v.ndim != 2:
        raise ValueError("m0 and v0 must be 2D")
    if v.shape[0] != v.shape[1]:
        raise ValueError("v0 must be square")

    k = int(k_no_intercept)
    if m.shape[0] == k and v.shape == (k, k):
        return m, v
    if m.shape[0] == (k + 1) and v.shape == (k + 1, k + 1):
        return m[1:, :], v[1:, 1:]
    raise ValueError("prior.niw has incompatible shapes for SSP")


def _asum_from_beta(*, beta: np.ndarray, n: int, p: int) -> np.ndarray:
    b = np.asarray(beta, dtype=float)
    if b.shape != (int(n) * int(p), int(n)):
        raise ValueError("beta must have shape (N*p, N)")

    a_sum = np.zeros((int(n), int(n)), dtype=float)
    for lag in range(int(p)):
        block = b[lag * int(n) : (lag + 1) * int(n), :]
        a_sum += block.T
    return a_sum


def sample_steady_state_mu(
    *,
    y: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    mu0: np.ndarray,
    v0_mu: float | np.ndarray,
    p: int,
    rng: np.random.Generator,
) -> np.ndarray:
    yt = np.asarray(y, dtype=float)
    b = np.asarray(beta, dtype=float)
    s = np.asarray(sigma, dtype=float)
    m0 = np.asarray(mu0, dtype=float).reshape(-1)

    if yt.ndim != 2:
        raise ValueError("y must be 2D")
    n = int(yt.shape[1])
    if b.shape != (n * int(p), n):
        raise ValueError("beta must have shape (N*p, N)")
    if s.shape != (n, n):
        raise ValueError("sigma must have shape (N, N)")
    if m0.shape != (n,):
        raise ValueError("mu0 must have shape (N,)")

    if isinstance(v0_mu, (float, int, np.floating, np.integer)) and not isinstance(v0_mu, bool):
        v_mu = np.full(n, float(v0_mu), dtype=float)
    else:
        v_mu = np.asarray(v0_mu, dtype=float).reshape(-1)
        if v_mu.shape != (n,):
            raise ValueError("v0_mu must be a scalar or have shape (N,)")
    if np.any(~np.isfinite(v_mu)) or np.any(v_mu <= 0):
        raise ValueError("v0_mu must be finite and > 0")

    x_lags, y_tgt = design_matrix(yt, int(p), include_intercept=False)
    r = y_tgt - x_lags @ b

    a_sum = _asum_from_beta(beta=b, n=n, p=int(p))
    bmat = np.eye(n, dtype=float) - a_sum

    inv_v0 = np.diag(1.0 / v_mu)
    inv_sigma = solve_psd(s, np.eye(n, dtype=float))
    t_eff = int(r.shape[0])
    precision = symmetrize(inv_v0 + float(t_eff) * (bmat.T @ inv_sigma @ bmat))
    rhs = inv_v0 @ m0 + bmat.T @ inv_sigma @ np.sum(r, axis=0)

    v_post = solve_psd(precision, np.eye(n, dtype=float))
    mu_hat = v_post @ rhs
    l = cholesky_jitter(symmetrize(v_post))
    return mu_hat + l @ rng.standard_normal(n)


def sample_steady_state_mu_svrw(
    *,
    y: np.ndarray,
    beta: np.ndarray,
    h: np.ndarray,
    mu0: np.ndarray,
    v0_mu: float | np.ndarray,
    p: int,
    rng: np.random.Generator,
) -> np.ndarray:
    yt = np.asarray(y, dtype=float)
    b = np.asarray(beta, dtype=float)
    ht = np.asarray(h, dtype=float)
    m0 = np.asarray(mu0, dtype=float).reshape(-1)

    if yt.ndim != 2:
        raise ValueError("y must be 2D")
    n = int(yt.shape[1])
    if b.shape != (n * int(p), n):
        raise ValueError("beta must have shape (N*p, N)")
    if m0.shape != (n,):
        raise ValueError("mu0 must have shape (N,)")

    if isinstance(v0_mu, (float, int, np.floating, np.integer)) and not isinstance(v0_mu, bool):
        v_mu = np.full(n, float(v0_mu), dtype=float)
    else:
        v_mu = np.asarray(v0_mu, dtype=float).reshape(-1)
        if v_mu.shape != (n,):
            raise ValueError("v0_mu must be a scalar or have shape (N,)")
    if np.any(~np.isfinite(v_mu)) or np.any(v_mu <= 0):
        raise ValueError("v0_mu must be finite and > 0")

    x_lags, y_tgt = design_matrix(yt, int(p), include_intercept=False)
    r = y_tgt - x_lags @ b
    if ht.shape != r.shape:
        raise ValueError("h must have shape (T-p, N) matching residuals")

    a_sum = _asum_from_beta(beta=b, n=n, p=int(p))
    bmat = np.eye(n, dtype=float) - a_sum

    inv_v0 = np.diag(1.0 / v_mu)
    precision = inv_v0.copy()
    rhs = inv_v0 @ m0

    for t in range(r.shape[0]):
        inv_sigma_t = np.diag(np.exp(-ht[t, :]))
        precision += bmat.T @ inv_sigma_t @ bmat
        rhs += bmat.T @ inv_sigma_t @ r[t, :]

    precision = symmetrize(precision)
    v_post = solve_psd(precision, np.eye(n, dtype=float))
    mu_hat = v_post @ rhs
    l = cholesky_jitter(symmetrize(v_post))
    return mu_hat + l @ rng.standard_normal(n)


def sample_mu_gamma(
    *,
    mu: np.ndarray,
    mu0: np.ndarray,
    spike_var: float,
    slab_var: float,
    inclusion_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    m = np.asarray(mu, dtype=float).reshape(-1)
    m0 = np.asarray(mu0, dtype=float).reshape(-1)
    if m.shape != m0.shape:
        raise ValueError("mu and mu0 must have the same shape")
    if spike_var <= 0 or slab_var <= 0:
        raise ValueError("spike_var and slab_var must be > 0")
    if not (0.0 < float(inclusion_prob) < 1.0):
        raise ValueError("inclusion_prob must be in (0, 1)")

    d = m - m0
    log_p1 = np.log(float(inclusion_prob)) - 0.5 * (np.log(float(slab_var)) + (d * d) / float(slab_var))
    log_p0 = np.log(1.0 - float(inclusion_prob)) - 0.5 * (np.log(float(spike_var)) + (d * d) / float(spike_var))
    logit = log_p1 - log_p0

    out = np.empty_like(m, dtype=bool)
    for i in range(m.shape[0]):
        li = float(logit[i])
        if li >= 0:
            p1 = 1.0 / (1.0 + np.exp(-li))
        else:
            e = np.exp(li)
            p1 = float(e / (1.0 + e))
        out[i] = bool(rng.uniform() < p1)
    return out
