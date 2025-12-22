from __future__ import annotations

import numpy as np

from .linalg import solve_psd


def v0_diag_from_gamma(
    *,
    gamma: np.ndarray,
    spike_var: float,
    slab_var: float,
    intercept_slab_var: float | None = None,
) -> np.ndarray:
    """Compute the diagonal of V0 implied by spike-and-slab indicators.

    Parameters
    ----------
    gamma:
        Boolean inclusion indicators of shape ``(K,)``.
    spike_var:
        Variance for excluded predictors.
    slab_var:
        Variance for included predictors.
    intercept_slab_var:
        Optional override for the intercept variance (index 0).

    Returns
    -------
    np.ndarray
        Vector of length ``K`` representing the diagonal of ``V0``.
    """
    g = np.asarray(gamma, dtype=bool)
    if g.ndim != 1:
        raise ValueError("gamma must be a 1D array")

    if spike_var <= 0 or slab_var <= 0:
        raise ValueError("spike_var and slab_var must be > 0")

    v = np.where(g, float(slab_var), float(spike_var)).astype(float, copy=False)

    if intercept_slab_var is not None:
        if intercept_slab_var <= 0:
            raise ValueError("intercept_slab_var must be > 0")
        if v.size < 1:
            raise ValueError("gamma must be non-empty when intercept_slab_var is provided")
        v[0] = float(intercept_slab_var)

    return v


def sample_gamma_rows(
    *,
    beta: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    spike_var: float,
    slab_var: float,
    inclusion_prob: float,
    fixed_mask: np.ndarray | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample SSVS inclusion indicators for coefficient rows.

    This updates ``gamma`` given the current coefficient draw ``beta`` and covariance
    ``sigma`` under a spike-and-slab prior.

    Parameters
    ----------
    beta:
        VAR coefficient matrix of shape ``(K, N)``.
    sigma:
        Innovation covariance matrix of shape ``(N, N)``.
    gamma:
        Current inclusion indicators of shape ``(K,)``.
    spike_var, slab_var:
        Spike-and-slab prior variances.
    inclusion_prob:
        Prior inclusion probability.
    fixed_mask:
        Optional boolean mask indicating predictors which are forced to stay included.
    rng:
        NumPy RNG.

    Returns
    -------
    np.ndarray
        Updated boolean indicators of shape ``(K,)``.
    """
    b = np.asarray(beta, dtype=float)
    s = np.asarray(sigma, dtype=float)
    g = np.asarray(gamma, dtype=bool)

    if b.ndim != 2:
        raise ValueError("beta must be 2D (K, N)")
    k, n = b.shape

    if s.shape != (n, n):
        raise ValueError("sigma must have shape (N, N)")
    if g.shape != (k,):
        raise ValueError("gamma must have shape (K,)")

    if spike_var <= 0 or slab_var <= 0:
        raise ValueError("spike_var and slab_var must be > 0")
    if not (0.0 < inclusion_prob < 1.0):
        raise ValueError("inclusion_prob must be in (0, 1)")

    if fixed_mask is None:
        fixed = np.zeros(k, dtype=bool)
    else:
        fixed = np.asarray(fixed_mask, dtype=bool)
        if fixed.shape != (k,):
            raise ValueError("fixed_mask must have shape (K,)")

    inv_sigma = solve_psd(s, np.eye(n, dtype=float))

    log_prior_odds = float(np.log(inclusion_prob) - np.log(1.0 - inclusion_prob))
    log_det_ratio = float((n / 2.0) * np.log(slab_var / spike_var))
    quad_coef = float(0.5 * (1.0 / slab_var - 1.0 / spike_var))

    out = g.copy()

    for r in range(k):
        if fixed[r]:
            continue

        br = b[r, :]
        q = float(br @ inv_sigma @ br)

        logit = log_prior_odds - log_det_ratio - quad_coef * q
        if logit >= 0:
            p1 = 1.0 / (1.0 + np.exp(-logit))
        else:
            e = np.exp(logit)
            p1 = float(e / (1.0 + e))

        out[r] = bool(rng.uniform() < p1)

    return out
