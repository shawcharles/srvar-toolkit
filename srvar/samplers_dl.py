from __future__ import annotations

import numpy as np
import scipy.linalg

from .linalg import cholesky_jitter, solve_psd, symmetrize
from .rng import gamma_rate, gig_rvs, inverse_gaussian

def _dl_update(
    *,
    beta: np.ndarray,
    psi: np.ndarray,
    vartheta: np.ndarray,
    zeta: float,
    abeta: float,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Dirichlet–Laplace (DL) state update.

    Updates the DL latent variables (psi, vartheta, zeta) and returns the implied
    diagonal prior precision vector over vec(beta).

    Conventions
    -----------
    - We vectorize ``beta`` in Fortran/column-major order (``order='F'``) to match
      MATLAB's memory layout. For a (K, N) coefficient matrix this yields:

        [beta[0,0], beta[1,0], ..., beta[K-1,0], beta[0,1], ..., beta[K-1,N-1]].

      Under this convention, the block of K coefficients for equation i occupies
      indices ``i*K:(i+1)*K``.
    """
    b = np.asarray(beta, dtype=float)
    if b.ndim != 2:
        raise ValueError("beta must be 2D (K, N)")
    k, n = b.shape
    km = int(k * n)

    ps = np.asarray(psi, dtype=float).reshape(-1)
    vt = np.asarray(vartheta, dtype=float).reshape(-1)
    if ps.shape != (km,) or vt.shape != (km,):
        raise ValueError("psi/vartheta must have shape (K*N,)")

    z = float(zeta)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("zeta must be finite and > 0")

    # Vectorize in MATLAB-compatible (column-major) order.
    bvec = b.reshape(-1, order="F")
    ab = float(abeta)
    if not np.isfinite(ab) or ab <= 0:
        raise ValueError("abeta must be finite and > 0")

    # eps guards against division by zero when coefficients are (near) exactly 0.
    denom = np.abs(bvec) + eps
    nu_psi = vt * (z / denom)
    # MATLAB reference uses: invpsi ~ InvGaussian(mu=nupsibeta, lambda=1).
    invpsi = np.asarray(inverse_gaussian(mu=nu_psi, lam=1.0, rng=rng), dtype=float)
    # Defensive clipping: avoids pathological values breaking downstream linear algebra.
    invpsi = np.clip(invpsi, 1e-12, 1e12)
    psi_new = 1.0 / (invpsi + eps)
    psi_new = np.clip(psi_new, 1e-12, 1e12)

    b_sum = float(2.0 * float(np.sum(np.abs(bvec) / (vt + eps))))
    zeta_new = float(gig_rvs(p=float(km) * (ab - 1.0), a=1.0, b=b_sum + eps, rng=rng))
    zeta_new = float(np.clip(zeta_new, 1e-12, 1e12))

    big_l = np.asarray(gig_rvs(p=ab - 1.0, a=1.0, b=2.0 * np.abs(bvec) + eps, rng=rng), dtype=float)
    big_l = np.clip(big_l, 1e-12, 1e12)
    s = float(np.sum(big_l))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("DL vartheta normalization failed")
    vartheta_new = big_l / s
    vartheta_new = np.clip(vartheta_new, 1e-12, 1e12)

    inv_v0_vec = 1.0 / (psi_new * (vartheta_new * vartheta_new) * (zeta_new * zeta_new) + eps)
    inv_v0_vec = np.clip(inv_v0_vec, 1e-12, 1e12)
    return psi_new, vartheta_new, zeta_new, inv_v0_vec


def _dl_sample_beta_sigma(
    *,
    x: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    inv_v0_vec: np.ndarray,
    s0: np.ndarray,
    nu0: float,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (beta, Sigma) under a DL diagonal prior precision.

    This sampler mirrors the homoskedastic MATLAB implementation used with DL:
    equation-wise Gaussian updates for beta and inverse-gamma updates for the
    per-equation variances (Sigma is treated as diagonal).

    Indexing
    --------
    ``inv_v0_vec`` is interpreted as a length K*N vector in the same Fortran order
    as ``beta.reshape(-1, order='F')``, so the K coefficients for equation i use:

        inv_v0_vec[i*K:(i+1)*K].
    """
    xt = np.asarray(x, dtype=float)
    yt = np.asarray(y, dtype=float)
    m0t = np.asarray(m0, dtype=float)
    inv_v0 = np.asarray(inv_v0_vec, dtype=float).reshape(-1)
    s0t = np.asarray(s0, dtype=float)
    if xt.ndim != 2 or yt.ndim != 2:
        raise ValueError("x and y must be 2D")

    t, k = xt.shape
    if yt.shape[0] != t:
        raise ValueError("x and y must have the same number of rows")
    n = yt.shape[1]
    if m0t.shape != (k, n):
        raise ValueError("m0 must have shape (K, N)")
    if s0t.shape != (n, n):
        raise ValueError("s0 must have shape (N, N)")
    if inv_v0.shape != (k * n,):
        raise ValueError("inv_v0_vec must have shape (K*N,)")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive")

    xtx = xt.T @ xt
    beta = np.empty((k, n), dtype=float)
    sig2 = np.empty(n, dtype=float)

    for i in range(n):
        inv_diag = inv_v0[i * k : (i + 1) * k]
        sig2[i] = float(s0t[i, i])

        # jitter stabilizes the precision matrix when inv_diag contains extreme values.
        ktheta = symmetrize(np.diag(inv_diag) + (xtx / max(sig2[i], 1e-12)) + jitter * np.eye(k, dtype=float))
        rhs = inv_diag * m0t[:, i] + (xt.T @ yt[:, i]) / max(sig2[i], 1e-12)
        thetahat = solve_psd(ktheta, rhs)
        l = cholesky_jitter(ktheta)
        z = rng.standard_normal(k)
        beta[:, i] = thetahat + scipy.linalg.solve_triangular(l.T, z, lower=False, check_finite=False)

    resid = yt - xt @ beta
    for i in range(n):
        shape = float(nu0 + t / 2.0)
        rate = float(s0t[i, i] + 0.5 * float(np.sum(resid[:, i] ** 2)))
        sig2[i] = float(1.0 / gamma_rate(shape=shape, rate=rate, rng=rng))
        sig2[i] = float(np.clip(sig2[i], 1e-12, 1e12))

    sigma_new = np.diag(sig2)
    sigma_new = symmetrize(np.asarray(sigma_new, dtype=float))
    return beta, sigma_new


def _dl_sample_beta_svrw(
    *,
    x: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    inv_v0_vec: np.ndarray,
    h: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> np.ndarray:
    xt = np.asarray(x, dtype=float)
    yt = np.asarray(y, dtype=float)
    m0t = np.asarray(m0, dtype=float)
    inv_v0 = np.asarray(inv_v0_vec, dtype=float).reshape(-1)
    ht = np.asarray(h, dtype=float)

    if xt.ndim != 2 or yt.ndim != 2:
        raise ValueError("x and y must be 2D")
    if ht.ndim != 2:
        raise ValueError("h must be 2D")
    if yt.shape != ht.shape:
        raise ValueError("y and h must have the same shape")

    t, k = xt.shape
    _t2, n = yt.shape
    if m0t.shape != (k, n):
        raise ValueError("m0 must have shape (K, N)")
    if inv_v0.shape != (k * n,):
        raise ValueError("inv_v0_vec must have shape (K*N,)")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive")

    beta = np.empty((k, n), dtype=float)
    for i in range(n):
        inv_diag = inv_v0[i * k : (i + 1) * k]
        w = np.exp(-ht[:, i])
        xtwx = xt.T @ (w[:, None] * xt)
        ktheta = symmetrize(xtwx + np.diag(inv_diag) + jitter * np.eye(k, dtype=float))
        rhs = inv_diag * m0t[:, i] + xt.T @ (w * yt[:, i])
        thetahat = solve_psd(ktheta, rhs)
        l = cholesky_jitter(ktheta)
        z = rng.standard_normal(k)
        theta = thetahat + scipy.linalg.solve_triangular(l.T, z, lower=False, check_finite=False)
        beta[:, i] = theta
    return beta
