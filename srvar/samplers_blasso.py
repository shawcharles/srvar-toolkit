from __future__ import annotations

import numpy as np

from .rng import gamma_rate, inverse_gaussian

def _blasso_v0_from_state(*, tau: np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=float).reshape(-1)
    if t.ndim != 1:
        raise ValueError("tau must be 1D")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("tau must be finite and > 0")
    return np.diag(t)


def _blasso_update_global(
    *,
    beta: np.ndarray,
    tau: np.ndarray,
    lambda_: float,
    a0: float,
    b0: float,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    b = np.asarray(beta, dtype=float)
    t = np.asarray(tau, dtype=float).reshape(-1)
    if b.ndim != 2:
        raise ValueError("beta must be 2D (K, N)")
    if b.shape[0] != t.shape[0]:
        raise ValueError("beta.shape[0] must match tau.shape[0]")

    rate = float(b0 + 0.5 * float(np.sum(t)))
    shape = float(a0 + t.shape[0])
    lam_new = float(gamma_rate(shape=shape, rate=rate, rng=rng))

    # Row-wise shrinkage compatible with a matrix-normal prior.
    row_energy = np.sum(b * b, axis=1)
    stau = np.sqrt(lam_new / (row_energy + eps))
    invtau = np.asarray(inverse_gaussian(mu=stau, lam=lam_new, rng=rng), dtype=float)
    invtau = np.clip(invtau, 1e-6, 1e6)
    tau_new = 1.0 / invtau
    tau_new = np.clip(tau_new, 1e-12, 1e12)
    return tau_new, lam_new


def _blasso_update_adaptive(
    *,
    beta: np.ndarray,
    tau: np.ndarray,
    lambda_c: float,
    lambda_L: float,
    a0_c: float,
    b0_c: float,
    a0_L: float,
    b0_L: float,
    c_mask: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float, float]:
    b = np.asarray(beta, dtype=float)
    t = np.asarray(tau, dtype=float).reshape(-1)
    cm = np.asarray(c_mask, dtype=bool).reshape(-1)
    if b.ndim != 2:
        raise ValueError("beta must be 2D (K, N)")
    if b.shape[0] != t.shape[0] or b.shape[0] != cm.shape[0]:
        raise ValueError("beta.shape[0] must match tau/c_mask shape")

    # group masks
    l_mask = ~cm

    lam_c_new = float(lambda_c)
    lam_L_new = float(lambda_L)

    if int(np.sum(cm)) > 0:
        rate_c = float(b0_c + 0.5 * float(np.sum(t[cm])))
        shape_c = float(a0_c + int(np.sum(cm)))
        lam_c_new = float(gamma_rate(shape=shape_c, rate=rate_c, rng=rng))

    if int(np.sum(l_mask)) > 0:
        rate_L = float(b0_L + 0.5 * float(np.sum(t[l_mask])))
        shape_L = float(a0_L + int(np.sum(l_mask)))
        lam_L_new = float(gamma_rate(shape=shape_L, rate=rate_L, rng=rng))

    lam_vec = np.where(cm, lam_c_new, lam_L_new)

    row_energy = np.sum(b * b, axis=1)
    stau = np.sqrt(lam_vec / (row_energy + eps))
    invtau = np.asarray(inverse_gaussian(mu=stau, lam=lam_vec, rng=rng), dtype=float)
    invtau = np.clip(invtau, 1e-6, 1e6)
    tau_new = 1.0 / invtau
    tau_new = np.clip(tau_new, 1e-12, 1e12)

    return tau_new, lam_c_new, lam_L_new
