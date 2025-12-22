from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.special

from .linalg import cholesky_jitter, solve_psd, symmetrize


@dataclass(frozen=True, slots=True)
class VolatilitySpec:
    enabled: bool = True
    epsilon: float = 1e-4
    h0_prior_mean: float = 1e-6
    h0_prior_var: float = 10.0
    sigma_eta_prior_nu0: float = 1.0
    sigma_eta_prior_s0: float = 0.01

    def __post_init__(self) -> None:
        if self.epsilon <= 0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be positive")
        if self.h0_prior_var <= 0 or not np.isfinite(self.h0_prior_var):
            raise ValueError("h0_prior_var must be positive")
        if self.sigma_eta_prior_nu0 <= 0 or not np.isfinite(self.sigma_eta_prior_nu0):
            raise ValueError("sigma_eta_prior_nu0 must be positive")
        if self.sigma_eta_prior_s0 <= 0 or not np.isfinite(self.sigma_eta_prior_s0):
            raise ValueError("sigma_eta_prior_s0 must be positive")


_KSC_PI = np.array([0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575], dtype=float)
_KSC_MI = (
    np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819], dtype=float)
    - 1.2704
)
_KSC_SIGI = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261], dtype=float)
_KSC_SQRTSIGI = np.sqrt(_KSC_SIGI)
_LOG_KSC_PI = np.log(_KSC_PI)
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def log_e2_star(e: np.ndarray, *, epsilon: float) -> np.ndarray:
    v = np.asarray(e, dtype=float)
    return np.log(v * v + float(epsilon))


def sample_mixture_indicators(*, y_star: np.ndarray, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_star, dtype=float).reshape(-1)
    ht = np.asarray(h, dtype=float).reshape(-1)

    if y.shape != ht.shape:
        raise ValueError("y_star and h must have the same shape")

    loc = ht[:, None] + _KSC_MI[None, :]
    z = (y[:, None] - loc) / _KSC_SQRTSIGI[None, :]
    loglik = -_LOG_SQRT_2PI - np.log(_KSC_SQRTSIGI)[None, :] - 0.5 * z * z

    log_q = _LOG_KSC_PI[None, :] + loglik
    log_q = log_q - scipy.special.logsumexp(log_q, axis=1, keepdims=True)
    q = np.exp(log_q)

    u = rng.random(y.shape[0])
    cdf = np.cumsum(q, axis=1)
    return (u[:, None] < cdf).argmax(axis=1).astype(int)


def sample_h_svrw(
    *,
    y_star: np.ndarray,
    h: np.ndarray,
    sigma_eta2: float,
    h0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(y_star, dtype=float).reshape(-1)
    ht = np.asarray(h, dtype=float).reshape(-1)

    if y.shape != ht.shape:
        raise ValueError("y_star and h must have the same shape")
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        raise ValueError("sigma_eta2 must be positive")

    t = y.shape[0]

    s = sample_mixture_indicators(y_star=y, h=ht, rng=rng)
    dconst = _KSC_MI[s]
    omega = _KSC_SIGI[s]
    inv_omega = 1.0 / omega

    inv_sig = 1.0 / float(sigma_eta2)

    diag_kh = (2.0 * inv_sig) * np.ones(t, dtype=float)
    diag_kh[-1] = inv_sig
    sub_kh = (-inv_sig) * np.ones(t - 1, dtype=float)

    diag_ph = diag_kh + inv_omega

    ab = np.zeros((2, t), dtype=float)
    ab[0, :] = diag_ph
    ab[1, :-1] = sub_kh

    c = scipy.linalg.cholesky_banded(ab, lower=True, check_finite=False)

    rhs = (y - dconst) * inv_omega
    rhs[0] = rhs[0] + h0 * inv_sig

    hhat = scipy.linalg.cho_solve_banded((c, True), rhs, check_finite=False)

    z = rng.standard_normal(t)

    ab_t = np.zeros((2, t), dtype=float)
    ab_t[0, 1:] = c[1, :-1]
    ab_t[1, :] = c[0, :]

    delta = scipy.linalg.solve_banded((0, 1), ab_t, z, check_finite=False)

    return hhat + delta


def sample_h0(*, h1: float, sigma_eta2: float, prior_mean: float, prior_var: float, rng: np.random.Generator) -> float:
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        raise ValueError("sigma_eta2 must be positive")
    if prior_var <= 0 or not np.isfinite(prior_var):
        raise ValueError("prior_var must be positive")

    kh0 = (1.0 / sigma_eta2) + (1.0 / prior_var)
    h0hat = ((prior_mean / prior_var) + (h1 / sigma_eta2)) / kh0
    sd = float(np.sqrt(1.0 / kh0))
    return float(h0hat + sd * rng.normal())


def sample_sigma_eta2(
    *,
    h: np.ndarray,
    h0: float,
    nu0: float,
    s0: float,
    rng: np.random.Generator,
) -> float:
    ht = np.asarray(h, dtype=float).reshape(-1)
    if nu0 <= 0 or not np.isfinite(nu0):
        raise ValueError("nu0 must be positive")
    if s0 <= 0 or not np.isfinite(s0):
        raise ValueError("s0 must be positive")

    eh = ht - np.concatenate([np.array([h0], dtype=float), ht[:-1]])
    shape = float(nu0 + ht.shape[0] / 2.0)
    scale = float(s0 + 0.5 * float(np.sum(eh * eh)))

    return float(1.0 / rng.gamma(shape=shape, scale=1.0 / scale))


def sample_beta_svrw(
    *,
    x: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    v0: np.ndarray,
    h: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> np.ndarray:
    xt = np.asarray(x, dtype=float)
    yt = np.asarray(y, dtype=float)
    m0t = np.asarray(m0, dtype=float)
    v0t = np.asarray(v0, dtype=float)
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
    if v0t.shape != (k, k):
        raise ValueError("v0 must have shape (K, K)")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive")

    inv_v0 = solve_psd(v0t, np.eye(k, dtype=float))
    beta = np.empty((k, n), dtype=float)

    for i in range(n):
        w = np.exp(-ht[:, i])
        xtwx = xt.T @ (w[:, None] * xt)
        ktheta = symmetrize(inv_v0 + xtwx + jitter * np.eye(k, dtype=float))

        rhs = inv_v0 @ m0t[:, i] + xt.T @ (w * yt[:, i])
        thetahat = solve_psd(ktheta, rhs)

        l = cholesky_jitter(ktheta)
        z = rng.standard_normal(k)
        theta = thetahat + scipy.linalg.solve_triangular(l.T, z, lower=False, check_finite=False)
        beta[:, i] = theta

    return beta
