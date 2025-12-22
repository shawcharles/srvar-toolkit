from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import scipy.linalg
import scipy.special

from .linalg import cholesky_jitter, solve_psd, symmetrize


@dataclass(frozen=True, slots=True)
class VolatilitySpec:
    """Stochastic volatility configuration (SV random walk; diagonal).

    When enabled in :class:`srvar.spec.ModelSpec`, estimation uses a diagonal
    stochastic volatility random-walk (SVRW) model for the VAR residual variances.

    Parameters
    ----------
    enabled:
        Whether stochastic volatility is enabled.
    epsilon:
        Small positive constant used in the transform
        ``log(e_t^2 + epsilon)`` to avoid ``log(0)``.
    h0_prior_mean:
        Prior mean for the initial log-variance state ``h0``.
    h0_prior_var:
        Prior variance for the initial log-variance state ``h0``.
    sigma_eta_prior_nu0, sigma_eta_prior_s0:
        Prior hyperparameters for the innovation variance of the log-volatility
        random walk.
    """
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

try:
    import numba as _nb

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _nb = None
    _HAVE_NUMBA = False


def _numba_enabled() -> bool:
    v = os.getenv("SRVAR_USE_NUMBA", "0").strip().lower()
    return v in {"1", "true", "yes", "on"}


if _HAVE_NUMBA:

    @_nb.njit(cache=True)
    def _sample_mixture_indicators_numba(
        y: np.ndarray,
        ht: np.ndarray,
        u: np.ndarray,
        ksc_mi: np.ndarray,
        ksc_sqrtsigi: np.ndarray,
        log_ksc_pi: np.ndarray,
    ) -> np.ndarray:
        t = y.shape[0]
        out = np.empty(t, dtype=np.int64)
        log_sqrtsigi = np.log(ksc_sqrtsigi)

        for i in range(t):
            maxv = -1.0e300
            log_q = np.empty(7, dtype=np.float64)

            for k in range(7):
                loc = ht[i] + ksc_mi[k]
                z = (y[i] - loc) / ksc_sqrtsigi[k]
                loglik = -_LOG_SQRT_2PI - log_sqrtsigi[k] - 0.5 * z * z
                v = log_ksc_pi[k] + loglik
                log_q[k] = v
                if v > maxv:
                    maxv = v

            s = 0.0
            for k in range(7):
                s += np.exp(log_q[k] - maxv)

            c = 0.0
            target = u[i]
            chosen = 0
            for k in range(7):
                c += np.exp(log_q[k] - maxv) / s
                if target <= c:
                    chosen = k
                    break

            out[i] = chosen

        return out


def log_e2_star(e: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Compute the log-squared residual transform used for SV mixture sampling.

    Given residuals ``e_t``, this returns ``log(e_t^2 + epsilon)``. The small ``epsilon``
    avoids ``log(0)`` and stabilizes sampling when residuals are extremely small.
    """
    v = np.asarray(e, dtype=float)
    return np.log(v * v + float(epsilon))


def sample_mixture_indicators(*, y_star: np.ndarray, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample KSC mixture indicators for the log-chi-square approximation.

    This implements the standard Kim-Shephard-Chib (KSC) 7-component Gaussian mixture
    approximation for the observation equation arising from:

        y*_t = log(e_t^2)  with  e_t ~ N(0, exp(h_t))

    Conditional on a discrete indicator ``s_t`` taking values in {0, ..., 6}, the model is:

        y*_t = h_t + m_{s_t} + u_t,   u_t ~ N(0, v_{s_t})

    where ``(pi, m, v)`` are fixed mixture weights/means/variances.

    Notes:
        The constant ``1.2704`` used in the mixture mean vector definition corresponds to
        ``E[log(chi^2_1)]`` and is used to match the centered form commonly reported for the
        KSC approximation.

    References:
        Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood
        inference and comparison with ARCH models.

    Returns:
        Integer array of shape (T,) with entries in {0, ..., 6}.
    """
    y = np.asarray(y_star, dtype=float).reshape(-1)
    ht = np.asarray(h, dtype=float).reshape(-1)

    if y.shape != ht.shape:
        raise ValueError("y_star and h must have the same shape")

    if _HAVE_NUMBA and _numba_enabled():
        u = rng.random(y.shape[0])
        return _sample_mixture_indicators_numba(y, ht, u, _KSC_MI, _KSC_SQRTSIGI, _LOG_KSC_PI).astype(int)

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
    """Sample a full path of log-volatilities under an SV random-walk prior.

    This updates ``h = (h_1, ..., h_T)`` in the SVRW model

        y*_t = log(e_t^2) \approx h_t + m_{s_t} + u_t
        h_t = h_{t-1} + eta_t,    eta_t ~ N(0, sigma_eta2)

    using the KSC mixture approximation and a precision-based Gaussian sampler for the
    resulting banded linear system.

    Args:
        y_star: Transformed residuals (T,).
        h: Current log-volatility values (T,).
        sigma_eta2: Innovation variance for the random walk.
        h0: Initial state value used in the prior for h_1.
        rng: NumPy RNG.

    Returns:
        Updated log-volatility path with shape (T,).
    """
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
