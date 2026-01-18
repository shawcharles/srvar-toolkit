from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats

from .linalg import cholesky_jitter, solve_psd, symmetrize


@dataclass(frozen=True, slots=True)
class VolatilitySpec:
    """Stochastic volatility configuration.

    When enabled in :class:`srvar.spec.ModelSpec`, estimation uses a diagonal
    stochastic volatility random-walk (SVRW) model for time-varying variances.

    By default, residual covariance is diagonal (independent shocks). Setting
    ``covariance='triangular'`` enables a triangular factorization:

        ``Sigma_t = Q^{-1} diag(exp(h_t)) (Q^{-1})'``

    where ``Q`` is upper-triangular with ones on the diagonal. This yields a
    full residual covariance matrix with time-varying variances and a time-invariant
    correlation structure.

    Parameters
    ----------
    enabled:
        Whether stochastic volatility is enabled.
    dynamics:
        Volatility state dynamics. ``"rw"`` uses a random-walk prior:

        ``h_t = h_{t-1} + eta_t``

        ``"ar1"`` uses an AR(1) prior:

        ``h_t = gamma0 + phi * h_{t-1} + eta_t``

        with ``phi`` and ``gamma0`` sampled in the Gibbs routine.
    covariance:
        Covariance structure. ``"diagonal"`` is independent shocks. ``"triangular"``
        uses an upper-triangular factor ``Q`` with ones on the diagonal (a CCCM-style
        multivariate SV factorization).
    q_prior_var:
        Prior variance for the off-diagonal elements of ``Q`` when
        ``covariance='triangular'``.
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
    phi_prior_mean, phi_prior_var:
        Prior mean/variance for the AR(1) coefficient ``phi`` when ``dynamics='ar1'``.
        A normal prior is used with truncation to ``(-1, 1)``.
    gamma0_prior_mean, gamma0_prior_var:
        Prior mean/variance for the AR(1) intercept ``gamma0`` when ``dynamics='ar1'``.
    """

    enabled: bool = True
    dynamics: Literal["rw", "ar1"] = "rw"
    covariance: Literal["diagonal", "triangular"] = "diagonal"
    q_prior_var: float = 1.0
    epsilon: float = 1e-4
    h0_prior_mean: float = 1e-6
    h0_prior_var: float = 10.0
    sigma_eta_prior_nu0: float = 1.0
    sigma_eta_prior_s0: float = 0.01
    phi_prior_mean: float = 0.95
    phi_prior_var: float = 0.1
    gamma0_prior_mean: float = 0.0
    gamma0_prior_var: float = 10.0

    def __post_init__(self) -> None:
        if self.dynamics not in {"rw", "ar1"}:
            raise ValueError("dynamics must be one of {'rw','ar1'}")
        if self.covariance not in {"diagonal", "triangular"}:
            raise ValueError("covariance must be one of {'diagonal','triangular'}")
        if self.covariance == "triangular" and (
            self.q_prior_var <= 0 or not np.isfinite(self.q_prior_var)
        ):
            raise ValueError("q_prior_var must be positive and finite when covariance='triangular'")
        if self.epsilon <= 0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be positive")
        if self.h0_prior_var <= 0 or not np.isfinite(self.h0_prior_var):
            raise ValueError("h0_prior_var must be positive")
        if self.sigma_eta_prior_nu0 <= 0 or not np.isfinite(self.sigma_eta_prior_nu0):
            raise ValueError("sigma_eta_prior_nu0 must be positive")
        if self.sigma_eta_prior_s0 <= 0 or not np.isfinite(self.sigma_eta_prior_s0):
            raise ValueError("sigma_eta_prior_s0 must be positive")
        if self.dynamics == "ar1":
            for name in [
                "phi_prior_mean",
                "phi_prior_var",
                "gamma0_prior_mean",
                "gamma0_prior_var",
            ]:
                v = float(getattr(self, name))
                if not np.isfinite(v):
                    raise ValueError(f"{name} must be finite when dynamics='ar1'")
            if self.phi_prior_var <= 0:
                raise ValueError("phi_prior_var must be positive when dynamics='ar1'")
            if self.gamma0_prior_var <= 0:
                raise ValueError("gamma0_prior_var must be positive when dynamics='ar1'")


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


def sample_mixture_indicators(
    *, y_star: np.ndarray, h: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
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
        return _sample_mixture_indicators_numba(
            y, ht, u, _KSC_MI, _KSC_SQRTSIGI, _LOG_KSC_PI
        ).astype(int)

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


def sample_h0(
    *, h1: float, sigma_eta2: float, prior_mean: float, prior_var: float, rng: np.random.Generator
) -> float:
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        raise ValueError("sigma_eta2 must be positive")
    if prior_var <= 0 or not np.isfinite(prior_var):
        raise ValueError("prior_var must be positive")

    kh0 = (1.0 / sigma_eta2) + (1.0 / prior_var)
    h0hat = ((prior_mean / prior_var) + (h1 / sigma_eta2)) / kh0
    sd = float(np.sqrt(1.0 / kh0))
    return float(h0hat + sd * rng.normal())


def _truncnorm_rvs(
    *,
    mean: float,
    sd: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> float:
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("sd must be positive")
    if not np.isfinite(lower) or not np.isfinite(upper) or not (lower < upper):
        raise ValueError("invalid truncation bounds")
    a = (lower - mean) / sd
    b = (upper - mean) / sd
    return float(scipy.stats.truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, random_state=rng))


def sample_h_ar1(
    *,
    y_star: np.ndarray,
    h: np.ndarray,
    sigma_eta2: float,
    h0: float,
    gamma0: float,
    phi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample log-volatilities under an AR(1) state equation.

    State equation:
        h_t = gamma0 + phi * h_{t-1} + eta_t,   eta_t ~ N(0, sigma_eta2)

    Observation equation uses the KSC mixture approximation as in :func:`sample_h_svrw`.
    """
    y = np.asarray(y_star, dtype=float).reshape(-1)
    ht = np.asarray(h, dtype=float).reshape(-1)
    if y.shape != ht.shape:
        raise ValueError("y_star and h must have the same shape")
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        raise ValueError("sigma_eta2 must be positive")
    if not np.isfinite(h0) or not np.isfinite(gamma0) or not np.isfinite(phi):
        raise ValueError("h0, gamma0, and phi must be finite")
    if not (-1.0 < float(phi) < 1.0):
        raise ValueError("phi must be in (-1, 1)")

    t = int(y.shape[0])

    s = sample_mixture_indicators(y_star=y, h=ht, rng=rng)
    dconst = _KSC_MI[s]
    omega = _KSC_SIGI[s]
    inv_omega = 1.0 / omega

    inv_sig = 1.0 / float(sigma_eta2)

    diag_kh = (1.0 + float(phi) ** 2) * inv_sig * np.ones(t, dtype=float)
    diag_kh[-1] = inv_sig
    if t == 1:
        diag_kh[0] = inv_sig

    sub_kh = (-float(phi) * inv_sig) * np.ones(max(t - 1, 0), dtype=float)

    diag_ph = diag_kh + inv_omega

    ab = np.zeros((2, t), dtype=float)
    ab[0, :] = diag_ph
    if t > 1:
        ab[1, :-1] = sub_kh

    c = scipy.linalg.cholesky_banded(ab, lower=True, check_finite=False)

    rhs = (y - dconst) * inv_omega

    r = np.full(t, float(gamma0), dtype=float)
    r[0] = float(gamma0) + float(phi) * float(h0)
    at_r = np.empty(t, dtype=float)
    if t == 1:
        at_r[0] = r[0]
    else:
        at_r[0] = r[0] - float(phi) * r[1]
        if t > 2:
            at_r[1:-1] = r[1:-1] - float(phi) * r[2:]
        at_r[-1] = r[-1]

    rhs = rhs + inv_sig * at_r

    hhat = scipy.linalg.cho_solve_banded((c, True), rhs, check_finite=False)

    z = rng.standard_normal(t)

    ab_t = np.zeros((2, t), dtype=float)
    if t > 1:
        ab_t[0, 1:] = c[1, :-1]
    ab_t[1, :] = c[0, :]

    delta = scipy.linalg.solve_banded((0, 1), ab_t, z, check_finite=False)

    return hhat + delta


def sample_h0_ar1(
    *,
    h1: float,
    sigma_eta2: float,
    gamma0: float,
    phi: float,
    prior_mean: float,
    prior_var: float,
    rng: np.random.Generator,
) -> float:
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        raise ValueError("sigma_eta2 must be positive")
    if prior_var <= 0 or not np.isfinite(prior_var):
        raise ValueError("prior_var must be positive")
    if (
        not np.isfinite(h1)
        or not np.isfinite(gamma0)
        or not np.isfinite(phi)
        or not np.isfinite(prior_mean)
    ):
        raise ValueError("inputs must be finite")

    k = (float(phi) ** 2) / float(sigma_eta2) + 1.0 / float(prior_var)
    h0hat = (
        (float(prior_mean) / float(prior_var))
        + (float(phi) * (float(h1) - float(gamma0)) / float(sigma_eta2))
    ) / k
    sd = float(np.sqrt(1.0 / k))
    return float(h0hat + sd * rng.normal())


def sample_sigma_eta2_ar1(
    *,
    h: np.ndarray,
    h0: float,
    gamma0: float,
    phi: float,
    nu0: float,
    s0: float,
    rng: np.random.Generator,
) -> float:
    ht = np.asarray(h, dtype=float).reshape(-1)
    if nu0 <= 0 or not np.isfinite(nu0):
        raise ValueError("nu0 must be positive")
    if s0 <= 0 or not np.isfinite(s0):
        raise ValueError("s0 must be positive")
    if not np.isfinite(h0) or not np.isfinite(gamma0) or not np.isfinite(phi):
        raise ValueError("h0, gamma0, and phi must be finite")

    prev = np.concatenate([np.array([float(h0)], dtype=float), ht[:-1]])
    eh = ht - (float(gamma0) + float(phi) * prev)
    shape = float(nu0 + ht.shape[0] / 2.0)
    scale = float(s0 + 0.5 * float(np.sum(eh * eh)))
    return float(1.0 / rng.gamma(shape=shape, scale=1.0 / scale))


def sample_ar1_params(
    *,
    h: np.ndarray,
    h0: float,
    sigma_eta2: float,
    phi_prior_mean: float,
    phi_prior_var: float,
    gamma0_prior_mean: float,
    gamma0_prior_var: float,
    rng: np.random.Generator,
    phi_bounds: tuple[float, float] = (-0.999, 0.999),
) -> tuple[float, float]:
    """Sample (gamma0, phi) for the AR(1) volatility state equation."""
    ht = np.asarray(h, dtype=float).reshape(-1)
    if ht.size < 1:
        raise ValueError("h must be non-empty")
    if sigma_eta2 <= 0 or not np.isfinite(sigma_eta2):
        raise ValueError("sigma_eta2 must be positive")
    if phi_prior_var <= 0 or gamma0_prior_var <= 0:
        raise ValueError("prior variances must be > 0")
    if not np.isfinite(h0):
        raise ValueError("h0 must be finite")
    for name, v in [
        ("phi_prior_mean", phi_prior_mean),
        ("phi_prior_var", phi_prior_var),
        ("gamma0_prior_mean", gamma0_prior_mean),
        ("gamma0_prior_var", gamma0_prior_var),
    ]:
        if not np.isfinite(float(v)):
            raise ValueError(f"{name} must be finite")

    phi_lo, phi_hi = float(phi_bounds[0]), float(phi_bounds[1])
    if not (-1.0 < phi_lo < phi_hi < 1.0):
        raise ValueError("phi_bounds must be inside (-1, 1)")

    y = ht
    x = np.concatenate([np.array([float(h0)], dtype=float), ht[:-1]])

    phi = float(phi_prior_mean)
    for _ in range(2):
        k_g0 = (1.0 / float(gamma0_prior_var)) + (float(y.shape[0]) / float(sigma_eta2))
        g0_hat = (
            (float(gamma0_prior_mean) / float(gamma0_prior_var))
            + (1.0 / float(sigma_eta2)) * float(np.sum(y - phi * x))
        ) / k_g0
        gamma0 = float(g0_hat + np.sqrt(1.0 / k_g0) * rng.normal())

        k_phi = (1.0 / float(phi_prior_var)) + (1.0 / float(sigma_eta2)) * float(np.sum(x * x))
        phi_hat = (
            (float(phi_prior_mean) / float(phi_prior_var))
            + (1.0 / float(sigma_eta2)) * float(np.sum(x * (y - gamma0)))
        ) / k_phi
        sd_phi = float(np.sqrt(1.0 / k_phi))
        phi = _truncnorm_rvs(mean=float(phi_hat), sd=sd_phi, lower=phi_lo, upper=phi_hi, rng=rng)

    return float(gamma0), float(phi)


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

        chol = cholesky_jitter(ktheta)
        z = rng.standard_normal(k)
        theta = thetahat + scipy.linalg.solve_triangular(chol.T, z, lower=False, check_finite=False)
        beta[:, i] = theta

    return beta
