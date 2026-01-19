from __future__ import annotations

import numpy as np

from .linalg import solve_psd
from .spec import ShockSpec


def _quadratic_forms(errors: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2:
        raise ValueError("errors must be a 2D array of shape (T, N)")
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim != 2 or sig.shape[0] != sig.shape[1] or sig.shape[0] != int(e.shape[1]):
        raise ValueError("sigma must have shape (N, N) matching errors.shape[1]")

    # q_t = e_t' Sigma^{-1} e_t for each t
    sol = solve_psd(sig, e.T)  # (N, T)
    q = np.sum(e.T * sol, axis=0)  # (T,)
    return np.asarray(q, dtype=float).reshape(-1)


def update_precision_scales(
    *,
    errors: np.ndarray,
    sigma: np.ndarray,
    spec: ShockSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample per-observation precision scales for robust shock models.

    The robust shock families supported here are implemented via an observation-level
    precision scale ``lambda_t`` such that:

        eps_t | lambda_t ~ Normal(0, Sigma / lambda_t)

    Returns
    -------
    lambda_ : np.ndarray
        Array of shape (T,) with strictly positive entries.
    """
    family = str(spec.family).lower()
    if family == "gaussian":
        t = int(np.asarray(errors).shape[0])
        return np.ones(t, dtype=float)

    q = _quadratic_forms(errors, sigma)  # (T,)
    n = int(np.asarray(errors).shape[1])

    if family == "student_t":
        nu = float(spec.df)
        shape = 0.5 * (nu + float(n))
        rate = 0.5 * (nu + q)
        lam = rng.gamma(shape=shape, scale=1.0 / rate, size=q.shape[0])
        return np.asarray(lam, dtype=float)

    if family == "mixture_outlier":
        prob = float(spec.outlier_prob)
        kappa = float(spec.outlier_variance)
        if not (0.0 < prob < 1.0):
            raise ValueError("outlier_prob must be in (0, 1) for mixture_outlier")
        if not (np.isfinite(kappa) and kappa > 1.0):
            raise ValueError("outlier_variance must be finite and > 1 for mixture_outlier")

        # mixture over lambda_t in {1, 1/kappa}
        log_p0 = np.log1p(-prob) - 0.5 * q
        log_p1 = np.log(prob) - 0.5 * float(n) * np.log(kappa) - 0.5 * (q / kappa)
        # p1 = exp(log_p1) / (exp(log_p0) + exp(log_p1))
        p1 = np.exp(log_p1 - np.logaddexp(log_p0, log_p1))
        z = rng.uniform(size=q.shape[0]) < p1
        lam = np.where(z, 1.0 / kappa, 1.0)
        return np.asarray(lam, dtype=float)

    raise ValueError(f"unknown shocks.family: {spec.family}")


def sample_innovation(
    *,
    sigma: np.ndarray,
    spec: ShockSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one innovation vector according to the shock specification."""
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim != 2 or sig.shape[0] != sig.shape[1] or sig.shape[0] < 1:
        raise ValueError("sigma must be a square (N, N) array")
    n = int(sig.shape[0])

    family = str(spec.family).lower()
    if family == "gaussian":
        return rng.multivariate_normal(mean=np.zeros(n, dtype=float), cov=sig)

    # Draw base Gaussian shock once and scale.
    z = rng.multivariate_normal(mean=np.zeros(n, dtype=float), cov=sig)

    if family == "student_t":
        nu = float(spec.df)
        lam = float(rng.gamma(shape=0.5 * nu, scale=2.0 / nu))
        return z / np.sqrt(lam)

    if family == "mixture_outlier":
        prob = float(spec.outlier_prob)
        kappa = float(spec.outlier_variance)
        is_outlier = bool(rng.uniform() < prob)
        return z * (np.sqrt(kappa) if is_outlier else 1.0)

    raise ValueError(f"unknown shocks.family: {spec.family}")
