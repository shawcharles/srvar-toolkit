from __future__ import annotations

import numpy as np
import scipy.linalg

from .bvar import posterior_niw
from .data.dataset import Dataset
from .linalg import cholesky_jitter, solve_psd, symmetrize
from .results import FitResult
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .sv import log_e2_star, sample_beta_svrw, sample_h0, sample_h_svrw, sample_sigma_eta2
from .var import design_matrix


def _normalize_factor_signs(*, lam: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize factor signs by enforcing positive diagonal loadings.

    Identification convention:
    - lower-triangular structure on the first k rows (enforced in sampling)
    - positive diagonal on the first k rows via sign flips
    """
    lt = np.asarray(lam, dtype=float)
    ft = np.asarray(f, dtype=float)
    if lt.ndim != 2 or ft.ndim != 2:
        raise ValueError("lam and f must be 2D arrays")
    if lt.shape[1] != ft.shape[1]:
        raise ValueError("lam and f must agree on factor dimension (k)")

    n, k = lt.shape
    _t, k2 = ft.shape
    if k2 != k:
        raise ValueError("lam and f must agree on factor dimension (k)")
    if k > n:
        raise ValueError("k must be <= N for sign normalization")

    lam_out = lt.copy()
    f_out = ft.copy()
    for j in range(k):
        if lam_out[j, j] < 0.0:
            lam_out[:, j] *= -1.0
            f_out[:, j] *= -1.0
    return lam_out, f_out


def _sample_factors(
    *,
    e: np.ndarray,
    lam: np.ndarray,
    h_eta: np.ndarray,
    h_f: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Sample factor paths f_t given residuals and current parameters."""
    et = np.asarray(e, dtype=float)
    lt = np.asarray(lam, dtype=float)
    h_et = np.asarray(h_eta, dtype=float)
    h_ft = np.asarray(h_f, dtype=float)

    if et.ndim != 2:
        raise ValueError("e must be 2D")
    if lt.ndim != 2:
        raise ValueError("lam must be 2D")
    if h_et.ndim != 2 or h_ft.ndim != 2:
        raise ValueError("h_eta and h_f must be 2D")
    if et.shape != h_et.shape:
        raise ValueError("e and h_eta must have the same shape (T, N)")

    t_eff, n = et.shape
    n_lam, k = lt.shape
    if n_lam != n:
        raise ValueError("lam must have shape (N, k)")
    if h_ft.shape != (t_eff, k):
        raise ValueError("h_f must have shape (T, k)")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive and finite")

    f = np.empty((t_eff, k), dtype=float)
    eye_k = np.eye(k, dtype=float)
    for t in range(t_eff):
        w = np.exp(-h_et[t, :])  # (N,)
        lw = w[:, None] * lt  # (N, k)
        kf = lt.T @ lw  # (k, k)
        kf = symmetrize(kf + np.diag(np.exp(-h_ft[t, :])) + jitter * eye_k)

        rhs = lt.T @ (w * et[t, :])
        fhat = solve_psd(kf, rhs)

        chol = cholesky_jitter(kf)
        z = rng.standard_normal(k)
        delta = scipy.linalg.solve_triangular(chol.T, z, lower=False, check_finite=False)
        f[t, :] = fhat + delta

    return f


def _sample_loadings(
    *,
    e: np.ndarray,
    f: np.ndarray,
    h_eta: np.ndarray,
    loading_prior_var: float,
    rng: np.random.Generator,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Sample factor loading matrix Lambda row-by-row."""
    et = np.asarray(e, dtype=float)
    ft = np.asarray(f, dtype=float)
    h_et = np.asarray(h_eta, dtype=float)

    if et.ndim != 2 or ft.ndim != 2 or h_et.ndim != 2:
        raise ValueError("e, f, and h_eta must be 2D arrays")
    if et.shape != h_et.shape:
        raise ValueError("e and h_eta must have the same shape (T, N)")
    t_eff, n = et.shape
    if ft.shape[0] != t_eff:
        raise ValueError("f must have shape (T, k)")
    k = int(ft.shape[1])
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError("k must be <= N")
    if loading_prior_var <= 0 or not np.isfinite(loading_prior_var):
        raise ValueError("loading_prior_var must be positive and finite")
    if jitter <= 0 or not np.isfinite(jitter):
        raise ValueError("jitter must be positive and finite")

    lam = np.zeros((n, k), dtype=float)
    prior_prec = 1.0 / float(loading_prior_var)

    for i in range(n):
        # Identification: lower-triangular first k rows.
        # For i < k, only loadings 0..i are free.
        m = k if i >= k else (i + 1)
        f_i = ft[:, :m]

        w = np.exp(-h_et[:, i])
        k_lam = f_i.T @ (w[:, None] * f_i)
        k_lam = symmetrize(k_lam + (prior_prec + jitter) * np.eye(m, dtype=float))

        rhs = f_i.T @ (w * et[:, i])
        mean = solve_psd(k_lam, rhs)

        chol = cholesky_jitter(k_lam)
        z = rng.standard_normal(m)
        delta = scipy.linalg.solve_triangular(chol.T, z, lower=False, check_finite=False)

        lam[i, :m] = mean + delta

    return lam


def _fit_fsv(
    *,
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    rng: np.random.Generator,
) -> FitResult:
    """Fit a VAR with factor stochastic volatility (FSV).

    v1 scope:
    - volatility.dynamics == "rw"
    - volatility.covariance == "factor"
    - prior.family == "niw"
    - no ELB, no steady_state, no robust shocks
    """
    vol = model.volatility
    if vol is None or not vol.enabled:
        raise ValueError("volatility must be enabled")
    if vol.covariance != "factor":
        raise ValueError("volatility.covariance must be 'factor' for _fit_fsv")
    if vol.dynamics != "rw":
        raise ValueError("FSV currently supports only volatility.dynamics='rw'")

    if model.elb is not None and model.elb.enabled:
        raise ValueError("FSV is not yet supported with ELB (model.elb)")
    if model.steady_state is not None:
        raise ValueError("FSV is not yet supported with steady_state (model.steady_state)")
    if model.shocks is not None and model.shocks.family != "gaussian":
        raise ValueError("FSV is not yet supported with robust shocks (model.shocks)")

    prior_family = prior.family.lower()
    if prior_family != "niw":
        raise ValueError("FSV currently supports only prior.family='niw'")

    y_raw = np.asarray(dataset.values, dtype=float)
    if y_raw.ndim != 2:
        raise ValueError("dataset.values must be 2D")
    if y_raw.shape[0] <= model.p:
        raise ValueError("dataset is too short for requested lag order p")

    x, y = design_matrix(y_raw, model.p, include_intercept=model.include_intercept)
    t_eff, n = y.shape

    k = int(vol.k_factors)
    if k > n:
        raise ValueError("model.volatility.k_factors must be <= N")

    niw = prior.niw
    if niw.m0.shape != (x.shape[1], n):
        raise ValueError("prior.niw.m0 has incompatible shape for dataset/model")
    if niw.v0.shape != (x.shape[1], x.shape[1]):
        raise ValueError("prior.niw.v0 has incompatible shape for dataset/model")

    mn, _vn, _sn, _nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
    beta = mn.copy()

    # Initialize volatility states (idiosyncratic and factor)
    resid0 = y - x @ beta
    h0_eta = np.log(np.var(resid0, axis=0) + 1e-12)
    h_eta = np.tile(h0_eta.reshape(1, -1), (t_eff, 1))
    sigma_eta2_eta = 0.05 * np.ones(n, dtype=float)

    h0_f = np.zeros(k, dtype=float)
    h_f = np.tile(h0_f.reshape(1, -1), (t_eff, 1))
    sigma_eta2_f = 0.05 * np.ones(k, dtype=float)

    # Initialize loadings and factors
    lam = np.zeros((n, k), dtype=float)
    for j in range(k):
        lam[j, j] = 0.1
    f = rng.normal(size=(t_eff, k)) * np.exp(0.5 * h_f)

    beta_keep: list[np.ndarray] = []
    lam_keep: list[np.ndarray] = []
    h_eta_keep: list[np.ndarray] = []
    h0_eta_keep: list[np.ndarray] = []
    sigma_eta2_eta_keep: list[np.ndarray] = []
    h_f_keep: list[np.ndarray] = []
    h0_f_keep: list[np.ndarray] = []
    sigma_eta2_f_keep: list[np.ndarray] = []
    f_keep: list[np.ndarray] | None = [] if vol.store_factor_draws else None

    for it in range(sampler.draws):
        # Step A: sample beta given (f, lam, h_eta)
        y_tilde = y - (f @ lam.T)
        beta = sample_beta_svrw(x=x, y=y_tilde, m0=niw.m0, v0=niw.v0, h=h_eta, rng=rng)

        # Reduced-form residuals
        e = y - x @ beta

        # Step B: sample factors given (e, lam, h_eta, h_f)
        f = _sample_factors(e=e, lam=lam, h_eta=h_eta, h_f=h_f, rng=rng)

        # Step C: sample loadings given (e, f, h_eta)
        lam = _sample_loadings(
            e=e, f=f, h_eta=h_eta, loading_prior_var=vol.loading_prior_var, rng=rng
        )
        lam, f = _normalize_factor_signs(lam=lam, f=f)

        # Step D: sample idiosyncratic vol states from eta = e - lam f
        eta = e - (f @ lam.T)
        for i in range(n):
            y_star = log_e2_star(eta[:, i], epsilon=vol.epsilon)
            h_eta[:, i] = sample_h_svrw(
                y_star=y_star,
                h=h_eta[:, i],
                sigma_eta2=float(sigma_eta2_eta[i]),
                h0=float(h0_eta[i]),
                rng=rng,
            )
            h0_eta[i] = sample_h0(
                h1=float(h_eta[0, i]),
                sigma_eta2=float(sigma_eta2_eta[i]),
                prior_mean=float(vol.h0_prior_mean),
                prior_var=float(vol.h0_prior_var),
                rng=rng,
            )
            sigma_eta2_eta[i] = sample_sigma_eta2(
                h=h_eta[:, i],
                h0=float(h0_eta[i]),
                nu0=float(vol.sigma_eta_prior_nu0),
                s0=float(vol.sigma_eta_prior_s0),
                rng=rng,
            )

        # Step E: sample factor vol states from factors
        for j in range(k):
            y_star = log_e2_star(f[:, j], epsilon=vol.epsilon)
            h_f[:, j] = sample_h_svrw(
                y_star=y_star,
                h=h_f[:, j],
                sigma_eta2=float(sigma_eta2_f[j]),
                h0=float(h0_f[j]),
                rng=rng,
            )
            h0_f[j] = sample_h0(
                h1=float(h_f[0, j]),
                sigma_eta2=float(sigma_eta2_f[j]),
                prior_mean=float(vol.h0_prior_mean),
                prior_var=float(vol.h0_prior_var),
                rng=rng,
            )
            sigma_eta2_f[j] = sample_sigma_eta2(
                h=h_f[:, j],
                h0=float(h0_f[j]),
                nu0=float(vol.sigma_eta_prior_nu0),
                s0=float(vol.sigma_eta_prior_s0),
                rng=rng,
            )

        if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
            beta_keep.append(beta.copy())
            lam_keep.append(lam.copy())
            h_eta_keep.append(h_eta.copy())
            h0_eta_keep.append(h0_eta.copy())
            sigma_eta2_eta_keep.append(sigma_eta2_eta.copy())
            h_f_keep.append(h_f.copy())
            h0_f_keep.append(h0_f.copy())
            sigma_eta2_f_keep.append(sigma_eta2_f.copy())
            if f_keep is not None:
                f_keep.append(f.copy())

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=np.stack(beta_keep) if beta_keep else None,
        sigma_draws=None,
        q_draws=None,
        h_draws=np.stack(h_eta_keep) if h_eta_keep else None,
        h0_draws=np.stack(h0_eta_keep) if h0_eta_keep else None,
        sigma_eta2_draws=np.stack(sigma_eta2_eta_keep) if sigma_eta2_eta_keep else None,
        lambda_draws=np.stack(lam_keep) if lam_keep else None,
        factor_draws=np.stack(f_keep) if f_keep else None,
        h_factor_draws=np.stack(h_f_keep) if h_f_keep else None,
        h0_factor_draws=np.stack(h0_f_keep) if h0_f_keep else None,
        sigma_eta2_factor_draws=np.stack(sigma_eta2_f_keep) if sigma_eta2_f_keep else None,
    )

