from __future__ import annotations

import numpy as np
import scipy.linalg

from .bvar import posterior_niw
from .data.dataset import Dataset
from .elb import sample_shadow_value_fsv
from .linalg import cholesky_jitter, solve_psd, symmetrize
from .results import FitResult
from .samplers_ssp import (
    _asum_from_beta,
    _strip_intercept_niw_blocks,
    sample_mu_gamma,
    sample_steady_state_mu_fsv,
)
from .shocks import update_precision_scales_factor_sv
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .sv import log_e2_star, sample_beta_svrw, sample_h0, sample_h_svrw, sample_sigma_eta2
from .var import demean_data, design_matrix


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
    - ELB is supported (shadow-rate data augmentation)
    - steady_state is supported
    - robust shocks are supported (student_t, mixture_outlier)
    """
    vol = model.volatility
    if vol is None or not vol.enabled:
        raise ValueError("volatility must be enabled")
    if vol.covariance != "factor":
        raise ValueError("volatility.covariance must be 'factor' for _fit_fsv")
    if vol.dynamics != "rw":
        raise ValueError("FSV currently supports only volatility.dynamics='rw'")

    robust = model.shocks is not None and model.shocks.family != "gaussian"

    prior_family = prior.family.lower()
    if prior_family != "niw":
        raise ValueError("FSV currently supports only prior.family='niw'")

    y_raw = np.asarray(dataset.values, dtype=float)
    if y_raw.ndim != 2:
        raise ValueError("dataset.values must be 2D")
    if y_raw.shape[0] <= model.p:
        raise ValueError("dataset is too short for requested lag order p")

    applies_to_idx: list[int] = []
    elb_t_idx: dict[int, np.ndarray] = {}

    y_lat = y_raw.copy()
    if model.elb is not None and model.elb.enabled:
        for name in model.elb.applies_to:
            try:
                applies_to_idx.append(dataset.variables.index(name))
            except ValueError as exc:
                raise ValueError(f"elb.applies_to contains unknown variable: {name}") from exc

        for j in applies_to_idx:
            mask = y_raw[:, j] <= (model.elb.bound + model.elb.tol)
            elb_t_idx[j] = np.where(mask)[0]
            y_lat[mask, j] = model.elb.bound - model.elb.init_offset

    ss = model.steady_state
    if ss is not None:
        n_lat = int(y_lat.shape[1])
        mu = np.asarray(ss.mu0, dtype=float).reshape(-1)
        if mu.shape != (n_lat,):
            raise ValueError("steady_state.mu0 must have shape (N,)")

        mu_gamma: np.ndarray | None = None
        if ss.ssvs is not None:
            mu_gamma = rng.uniform(size=n_lat) < float(ss.ssvs.inclusion_prob)

        y_dm = demean_data(y_lat, mu)
        x, y = design_matrix(y_dm, model.p, include_intercept=False)
        t_eff, n = y.shape

        k = int(vol.k_factors)
        if k > n:
            raise ValueError("model.volatility.k_factors must be <= N")

        niw = prior.niw
        m0_ssp, v0_ssp = _strip_intercept_niw_blocks(
            m0=niw.m0,
            v0=niw.v0,
            k_no_intercept=x.shape[1],
        )
        mn, _vn, _sn, _nun = posterior_niw(
            x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0
        )
        beta_lags = mn.copy()

        # Initialize volatility states (idiosyncratic and factor) on demeaned residuals.
        resid0 = y - x @ beta_lags
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

        prec = np.ones(t_eff, dtype=float) if robust else None

        beta_keep: list[np.ndarray] = []
        mu_keep: list[np.ndarray] = []
        mu_gamma_keep: list[np.ndarray] = []
        lam_keep: list[np.ndarray] = []
        h_eta_keep: list[np.ndarray] = []
        h0_eta_keep: list[np.ndarray] = []
        sigma_eta2_eta_keep: list[np.ndarray] = []
        h_f_keep: list[np.ndarray] = []
        h0_f_keep: list[np.ndarray] = []
        sigma_eta2_f_keep: list[np.ndarray] = []
        f_keep: list[np.ndarray] | None = [] if vol.store_factor_draws else None
        y_lat_keep: list[np.ndarray] | None = (
            [] if (model.elb is not None and model.elb.enabled) else None
        )

        for it in range(sampler.draws):
            y_dm = demean_data(y_lat, mu)
            x, y = design_matrix(y_dm, model.p, include_intercept=False)

            # Step A: sample beta_lags given (f, lam, h_eta[, prec])
            y_tilde = y - (f @ lam.T)
            h_eta_adj = h_eta
            if robust:
                if prec is None:
                    raise RuntimeError("robust shocks enabled but precision state is missing")
                h_eta_adj = h_eta - np.log(prec).reshape(-1, 1)

            beta_lags = sample_beta_svrw(
                x=x, y=y_tilde, m0=m0_ssp, v0=v0_ssp, h=h_eta_adj, rng=rng
            )

            # Step A2: sample mu given beta_lags, factor_mean, and diagonal idiosyncratic likelihood.
            factor_mean = f @ lam.T  # (T_eff, N)
            v_mu = ss.v0_mu
            if ss.ssvs is not None:
                if mu_gamma is None:
                    raise RuntimeError("mu_gamma state missing")
                v_mu = np.where(mu_gamma, float(ss.ssvs.slab_var), float(ss.ssvs.spike_var))

            mu = sample_steady_state_mu_fsv(
                y=y_lat,
                beta=beta_lags,
                h_eta=h_eta_adj,
                factor_mean=factor_mean,
                mu0=ss.mu0,
                v0_mu=v_mu,
                p=model.p,
                rng=rng,
            )

            if ss.ssvs is not None:
                mu_gamma = sample_mu_gamma(
                    mu=mu,
                    mu0=ss.mu0,
                    spike_var=float(ss.ssvs.spike_var),
                    slab_var=float(ss.ssvs.slab_var),
                    inclusion_prob=float(ss.ssvs.inclusion_prob),
                    rng=rng,
                )

            # Construct implied intercept and full beta matrix for ELB updates and downstream use.
            a_sum = _asum_from_beta(beta=beta_lags, n=n, p=model.p)
            c = (np.eye(n, dtype=float) - a_sum) @ mu
            beta_full = np.vstack([c.reshape(1, -1), beta_lags])

            # Optional ELB data augmentation step (shadow-rate updates).
            if model.elb is not None and model.elb.enabled:
                factor_mean_full = np.zeros_like(y_lat)
                factor_mean_full[model.p :, :] = factor_mean

                h_eta_full = np.vstack([np.tile(h0_eta.reshape(1, -1), (model.p, 1)), h_eta])
                if robust:
                    if prec is None:
                        raise RuntimeError("robust shocks enabled but precision state is missing")
                    prec_full = np.ones(y_lat.shape[0], dtype=float)
                    prec_full[model.p :] = prec
                    h_eta_full = h_eta_full - np.log(prec_full).reshape(-1, 1)
                for j in applies_to_idx:
                    for t in elb_t_idx[j]:
                        y_lat[t, j] = sample_shadow_value_fsv(
                            y=y_lat,
                            h_eta=h_eta_full,
                            factor_mean=factor_mean_full,
                            t=int(t),
                            j=int(j),
                            p=model.p,
                            beta=beta_full,
                            upper=model.elb.bound,
                            include_intercept=True,
                            rng=rng,
                        )

                y_dm = demean_data(y_lat, mu)
                x, y = design_matrix(y_dm, model.p, include_intercept=False)

            # Reduced-form residuals in demeaned representation
            e = y - x @ beta_lags

            # Step B: sample factors given (e, lam, h_eta, h_f[, prec])
            h_f_adj = h_f
            if robust:
                if prec is None:
                    raise RuntimeError("robust shocks enabled but precision state is missing")
                log_prec = np.log(prec).reshape(-1, 1)
                h_eta_adj = h_eta - log_prec
                h_f_adj = h_f - log_prec

            f = _sample_factors(e=e, lam=lam, h_eta=h_eta_adj, h_f=h_f_adj, rng=rng)

            # Step C: sample loadings given (e, f, h_eta)
            lam = _sample_loadings(
                e=e, f=f, h_eta=h_eta_adj, loading_prior_var=vol.loading_prior_var, rng=rng
            )
            lam, f = _normalize_factor_signs(lam=lam, f=f)

            # Step D: sample idiosyncratic vol states from eta = e - lam f
            eta = e - (f @ lam.T)
            if robust:
                if prec is None:
                    raise RuntimeError("robust shocks enabled but precision state is missing")
                eta = eta * np.sqrt(prec).reshape(-1, 1)
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
            f_sv = f
            if robust:
                if prec is None:
                    raise RuntimeError("robust shocks enabled but precision state is missing")
                f_sv = f * np.sqrt(prec).reshape(-1, 1)
            for j in range(k):
                y_star = log_e2_star(f_sv[:, j], epsilon=vol.epsilon)
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

            if robust:
                if model.shocks is None:
                    raise RuntimeError("robust shocks enabled but model.shocks is missing")
                prec = update_precision_scales_factor_sv(
                    errors=e, loadings=lam, h_eta=h_eta, h_f=h_f, spec=model.shocks, rng=rng
                )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta_full.copy())
                mu_keep.append(mu.copy())
                if mu_gamma is not None:
                    mu_gamma_keep.append(mu_gamma.copy())
                lam_keep.append(lam.copy())
                h_eta_keep.append(h_eta.copy())
                h0_eta_keep.append(h0_eta.copy())
                sigma_eta2_eta_keep.append(sigma_eta2_eta.copy())
                h_f_keep.append(h_f.copy())
                h0_f_keep.append(h0_f.copy())
                sigma_eta2_f_keep.append(sigma_eta2_f.copy())
                if f_keep is not None:
                    f_keep.append(f.copy())
                if y_lat_keep is not None:
                    y_lat_keep.append(y_lat.copy())

        latent_dataset = None
        if model.elb is not None and model.elb.enabled:
            latent_dataset = Dataset.from_arrays(
                values=y_lat, variables=dataset.variables, time_index=dataset.time_index
            )

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=None,
            latent_dataset=latent_dataset,
            latent_draws=np.stack(y_lat_keep) if y_lat_keep else None,
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
            mu_draws=np.stack(mu_keep) if mu_keep else None,
            mu_gamma_draws=np.stack(mu_gamma_keep) if mu_gamma_keep else None,
        )

    x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)
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

    prec = np.ones(t_eff, dtype=float) if robust else None

    beta_keep: list[np.ndarray] = []
    lam_keep: list[np.ndarray] = []
    h_eta_keep: list[np.ndarray] = []
    h0_eta_keep: list[np.ndarray] = []
    sigma_eta2_eta_keep: list[np.ndarray] = []
    h_f_keep: list[np.ndarray] = []
    h0_f_keep: list[np.ndarray] = []
    sigma_eta2_f_keep: list[np.ndarray] = []
    f_keep: list[np.ndarray] | None = [] if vol.store_factor_draws else None
    y_lat_keep: list[np.ndarray] | None = [] if (model.elb is not None and model.elb.enabled) else None

    for it in range(sampler.draws):
        # Step A: sample beta given (f, lam, h_eta[, prec])
        y_tilde = y - (f @ lam.T)
        h_eta_adj = h_eta
        if robust:
            if prec is None:
                raise RuntimeError("robust shocks enabled but precision state is missing")
            h_eta_adj = h_eta - np.log(prec).reshape(-1, 1)

        beta = sample_beta_svrw(x=x, y=y_tilde, m0=niw.m0, v0=niw.v0, h=h_eta_adj, rng=rng)

        # Optional ELB data augmentation step (shadow-rate updates).
        if model.elb is not None and model.elb.enabled:
            factor_mean = f @ lam.T  # (T_eff, N), aligned to times p..T-1
            factor_mean_full = np.zeros_like(y_lat)
            factor_mean_full[model.p :, :] = factor_mean

            h_eta_full = np.vstack([np.tile(h0_eta.reshape(1, -1), (model.p, 1)), h_eta])
            if robust:
                if prec is None:
                    raise RuntimeError("robust shocks enabled but precision state is missing")
                prec_full = np.ones(y_lat.shape[0], dtype=float)
                prec_full[model.p :] = prec
                h_eta_full = h_eta_full - np.log(prec_full).reshape(-1, 1)
            for j in applies_to_idx:
                for t in elb_t_idx[j]:
                    y_lat[t, j] = sample_shadow_value_fsv(
                        y=y_lat,
                        h_eta=h_eta_full,
                        factor_mean=factor_mean_full,
                        t=int(t),
                        j=int(j),
                        p=model.p,
                        beta=beta,
                        upper=model.elb.bound,
                        include_intercept=model.include_intercept,
                        rng=rng,
                    )

            x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        # Reduced-form residuals
        e = y - x @ beta

        # Step B: sample factors given (e, lam, h_eta, h_f[, prec])
        h_f_adj = h_f
        if robust:
            if prec is None:
                raise RuntimeError("robust shocks enabled but precision state is missing")
            log_prec = np.log(prec).reshape(-1, 1)
            h_eta_adj = h_eta - log_prec
            h_f_adj = h_f - log_prec

        f = _sample_factors(e=e, lam=lam, h_eta=h_eta_adj, h_f=h_f_adj, rng=rng)

        # Step C: sample loadings given (e, f, h_eta)
        lam = _sample_loadings(
            e=e, f=f, h_eta=h_eta_adj, loading_prior_var=vol.loading_prior_var, rng=rng
        )
        lam, f = _normalize_factor_signs(lam=lam, f=f)

        # Step D: sample idiosyncratic vol states from eta = e - lam f
        eta = e - (f @ lam.T)
        if robust:
            if prec is None:
                raise RuntimeError("robust shocks enabled but precision state is missing")
            eta = eta * np.sqrt(prec).reshape(-1, 1)
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
        f_sv = f
        if robust:
            if prec is None:
                raise RuntimeError("robust shocks enabled but precision state is missing")
            f_sv = f * np.sqrt(prec).reshape(-1, 1)
        for j in range(k):
            y_star = log_e2_star(f_sv[:, j], epsilon=vol.epsilon)
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

        if robust:
            if model.shocks is None:
                raise RuntimeError("robust shocks enabled but model.shocks is missing")
            prec = update_precision_scales_factor_sv(
                errors=e, loadings=lam, h_eta=h_eta, h_f=h_f, spec=model.shocks, rng=rng
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
            if y_lat_keep is not None:
                y_lat_keep.append(y_lat.copy())

    latent_dataset = None
    if model.elb is not None and model.elb.enabled:
        latent_dataset = Dataset.from_arrays(
            values=y_lat, variables=dataset.variables, time_index=dataset.time_index
        )

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        latent_dataset=latent_dataset,
        latent_draws=np.stack(y_lat_keep) if y_lat_keep else None,
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
