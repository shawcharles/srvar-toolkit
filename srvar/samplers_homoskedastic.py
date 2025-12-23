from __future__ import annotations

import numpy as np

from .bvar import posterior_niw, sample_posterior_niw
from .data.dataset import Dataset
from .elb import sample_shadow_value
from .results import FitResult, PosteriorNIW
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .ssvs import sample_gamma_rows, v0_diag_from_gamma
from .var import demean_data, design_matrix

from .samplers_blasso import _blasso_update_adaptive, _blasso_update_global, _blasso_v0_from_state
from .samplers_dl import _dl_sample_beta_sigma, _dl_update
from .samplers_ssp import _asum_from_beta, _strip_intercept_niw_blocks, sample_mu_gamma, sample_steady_state_mu

def _fit_no_elb(
    *,
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    prior_family: str,
    rng: np.random.Generator,
) -> FitResult:
    ss = model.steady_state
    if ss is not None:
        y_lat = np.asarray(dataset.values, dtype=float)
        n = int(y_lat.shape[1])
        mu = np.asarray(ss.mu0, dtype=float).reshape(-1)
        if mu.shape != (n,):
            raise ValueError("steady_state.mu0 must have shape (N,)")

        mu_gamma: np.ndarray | None = None
        if ss.ssvs is not None:
            mu_gamma = rng.uniform(size=n) < float(ss.ssvs.inclusion_prob)

        niw = prior.niw

        gamma: np.ndarray | None = None
        fixed_mask: np.ndarray | None = None

        tau: np.ndarray | None = None
        lambda_: float | None = None
        lambda_c: float | None = None
        lambda_L: float | None = None
        c_mask: np.ndarray | None = None

        dl_psi: np.ndarray | None = None
        dl_vartheta: np.ndarray | None = None
        dl_zeta: float | None = None
        dl_inv_v0: np.ndarray | None = None

        beta_keep: list[np.ndarray] = []
        sigma_keep: list[np.ndarray] = []
        gamma_keep: list[np.ndarray] = []
        mu_keep: list[np.ndarray] = []
        mu_gamma_keep: list[np.ndarray] = []
        last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            y_dm = demean_data(y_lat, mu)
            x, y = design_matrix(y_dm, model.p, include_intercept=False)
            _t_eff, k = x.shape

            m0_ssp, v0_ssp = _strip_intercept_niw_blocks(m0=niw.m0, v0=niw.v0, k_no_intercept=k)

            v0_used = v0_ssp
            if prior_family == "niw":
                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

            elif prior_family == "ssvs":
                if prior.ssvs is None:
                    raise ValueError("prior.family='ssvs' requires prior.ssvs")
                spec = prior.ssvs

                if gamma is None:
                    gamma = rng.uniform(size=k) < float(spec.inclusion_prob)
                    fixed_mask = np.zeros(k, dtype=bool)

                v0_diag = v0_diag_from_gamma(
                    gamma=gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    intercept_slab_var=None,
                )
                v0_used = np.diag(v0_diag)

                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_used, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

                gamma = sample_gamma_rows(
                    beta=beta_lags,
                    sigma=sigma,
                    gamma=gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    inclusion_prob=spec.inclusion_prob,
                    fixed_mask=fixed_mask,
                    rng=rng,
                )

            elif prior_family == "blasso":
                if prior.blasso is None:
                    raise ValueError("prior.family='blasso' requires prior.blasso")
                spec_b = prior.blasso

                if tau is None:
                    tau = np.full(k, float(spec_b.tau_init), dtype=float)
                    lambda_ = float(spec_b.lambda_init)
                    lambda_c = float(spec_b.lambda_init)
                    lambda_L = float(spec_b.lambda_init)
                    c_mask = np.zeros(k, dtype=bool)

                if tau is None:
                    raise RuntimeError("blasso state missing")
                v0_used = _blasso_v0_from_state(tau=tau)

                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_used, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

                if spec_b.mode == "global":
                    if lambda_ is None:
                        raise RuntimeError("lambda missing")
                    tau, lambda_ = _blasso_update_global(
                        beta=beta_lags,
                        tau=tau,
                        lambda_=lambda_,
                        a0=float(spec_b.a0_global),
                        b0=float(spec_b.b0_global),
                        rng=rng,
                    )
                else:
                    if lambda_c is None or lambda_L is None or c_mask is None:
                        raise RuntimeError("blasso adaptive state missing")
                    tau, lambda_c, lambda_L = _blasso_update_adaptive(
                        beta=beta_lags,
                        tau=tau,
                        lambda_c=lambda_c,
                        lambda_L=lambda_L,
                        a0_c=float(spec_b.a0_c),
                        b0_c=float(spec_b.b0_c),
                        a0_L=float(spec_b.a0_L),
                        b0_L=float(spec_b.b0_L),
                        c_mask=c_mask,
                        rng=rng,
                    )

            elif prior_family == "dl":
                if prior.dl is None:
                    raise ValueError("prior.family='dl' requires prior.dl")
                spec_d = prior.dl

                if dl_psi is None:
                    km = int(k * y.shape[1])
                    dl_psi = np.full(km, float(spec_d.dl_scaler), dtype=float)
                    dl_vartheta = np.full(km, float(spec_d.dl_scaler), dtype=float)
                    dl_zeta = float(spec_d.dl_scaler)
                    dl_inv_v0 = 1.0 / (dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6)

                if dl_inv_v0 is None:
                    raise RuntimeError("dl state missing")

                beta_lags, sigma = _dl_sample_beta_sigma(
                    x=x,
                    y=y,
                    m0=m0_ssp,
                    inv_v0_vec=dl_inv_v0,
                    s0=niw.s0,
                    nu0=niw.nu0,
                    rng=rng,
                )
                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

                if dl_psi is None or dl_vartheta is None or dl_zeta is None:
                    raise RuntimeError("dl state missing")
                dl_psi, dl_vartheta, dl_zeta, dl_inv_v0 = _dl_update(
                    beta=beta_lags,
                    psi=dl_psi,
                    vartheta=dl_vartheta,
                    zeta=dl_zeta,
                    abeta=float(spec_d.abeta),
                    rng=rng,
                )
            else:
                raise ValueError(f"Unknown prior family: {prior_family}")

            v_mu = ss.v0_mu
            if ss.ssvs is not None:
                if mu_gamma is None:
                    raise RuntimeError("mu_gamma state missing")
                v_mu = np.where(mu_gamma, float(ss.ssvs.slab_var), float(ss.ssvs.spike_var))

            mu = sample_steady_state_mu(
                y=y_lat,
                beta=beta_lags,
                sigma=sigma,
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

            a_sum = _asum_from_beta(beta=beta_lags, n=n, p=model.p)
            c = (np.eye(n, dtype=float) - a_sum) @ mu
            beta_full = beta_lags
            if model.include_intercept:
                beta_full = np.vstack([c.reshape(1, -1), beta_lags])

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta_full.copy())
                sigma_keep.append(sigma.copy())
                mu_keep.append(mu.copy())
                if mu_gamma is not None:
                    mu_gamma_keep.append(mu_gamma.copy())
                if gamma is not None:
                    g = gamma
                    if model.include_intercept:
                        g = np.concatenate([np.array([True], dtype=bool), gamma])
                    gamma_keep.append(g.copy())

        if last_posterior is None:
            raise RuntimeError("sampler.draws produced no posterior")

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=last_posterior,
            beta_draws=np.stack(beta_keep) if beta_keep else None,
            sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
            gamma_draws=np.stack(gamma_keep) if gamma_keep else None,
            mu_draws=np.stack(mu_keep) if mu_keep else None,
            mu_gamma_draws=np.stack(mu_gamma_keep) if mu_gamma_keep else None,
        )

    x, y = design_matrix(dataset.values, model.p, include_intercept=model.include_intercept)

    niw = prior.niw
    if prior_family == "niw":
        mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)

        posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

        beta_all, sigma_all = sample_posterior_niw(
            mn=mn,
            vn=vn,
            sn=sn,
            nun=nun,
            draws=sampler.draws,
            rng=rng,
        )
        keep_idx = np.arange(sampler.burn_in, sampler.draws, sampler.thin, dtype=int)
        beta_keep = beta_all[keep_idx] if keep_idx.size > 0 else None
        sigma_keep = sigma_all[keep_idx] if keep_idx.size > 0 else None

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=posterior,
            beta_draws=beta_keep,
            sigma_draws=sigma_keep,
        )

    if prior_family == "blasso":
        if prior.blasso is None:
            raise ValueError("prior.family='blasso' requires prior.blasso")

        spec_b = prior.blasso
        _t_eff, k = x.shape
        _n = y.shape[1]
        if niw.m0.shape != (k, _n):
            raise ValueError("blasso requires prior.niw.m0 with shape (K, N)")
        if niw.s0.shape != (_n, _n):
            raise ValueError("blasso requires prior.niw.s0 with shape (N, N)")

        tau = np.full(k, float(spec_b.tau_init), dtype=float)
        lambda_ = float(spec_b.lambda_init)
        lambda_c = float(spec_b.lambda_init)
        lambda_L = float(spec_b.lambda_init)

        c_mask = np.zeros(k, dtype=bool)
        if model.include_intercept:
            c_mask[0] = True

        beta_keep: list[np.ndarray] = []
        sigma_keep: list[np.ndarray] = []
        last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            v0 = _blasso_v0_from_state(tau=tau)
            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
            beta = beta_draws[0]
            sigma = sigma_draws[0]

            if spec_b.mode == "global":
                tau, lambda_ = _blasso_update_global(
                    beta=beta,
                    tau=tau,
                    lambda_=lambda_,
                    a0=float(spec_b.a0_global),
                    b0=float(spec_b.b0_global),
                    rng=rng,
                )
            else:
                tau, lambda_c, lambda_L = _blasso_update_adaptive(
                    beta=beta,
                    tau=tau,
                    lambda_c=lambda_c,
                    lambda_L=lambda_L,
                    a0_c=float(spec_b.a0_c),
                    b0_c=float(spec_b.b0_c),
                    a0_L=float(spec_b.a0_L),
                    b0_L=float(spec_b.b0_L),
                    c_mask=c_mask,
                    rng=rng,
                )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta.copy())
                sigma_keep.append(sigma.copy())

        if last_posterior is None:
            raise RuntimeError("sampler.draws produced no posterior")

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=last_posterior,
            beta_draws=np.stack(beta_keep) if beta_keep else None,
            sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
        )

    if prior_family == "dl":
        if prior.dl is None:
            raise ValueError("prior.family='dl' requires prior.dl")

        spec_d = prior.dl
        _t_eff, k = x.shape
        _n = y.shape[1]
        if niw.m0.shape != (k, _n):
            raise ValueError("dl requires prior.niw.m0 with shape (K, N)")
        if niw.s0.shape != (_n, _n):
            raise ValueError("dl requires prior.niw.s0 with shape (N, N)")

        km = int(k * _n)
        psi = np.full(km, float(spec_d.dl_scaler), dtype=float)
        vartheta = np.full(km, float(spec_d.dl_scaler), dtype=float)
        zeta = float(spec_d.dl_scaler)
        inv_v0 = 1.0 / (psi * (vartheta * vartheta) * (zeta * zeta) + 1e-6)

        beta_keep: list[np.ndarray] = []
        sigma_keep: list[np.ndarray] = []
        last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            beta, sigma = _dl_sample_beta_sigma(
                x=x,
                y=y,
                m0=niw.m0,
                inv_v0_vec=inv_v0,
                s0=niw.s0,
                nu0=niw.nu0,
                rng=rng,
            )
            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            psi, vartheta, zeta, inv_v0 = _dl_update(
                beta=beta,
                psi=psi,
                vartheta=vartheta,
                zeta=zeta,
                abeta=float(spec_d.abeta),
                rng=rng,
            )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta.copy())
                sigma_keep.append(sigma.copy())

        if last_posterior is None:
            raise RuntimeError("sampler.draws produced no posterior")

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=last_posterior,
            beta_draws=np.stack(beta_keep) if beta_keep else None,
            sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
        )

    if prior.ssvs is None:
        raise ValueError("prior.family='ssvs' requires prior.ssvs")

    spec = prior.ssvs
    _t_eff, k = x.shape
    _n = y.shape[1]
    if niw.m0.shape != (k, _n):
        raise ValueError("ssvs requires prior.niw.m0 with shape (K, N)")
    if niw.s0.shape != (_n, _n):
        raise ValueError("ssvs requires prior.niw.s0 with shape (N, N)")

    gamma = rng.uniform(size=k) < spec.inclusion_prob
    fixed_mask = np.zeros(k, dtype=bool)
    if model.include_intercept and spec.fix_intercept:
        fixed_mask[0] = True
        gamma[0] = True

    beta_keep: list[np.ndarray] = []
    sigma_keep: list[np.ndarray] = []
    gamma_keep: list[np.ndarray] = []
    last_posterior: PosteriorNIW | None = None

    for it in range(sampler.draws):
        v0_diag = v0_diag_from_gamma(
            gamma=gamma,
            spike_var=spec.spike_var,
            slab_var=spec.slab_var,
            intercept_slab_var=spec.intercept_slab_var,
        )
        v0 = np.diag(v0_diag)

        mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
        last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

        beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
        beta = beta_draws[0]
        sigma = sigma_draws[0]

        gamma = sample_gamma_rows(
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            spike_var=spec.spike_var,
            slab_var=spec.slab_var,
            inclusion_prob=spec.inclusion_prob,
            fixed_mask=fixed_mask,
            rng=rng,
        )

        if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
            beta_keep.append(beta.copy())
            sigma_keep.append(sigma.copy())
            gamma_keep.append(gamma.copy())

    if last_posterior is None:
        raise RuntimeError("sampler.draws produced no posterior")

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=last_posterior,
        beta_draws=np.stack(beta_keep) if beta_keep else None,
        sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
        gamma_draws=np.stack(gamma_keep) if gamma_keep else None,
    )


def _fit_elb_gibbs(
    *,
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    prior_family: str,
    rng: np.random.Generator,
) -> FitResult:
    elb = model.elb
    if elb is None or not elb.enabled:
        raise ValueError("elb must be enabled")

    applies_to_idx: list[int] = []
    for name in elb.applies_to:
        try:
            applies_to_idx.append(dataset.variables.index(name))
        except ValueError as e:
            raise ValueError(f"elb.applies_to contains unknown variable: {name}") from e

    y_lat = dataset.values.copy()

    elb_t_idx: dict[int, np.ndarray] = {}
    for j in applies_to_idx:
        mask = dataset.values[:, j] <= (elb.bound + elb.tol)
        elb_t_idx[j] = np.where(mask)[0]
        y_lat[mask, j] = elb.bound - elb.init_offset

    ss = model.steady_state
    if ss is not None:
        n = int(y_lat.shape[1])
        mu = np.asarray(ss.mu0, dtype=float).reshape(-1)
        if mu.shape != (n,):
            raise ValueError("steady_state.mu0 must have shape (N,)")

        mu_gamma: np.ndarray | None = None
        if ss.ssvs is not None:
            mu_gamma = rng.uniform(size=n) < float(ss.ssvs.inclusion_prob)

        beta_keep: list[np.ndarray] = []
        sigma_keep: list[np.ndarray] = []
        y_lat_keep: list[np.ndarray] = []
        gamma_keep: list[np.ndarray] = []
        mu_keep: list[np.ndarray] = []
        mu_gamma_keep: list[np.ndarray] = []

        niw = prior.niw

        gamma: np.ndarray | None = None
        fixed_mask: np.ndarray | None = None

        tau: np.ndarray | None = None
        lambda_: float | None = None
        lambda_c: float | None = None
        lambda_L: float | None = None
        c_mask: np.ndarray | None = None

        dl_psi: np.ndarray | None = None
        dl_vartheta: np.ndarray | None = None
        dl_zeta: float | None = None
        dl_inv_v0: np.ndarray | None = None

        last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            y_dm = demean_data(y_lat, mu)
            x, y = design_matrix(y_dm, model.p, include_intercept=False)
            _t_eff, k = x.shape

            m0_ssp, v0_ssp = _strip_intercept_niw_blocks(m0=niw.m0, v0=niw.v0, k_no_intercept=k)

            if prior_family == "niw":
                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

            elif prior_family == "ssvs":
                if prior.ssvs is None:
                    raise ValueError("prior.family='ssvs' requires prior.ssvs")
                spec = prior.ssvs

                if gamma is None:
                    gamma = rng.uniform(size=k) < float(spec.inclusion_prob)
                    fixed_mask = np.zeros(k, dtype=bool)

                v0_diag = v0_diag_from_gamma(
                    gamma=gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    intercept_slab_var=None,
                )
                v0 = np.diag(v0_diag)

                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

                gamma = sample_gamma_rows(
                    beta=beta_lags,
                    sigma=sigma,
                    gamma=gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    inclusion_prob=spec.inclusion_prob,
                    fixed_mask=fixed_mask,
                    rng=rng,
                )

            elif prior_family == "blasso":
                if prior.blasso is None:
                    raise ValueError("prior.family='blasso' requires prior.blasso")
                spec_b = prior.blasso

                if tau is None:
                    tau = np.full(k, float(spec_b.tau_init), dtype=float)
                    lambda_ = float(spec_b.lambda_init)
                    lambda_c = float(spec_b.lambda_init)
                    lambda_L = float(spec_b.lambda_init)
                    c_mask = np.zeros(k, dtype=bool)

                if tau is None:
                    raise RuntimeError("blasso state missing")
                v0 = _blasso_v0_from_state(tau=tau)

                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

                if spec_b.mode == "global":
                    if lambda_ is None:
                        raise RuntimeError("lambda missing")
                    tau, lambda_ = _blasso_update_global(
                        beta=beta_lags,
                        tau=tau,
                        lambda_=lambda_,
                        a0=float(spec_b.a0_global),
                        b0=float(spec_b.b0_global),
                        rng=rng,
                    )
                else:
                    if lambda_c is None or lambda_L is None or c_mask is None:
                        raise RuntimeError("blasso adaptive state missing")
                    tau, lambda_c, lambda_L = _blasso_update_adaptive(
                        beta=beta_lags,
                        tau=tau,
                        lambda_c=lambda_c,
                        lambda_L=lambda_L,
                        a0_c=float(spec_b.a0_c),
                        b0_c=float(spec_b.b0_c),
                        a0_L=float(spec_b.a0_L),
                        b0_L=float(spec_b.b0_L),
                        c_mask=c_mask,
                        rng=rng,
                    )

            elif prior_family == "dl":
                if prior.dl is None:
                    raise ValueError("prior.family='dl' requires prior.dl")
                spec_d = prior.dl

                if dl_psi is None:
                    km = int(k * y.shape[1])
                    dl_psi = np.full(km, float(spec_d.dl_scaler), dtype=float)
                    dl_vartheta = np.full(km, float(spec_d.dl_scaler), dtype=float)
                    dl_zeta = float(spec_d.dl_scaler)
                    dl_inv_v0 = 1.0 / (dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6)

                if dl_inv_v0 is None:
                    raise RuntimeError("dl state missing")

                beta_lags, sigma = _dl_sample_beta_sigma(
                    x=x,
                    y=y,
                    m0=m0_ssp,
                    inv_v0_vec=dl_inv_v0,
                    s0=niw.s0,
                    nu0=niw.nu0,
                    rng=rng,
                )

                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

                if dl_psi is None or dl_vartheta is None or dl_zeta is None:
                    raise RuntimeError("dl state missing")
                dl_psi, dl_vartheta, dl_zeta, dl_inv_v0 = _dl_update(
                    beta=beta_lags,
                    psi=dl_psi,
                    vartheta=dl_vartheta,
                    zeta=dl_zeta,
                    abeta=float(spec_d.abeta),
                    rng=rng,
                )
            else:
                raise ValueError(f"Unknown prior family: {prior_family}")

            v_mu = ss.v0_mu
            if ss.ssvs is not None:
                if mu_gamma is None:
                    raise RuntimeError("mu_gamma state missing")
                v_mu = np.where(mu_gamma, float(ss.ssvs.slab_var), float(ss.ssvs.spike_var))

            mu = sample_steady_state_mu(
                y=y_lat,
                beta=beta_lags,
                sigma=sigma,
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

            a_sum = _asum_from_beta(beta=beta_lags, n=n, p=model.p)
            c = (np.eye(n, dtype=float) - a_sum) @ mu
            beta_full = beta_lags
            if model.include_intercept:
                beta_full = np.vstack([c.reshape(1, -1), beta_lags])

            for j in applies_to_idx:
                for t in elb_t_idx[j]:
                    y_lat[t, j] = sample_shadow_value(
                        y=y_lat,
                        t=int(t),
                        j=int(j),
                        p=model.p,
                        beta=beta_full,
                        sigma=sigma,
                        upper=elb.bound,
                        include_intercept=model.include_intercept,
                        rng=rng,
                    )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta_full.copy())
                sigma_keep.append(sigma.copy())
                y_lat_keep.append(y_lat.copy())
                mu_keep.append(mu.copy())
                if mu_gamma is not None:
                    mu_gamma_keep.append(mu_gamma.copy())
                if gamma is not None:
                    g = gamma
                    if model.include_intercept:
                        g = np.concatenate([np.array([True], dtype=bool), gamma])
                    gamma_keep.append(g.copy())

        if last_posterior is None:
            raise RuntimeError("sampler.draws produced no posterior")

        latent_dataset = Dataset.from_arrays(
            values=y_lat,
            variables=dataset.variables,
            time_index=dataset.time_index,
        )

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=last_posterior,
            latent_dataset=latent_dataset,
            latent_draws=np.stack(y_lat_keep) if y_lat_keep else None,
            beta_draws=np.stack(beta_keep) if beta_keep else None,
            sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
            gamma_draws=np.stack(gamma_keep) if gamma_keep else None,
            mu_draws=np.stack(mu_keep) if mu_keep else None,
            mu_gamma_draws=np.stack(mu_gamma_keep) if mu_gamma_keep else None,
        )

    beta_keep: list[np.ndarray] = []
    sigma_keep: list[np.ndarray] = []
    y_lat_keep: list[np.ndarray] = []
    gamma_keep: list[np.ndarray] = []

    niw = prior.niw

    gamma: np.ndarray | None = None
    fixed_mask: np.ndarray | None = None
    spec = prior.ssvs if prior_family == "ssvs" else None
    blasso = prior.blasso if prior_family == "blasso" else None
    dl = prior.dl if prior_family == "dl" else None

    tau: np.ndarray | None = None
    lambda_: float | None = None
    lambda_c: float | None = None
    lambda_L: float | None = None
    c_mask: np.ndarray | None = None

    dl_psi: np.ndarray | None = None
    dl_vartheta: np.ndarray | None = None
    dl_zeta: float | None = None
    dl_inv_v0: np.ndarray | None = None

    last_posterior: PosteriorNIW | None = None

    for it in range(sampler.draws):
        x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        if prior_family == "niw":
            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
            beta = beta_draws[0]
            sigma = sigma_draws[0]
        elif prior_family == "ssvs":
            if spec is None:
                raise ValueError("prior.family='ssvs' requires prior.ssvs")

            t_eff, k = x.shape
            _n = y.shape[1]
            if niw.m0.shape != (k, _n):
                raise ValueError("ssvs requires prior.niw.m0 with shape (K, N)")
            if niw.s0.shape != (_n, _n):
                raise ValueError("ssvs requires prior.niw.s0 with shape (N, N)")

            if gamma is None:
                gamma = rng.uniform(size=k) < spec.inclusion_prob
                fixed_mask = np.zeros(k, dtype=bool)
                if model.include_intercept and spec.fix_intercept:
                    fixed_mask[0] = True
                    gamma[0] = True

            v0_diag = v0_diag_from_gamma(
                gamma=gamma,
                spike_var=spec.spike_var,
                slab_var=spec.slab_var,
                intercept_slab_var=spec.intercept_slab_var,
            )
            v0 = np.diag(v0_diag)

            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
            beta = beta_draws[0]
            sigma = sigma_draws[0]

            gamma = sample_gamma_rows(
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                spike_var=spec.spike_var,
                slab_var=spec.slab_var,
                inclusion_prob=spec.inclusion_prob,
                fixed_mask=fixed_mask,
                rng=rng,
            )

        elif prior_family == "dl":
            if dl is None:
                raise ValueError("prior.family='dl' requires prior.dl")

            t_eff, k = x.shape
            _n = y.shape[1]
            if niw.m0.shape != (k, _n):
                raise ValueError("dl requires prior.niw.m0 with shape (K, N)")
            if niw.s0.shape != (_n, _n):
                raise ValueError("dl requires prior.niw.s0 with shape (N, N)")

            if dl_psi is None:
                km = int(k * _n)
                dl_psi = np.full(km, float(dl.dl_scaler), dtype=float)
                dl_vartheta = np.full(km, float(dl.dl_scaler), dtype=float)
                dl_zeta = float(dl.dl_scaler)
                dl_inv_v0 = 1.0 / (dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6)

            if dl_inv_v0 is None:
                raise RuntimeError("dl state missing")

            beta, sigma = _dl_sample_beta_sigma(
                x=x,
                y=y,
                m0=niw.m0,
                inv_v0_vec=dl_inv_v0,
                s0=niw.s0,
                nu0=niw.nu0,
                rng=rng,
            )

            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            if dl_psi is None or dl_vartheta is None or dl_zeta is None:
                raise RuntimeError("dl state missing")
            dl_psi, dl_vartheta, dl_zeta, dl_inv_v0 = _dl_update(
                beta=beta,
                psi=dl_psi,
                vartheta=dl_vartheta,
                zeta=dl_zeta,
                abeta=float(dl.abeta),
                rng=rng,
            )

        else:
            if blasso is None:
                raise ValueError("prior.family='blasso' requires prior.blasso")

            t_eff, k = x.shape
            _n = y.shape[1]
            if niw.m0.shape != (k, _n):
                raise ValueError("blasso requires prior.niw.m0 with shape (K, N)")
            if niw.s0.shape != (_n, _n):
                raise ValueError("blasso requires prior.niw.s0 with shape (N, N)")

            if tau is None:
                tau = np.full(k, float(blasso.tau_init), dtype=float)
                lambda_ = float(blasso.lambda_init)
                lambda_c = float(blasso.lambda_init)
                lambda_L = float(blasso.lambda_init)
                c_mask = np.zeros(k, dtype=bool)
                if model.include_intercept:
                    c_mask[0] = True

            v0 = _blasso_v0_from_state(tau=tau)
            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
            beta = beta_draws[0]
            sigma = sigma_draws[0]

            if blasso.mode == "global":
                if lambda_ is None or tau is None:
                    raise RuntimeError("blasso global state missing")
                tau, lambda_ = _blasso_update_global(
                    beta=beta,
                    tau=tau,
                    lambda_=lambda_,
                    a0=float(blasso.a0_global),
                    b0=float(blasso.b0_global),
                    rng=rng,
                )
            else:
                if lambda_c is None or lambda_L is None or tau is None or c_mask is None:
                    raise RuntimeError("blasso adaptive state missing")
                tau, lambda_c, lambda_L = _blasso_update_adaptive(
                    beta=beta,
                    tau=tau,
                    lambda_c=lambda_c,
                    lambda_L=lambda_L,
                    a0_c=float(blasso.a0_c),
                    b0_c=float(blasso.b0_c),
                    a0_L=float(blasso.a0_L),
                    b0_L=float(blasso.b0_L),
                    c_mask=c_mask,
                    rng=rng,
                )

        for j in applies_to_idx:
            for t in elb_t_idx[j]:
                y_lat[t, j] = sample_shadow_value(
                    y=y_lat,
                    t=int(t),
                    j=int(j),
                    p=model.p,
                    beta=beta,
                    sigma=sigma,
                    upper=elb.bound,
                    include_intercept=model.include_intercept,
                    rng=rng,
                )

        if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
            beta_keep.append(beta.copy())
            sigma_keep.append(sigma.copy())
            y_lat_keep.append(y_lat.copy())
            if gamma is not None:
                gamma_keep.append(gamma.copy())

    if last_posterior is None:
        raise RuntimeError("sampler.draws produced no posterior")

    latent_dataset = Dataset.from_arrays(
        values=y_lat,
        variables=dataset.variables,
        time_index=dataset.time_index,
    )

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=last_posterior,
        latent_dataset=latent_dataset,
        latent_draws=np.stack(y_lat_keep) if y_lat_keep else None,
        beta_draws=np.stack(beta_keep) if beta_keep else None,
        sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
        gamma_draws=np.stack(gamma_keep) if gamma_keep else None,
    )
