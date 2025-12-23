from __future__ import annotations

import numpy as np

from .bvar import posterior_niw
from .data.dataset import Dataset
from .elb import sample_shadow_value_svrw
from .linalg import symmetrize
from .results import FitResult
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .sv import log_e2_star, sample_beta_svrw, sample_h0, sample_h_svrw, sample_sigma_eta2
from .var import demean_data, design_matrix

from .samplers_blasso import _blasso_update_adaptive, _blasso_update_global, _blasso_v0_from_state
from .samplers_dl import _dl_sample_beta_svrw, _dl_update
from .samplers_ssp import _asum_from_beta, _strip_intercept_niw_blocks, sample_mu_gamma, sample_steady_state_mu_svrw

def _fit_svrw(
    *,
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    rng: np.random.Generator,
) -> FitResult:
    vol = model.volatility
    if vol is None or not vol.enabled:
        raise ValueError("volatility must be enabled")

    ss = model.steady_state
    if ss is not None:
        applies_to_idx: list[int] = []
        elb_t_idx: dict[int, np.ndarray] = {}

        y_lat = dataset.values.copy()
        if model.elb is not None and model.elb.enabled:
            for name in model.elb.applies_to:
                try:
                    applies_to_idx.append(dataset.variables.index(name))
                except ValueError as e:
                    raise ValueError(f"elb.applies_to contains unknown variable: {name}") from e

            for j in applies_to_idx:
                mask = dataset.values[:, j] <= (model.elb.bound + model.elb.tol)
                elb_t_idx[j] = np.where(mask)[0]
                y_lat[mask, j] = model.elb.bound - model.elb.init_offset

        y_lat = np.asarray(y_lat, dtype=float)
        n = int(y_lat.shape[1])
        mu = np.asarray(ss.mu0, dtype=float).reshape(-1)
        if mu.shape != (n,):
            raise ValueError("steady_state.mu0 must have shape (N,)")

        mu_gamma: np.ndarray | None = None
        if ss.ssvs is not None:
            mu_gamma = rng.uniform(size=n) < float(ss.ssvs.inclusion_prob)

        y_dm = demean_data(y_lat, mu)
        x, y = design_matrix(y_dm, model.p, include_intercept=False)
        t_eff, _n = y.shape

        niw = prior.niw
        prior_family = prior.family.lower()

        blasso = prior.blasso if prior_family == "blasso" else None
        if prior_family == "blasso" and blasso is None:
            raise ValueError("prior.family='blasso' requires prior.blasso")
        dl = prior.dl if prior_family == "dl" else None
        if prior_family == "dl" and dl is None:
            raise ValueError("prior.family='dl' requires prior.dl")
        if prior_family == "ssvs":
            raise ValueError("prior.family='ssvs' is not supported with volatility")

        tau: np.ndarray | None = None
        lambda_: float | None = None
        lambda_c: float | None = None
        lambda_L: float | None = None
        c_mask: np.ndarray | None = None

        dl_psi: np.ndarray | None = None
        dl_vartheta: np.ndarray | None = None
        dl_zeta: float | None = None
        dl_inv_v0: np.ndarray | None = None

        m0_ssp, v0_ssp = _strip_intercept_niw_blocks(m0=niw.m0, v0=niw.v0, k_no_intercept=x.shape[1])
        mn, _vn, _sn, _nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0)
        beta_lags = mn.copy()

        if prior_family == "blasso":
            if blasso is None:
                raise RuntimeError("blasso spec missing")
            tau = np.full(x.shape[1], float(blasso.tau_init), dtype=float)
            if blasso.mode == "global":
                lambda_ = float(blasso.lambda_init)
            else:
                lambda_c = float(blasso.lambda_init)
                lambda_L = float(blasso.lambda_init)
                c_mask = np.zeros(x.shape[1], dtype=bool)

        if prior_family == "dl":
            if dl is None:
                raise RuntimeError("dl spec missing")
            km = int(x.shape[1] * y.shape[1])
            dl_psi = np.full(km, float(dl.dl_scaler), dtype=float)
            dl_vartheta = np.full(km, float(dl.dl_scaler), dtype=float)
            dl_zeta = float(dl.dl_scaler)
            dl_inv_v0 = 1.0 / (dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6)

        h0 = np.log(np.var(y, axis=0) + 1e-12)
        h = np.tile(h0.reshape(1, -1), (t_eff, 1))
        sigma_eta2 = 0.05 * np.ones(n, dtype=float)

        beta_keep: list[np.ndarray] = []
        h_keep: list[np.ndarray] = []
        h0_keep: list[np.ndarray] = []
        sigma_eta2_keep: list[np.ndarray] = []
        y_lat_keep: list[np.ndarray] | None = [] if (model.elb is not None and model.elb.enabled) else None
        mu_keep: list[np.ndarray] = []
        mu_gamma_keep: list[np.ndarray] = []

        for it in range(sampler.draws):
            y_dm = demean_data(y_lat, mu)
            x, y = design_matrix(y_dm, model.p, include_intercept=False)
            m0_ssp, v0_ssp = _strip_intercept_niw_blocks(m0=niw.m0, v0=niw.v0, k_no_intercept=x.shape[1])

            if prior_family == "dl":
                if dl_inv_v0 is None:
                    raise RuntimeError("dl state missing")
                beta_lags = _dl_sample_beta_svrw(x=x, y=y, m0=m0_ssp, inv_v0_vec=dl_inv_v0, h=h, rng=rng)
            else:
                if prior_family == "blasso":
                    if tau is None:
                        raise RuntimeError("blasso state missing")
                    v0 = _blasso_v0_from_state(tau=tau)
                else:
                    v0 = v0_ssp
                beta_lags = sample_beta_svrw(x=x, y=y, m0=m0_ssp, v0=v0, h=h, rng=rng)

            v_mu = ss.v0_mu
            if ss.ssvs is not None:
                if mu_gamma is None:
                    raise RuntimeError("mu_gamma state missing")
                v_mu = np.where(mu_gamma, float(ss.ssvs.slab_var), float(ss.ssvs.spike_var))

            mu = sample_steady_state_mu_svrw(
                y=y_lat,
                beta=beta_lags,
                h=h,
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

            if model.elb is not None and model.elb.enabled:
                h_full = np.vstack([np.tile(h0.reshape(1, -1), (model.p, 1)), h])
                for j in applies_to_idx:
                    for t in elb_t_idx[j]:
                        y_lat[t, j] = sample_shadow_value_svrw(
                            y=y_lat,
                            h=h_full,
                            t=int(t),
                            j=int(j),
                            p=model.p,
                            beta=beta_full,
                            upper=model.elb.bound,
                            include_intercept=model.include_intercept,
                            rng=rng,
                        )

                y_dm = demean_data(y_lat, mu)
                x, y = design_matrix(y_dm, model.p, include_intercept=False)

            e = y - x @ beta_lags

            for i in range(n):
                y_star = log_e2_star(e[:, i], epsilon=vol.epsilon)
                h[:, i] = sample_h_svrw(
                    y_star=y_star,
                    h=h[:, i],
                    sigma_eta2=float(sigma_eta2[i]),
                    h0=float(h0[i]),
                    rng=rng,
                )
                h0[i] = sample_h0(
                    h1=float(h[0, i]),
                    sigma_eta2=float(sigma_eta2[i]),
                    prior_mean=vol.h0_prior_mean,
                    prior_var=vol.h0_prior_var,
                    rng=rng,
                )
                sigma_eta2[i] = sample_sigma_eta2(
                    h=h[:, i],
                    h0=float(h0[i]),
                    nu0=vol.sigma_eta_prior_nu0,
                    s0=vol.sigma_eta_prior_s0,
                    rng=rng,
                )

            if prior_family == "blasso":
                if blasso is None or tau is None:
                    raise RuntimeError("blasso state missing")
                if blasso.mode == "global":
                    if lambda_ is None:
                        raise RuntimeError("lambda missing")
                    tau, lambda_ = _blasso_update_global(
                        beta=beta_lags,
                        tau=tau,
                        lambda_=lambda_,
                        a0=float(blasso.a0_global),
                        b0=float(blasso.b0_global),
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
                        a0_c=float(blasso.a0_c),
                        b0_c=float(blasso.b0_c),
                        a0_L=float(blasso.a0_L),
                        b0_L=float(blasso.b0_L),
                        c_mask=c_mask,
                        rng=rng,
                    )
            elif prior_family == "dl":
                if dl_psi is None or dl_vartheta is None or dl_zeta is None or dl_inv_v0 is None or dl is None:
                    raise RuntimeError("dl state missing")
                dl_psi, dl_vartheta, dl_zeta, dl_inv_v0 = _dl_update(
                    beta=beta_lags,
                    psi=dl_psi,
                    vartheta=dl_vartheta,
                    zeta=dl_zeta,
                    abeta=float(dl.abeta),
                    rng=rng,
                )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta_full.copy())
                h_keep.append(h.copy())
                h0_keep.append(h0.copy())
                sigma_eta2_keep.append(sigma_eta2.copy())
                mu_keep.append(mu.copy())
                if mu_gamma is not None:
                    mu_gamma_keep.append(mu_gamma.copy())
                if y_lat_keep is not None:
                    y_lat_keep.append(y_lat.copy())

        latent_dataset = None
        if model.elb is not None and model.elb.enabled:
            latent_dataset = Dataset.from_arrays(values=y_lat, variables=dataset.variables, time_index=dataset.time_index)

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
            h_draws=np.stack(h_keep) if h_keep else None,
            h0_draws=np.stack(h0_keep) if h0_keep else None,
            sigma_eta2_draws=np.stack(sigma_eta2_keep) if sigma_eta2_keep else None,
            mu_draws=np.stack(mu_keep) if mu_keep else None,
            mu_gamma_draws=np.stack(mu_gamma_keep) if mu_gamma_keep else None,
        )

    applies_to_idx: list[int] = []
    elb_t_idx: dict[int, np.ndarray] = {}

    y_lat = dataset.values.copy()
    if model.elb is not None and model.elb.enabled:
        for name in model.elb.applies_to:
            try:
                applies_to_idx.append(dataset.variables.index(name))
            except ValueError as e:
                raise ValueError(f"elb.applies_to contains unknown variable: {name}") from e

        for j in applies_to_idx:
            mask = dataset.values[:, j] <= (model.elb.bound + model.elb.tol)
            elb_t_idx[j] = np.where(mask)[0]
            y_lat[mask, j] = model.elb.bound - model.elb.init_offset

    x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)
    t_eff, n = y.shape

    niw = prior.niw
    prior_family = prior.family.lower()
    blasso = prior.blasso if prior_family == "blasso" else None
    if prior_family == "blasso" and blasso is None:
        raise ValueError("prior.family='blasso' requires prior.blasso")
    dl = prior.dl if prior_family == "dl" else None
    if prior_family == "dl" and dl is None:
        raise ValueError("prior.family='dl' requires prior.dl")

    tau: np.ndarray | None = None
    lambda_: float | None = None
    lambda_c: float | None = None
    lambda_L: float | None = None
    c_mask: np.ndarray | None = None

    dl_psi: np.ndarray | None = None
    dl_vartheta: np.ndarray | None = None
    dl_zeta: float | None = None
    dl_inv_v0: np.ndarray | None = None

    mn, _vn, _sn, _nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
    beta = mn.copy()

    if prior_family == "blasso":
        if blasso is None:
            raise RuntimeError("blasso spec missing")
        tau = np.full(x.shape[1], float(blasso.tau_init), dtype=float)
        if blasso.mode == "global":
            lambda_ = float(blasso.lambda_init)
        else:
            lambda_c = float(blasso.lambda_init)
            lambda_L = float(blasso.lambda_init)
            c_mask = np.zeros(x.shape[1], dtype=bool)
            if model.include_intercept:
                c_mask[0] = True

    if prior_family == "dl":
        if dl is None:
            raise RuntimeError("dl spec missing")
        km = int(x.shape[1] * y.shape[1])
        dl_psi = np.full(km, float(dl.dl_scaler), dtype=float)
        dl_vartheta = np.full(km, float(dl.dl_scaler), dtype=float)
        dl_zeta = float(dl.dl_scaler)
        dl_inv_v0 = 1.0 / (dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6)

    h0 = np.log(np.var(y, axis=0) + 1e-12)
    h = np.tile(h0.reshape(1, -1), (t_eff, 1))
    sigma_eta2 = 0.05 * np.ones(n, dtype=float)

    beta_keep: list[np.ndarray] = []
    h_keep: list[np.ndarray] = []
    h0_keep: list[np.ndarray] = []
    sigma_eta2_keep: list[np.ndarray] = []
    y_lat_keep: list[np.ndarray] | None = [] if (model.elb is not None and model.elb.enabled) else None

    for it in range(sampler.draws):
        x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        if prior_family == "dl":
            if dl_inv_v0 is None:
                raise RuntimeError("dl state missing")
            beta = _dl_sample_beta_svrw(x=x, y=y, m0=niw.m0, inv_v0_vec=dl_inv_v0, h=h, rng=rng)
        else:
            if prior_family == "blasso":
                if tau is None or blasso is None:
                    raise RuntimeError("blasso state missing")
                v0 = _blasso_v0_from_state(tau=tau)
            else:
                v0 = niw.v0
            beta = sample_beta_svrw(x=x, y=y, m0=niw.m0, v0=v0, h=h, rng=rng)

        if model.elb is not None and model.elb.enabled:
            h_full = np.vstack([np.tile(h0.reshape(1, -1), (model.p, 1)), h])
            for j in applies_to_idx:
                for t in elb_t_idx[j]:
                    y_lat[t, j] = sample_shadow_value_svrw(
                        y=y_lat,
                        h=h_full,
                        t=int(t),
                        j=int(j),
                        p=model.p,
                        beta=beta,
                        upper=model.elb.bound,
                        include_intercept=model.include_intercept,
                        rng=rng,
                    )

            x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        e = y - x @ beta

        for i in range(n):
            y_star = log_e2_star(e[:, i], epsilon=vol.epsilon)
            h[:, i] = sample_h_svrw(
                y_star=y_star,
                h=h[:, i],
                sigma_eta2=float(sigma_eta2[i]),
                h0=float(h0[i]),
                rng=rng,
            )
            h0[i] = sample_h0(
                h1=float(h[0, i]),
                sigma_eta2=float(sigma_eta2[i]),
                prior_mean=vol.h0_prior_mean,
                prior_var=vol.h0_prior_var,
                rng=rng,
            )
            sigma_eta2[i] = sample_sigma_eta2(
                h=h[:, i],
                h0=float(h0[i]),
                nu0=vol.sigma_eta_prior_nu0,
                s0=vol.sigma_eta_prior_s0,
                rng=rng,
            )

        if prior_family == "blasso":
            if blasso is None or tau is None:
                raise RuntimeError("blasso state missing")
            if blasso.mode == "global":
                if lambda_ is None:
                    raise RuntimeError("lambda missing")
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

        elif prior_family == "dl":
            if dl_psi is None or dl_vartheta is None or dl_zeta is None or dl_inv_v0 is None or dl is None:
                raise RuntimeError("dl state missing")
            dl_psi, dl_vartheta, dl_zeta, dl_inv_v0 = _dl_update(
                beta=beta,
                psi=dl_psi,
                vartheta=dl_vartheta,
                zeta=dl_zeta,
                abeta=float(dl.abeta),
                rng=rng,
            )

        if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
            beta_keep.append(beta.copy())
            h_keep.append(h.copy())
            h0_keep.append(h0.copy())
            sigma_eta2_keep.append(sigma_eta2.copy())
            if y_lat_keep is not None:
                y_lat_keep.append(y_lat.copy())

    latent_dataset = None
    if model.elb is not None and model.elb.enabled:
        latent_dataset = Dataset.from_arrays(values=y_lat, variables=dataset.variables, time_index=dataset.time_index)

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
        h_draws=np.stack(h_keep) if h_keep else None,
        h0_draws=np.stack(h0_keep) if h0_keep else None,
        sigma_eta2_draws=np.stack(sigma_eta2_keep) if sigma_eta2_keep else None,
    )
