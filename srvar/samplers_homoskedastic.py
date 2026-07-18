from __future__ import annotations

import numpy as np

from .bvar import posterior_niw, sample_posterior_niw
from .data.dataset import Dataset
from .elb import sample_shadow_value
from .results import FitResult, PosteriorNIW
from .samplers_blasso import _blasso_update_adaptive, _blasso_update_global, _blasso_v0_from_state
from .samplers_dl import _dl_sample_beta_sigma, _dl_update
from .samplers_ssp import (
    _asum_from_beta,
    _strip_intercept_eqwise_precision,
    _strip_intercept_niw_blocks,
    sample_mu_gamma,
    sample_steady_state_mu,
)
from .shocks import update_precision_scales
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .ssvs import sample_gamma_rows, v0_diag_from_gamma
from .var import demean_data, design_matrix


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

        ss_gamma: np.ndarray | None = None
        ss_fixed_mask: np.ndarray | None = None

        tau: np.ndarray | None = None
        lambda_: float | None = None
        lambda_c: float | None = None
        lambda_L: float | None = None
        c_mask: np.ndarray | None = None

        dl_psi: np.ndarray | None = None
        dl_vartheta: np.ndarray | None = None
        dl_zeta: float | None = None
        dl_inv_v0: np.ndarray | None = None
        canonical = prior.minnesota_canonical if prior_family == "minnesota_canonical" else None

        ss_beta_keep: list[np.ndarray] = []
        ss_sigma_keep: list[np.ndarray] = []
        ss_gamma_keep: list[np.ndarray] = []
        ss_mu_keep: list[np.ndarray] = []
        ss_mu_gamma_keep: list[np.ndarray] = []
        ss_last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            y_dm = demean_data(y_lat, mu)
            x, y = design_matrix(y_dm, model.p, include_intercept=False)
            _t_eff, k = x.shape

            m0_ssp, v0_ssp = _strip_intercept_niw_blocks(m0=niw.m0, v0=niw.v0, k_no_intercept=k)

            v0_used = v0_ssp
            if prior_family == "niw":
                mn, vn, sn, nun = posterior_niw(
                    x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0
                )
                ss_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

            elif prior_family == "ssvs":
                if prior.ssvs is None:
                    raise ValueError("prior.family='ssvs' requires prior.ssvs")
                spec = prior.ssvs

                if ss_gamma is None:
                    ss_gamma = np.asarray(
                        rng.uniform(size=k) < float(spec.inclusion_prob), dtype=bool
                    )
                    ss_fixed_mask = np.zeros(k, dtype=bool)
                if ss_fixed_mask is None:
                    raise RuntimeError("ssvs fixed-mask state missing")

                v0_diag = v0_diag_from_gamma(
                    gamma=ss_gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    intercept_slab_var=None,
                )
                v0_used = np.diag(v0_diag)

                mn, vn, sn, nun = posterior_niw(
                    x=x, y=y, m0=m0_ssp, v0=v0_used, s0=niw.s0, nu0=niw.nu0
                )
                ss_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

                ss_gamma = sample_gamma_rows(
                    beta=beta_lags,
                    sigma=sigma,
                    gamma=ss_gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    inclusion_prob=spec.inclusion_prob,
                    fixed_mask=ss_fixed_mask,
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

                mn, vn, sn, nun = posterior_niw(
                    x=x, y=y, m0=m0_ssp, v0=v0_used, s0=niw.s0, nu0=niw.nu0
                )
                ss_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
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
                    dl_inv_v0 = 1.0 / (
                        dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6
                    )

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
                mn, vn, sn, nun = posterior_niw(
                    x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0
                )
                ss_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

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
            elif prior_family == "minnesota_canonical":
                if canonical is None:
                    raise ValueError(
                        "prior_family='minnesota_canonical' requires prior.minnesota_canonical"
                    )
                beta_lags, sigma = _dl_sample_beta_sigma(
                    x=x,
                    y=y,
                    m0=m0_ssp,
                    inv_v0_vec=_strip_intercept_eqwise_precision(
                        inv_v0_vec=canonical.inv_v0_vec,
                        n=y.shape[1],
                        k_no_intercept=k,
                    ),
                    s0=niw.s0,
                    nu0=niw.nu0,
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
                ss_beta_keep.append(beta_full.copy())
                ss_sigma_keep.append(sigma.copy())
                ss_mu_keep.append(mu.copy())
                if mu_gamma is not None:
                    ss_mu_gamma_keep.append(mu_gamma.copy())
                if ss_gamma is not None:
                    g = ss_gamma
                    if model.include_intercept:
                        g = np.concatenate([np.array([True], dtype=bool), ss_gamma])
                    ss_gamma_keep.append(g.copy())

        if ss_last_posterior is None and prior_family != "minnesota_canonical":
            raise RuntimeError("sampler.draws produced no posterior")

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=ss_last_posterior,
            beta_draws=np.stack(ss_beta_keep) if ss_beta_keep else None,
            sigma_draws=np.stack(ss_sigma_keep) if ss_sigma_keep else None,
            gamma_draws=np.stack(ss_gamma_keep) if ss_gamma_keep else None,
            mu_draws=np.stack(ss_mu_keep) if ss_mu_keep else None,
            mu_gamma_draws=np.stack(ss_mu_gamma_keep) if ss_mu_gamma_keep else None,
        )

    x, y = design_matrix(dataset.values, model.p, include_intercept=model.include_intercept)

    niw = prior.niw
    robust = model.shocks is not None and model.shocks.family != "gaussian"
    if robust and model.steady_state is not None:
        raise ValueError("robust shocks with steady_state are not supported")
    if prior_family == "niw":
        if robust:
            assert model.shocks is not None
            lam = np.ones(int(x.shape[0]), dtype=float)

            robust_niw_beta_keep: list[np.ndarray] = []
            robust_niw_sigma_keep: list[np.ndarray] = []
            robust_niw_last_posterior: PosteriorNIW | None = None

            for it in range(sampler.draws):
                sqrt_lam = np.sqrt(lam).reshape(-1, 1)
                x_w = x * sqrt_lam
                y_w = y * sqrt_lam

                mn, vn, sn, nun = posterior_niw(
                    x=x_w, y=y_w, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0
                )
                robust_niw_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
                beta = beta_draws[0]
                sigma = sigma_draws[0]

                resid = y - x @ beta
                lam = update_precision_scales(errors=resid, sigma=sigma, spec=model.shocks, rng=rng)

                if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                    robust_niw_beta_keep.append(beta.copy())
                    robust_niw_sigma_keep.append(sigma.copy())

            if robust_niw_last_posterior is None:
                raise RuntimeError("sampler.draws produced no posterior")

            return FitResult(
                dataset=dataset,
                model=model,
                prior=prior,
                sampler=sampler,
                posterior=robust_niw_last_posterior,
                beta_draws=np.stack(robust_niw_beta_keep) if robust_niw_beta_keep else None,
                sigma_draws=np.stack(robust_niw_sigma_keep) if robust_niw_sigma_keep else None,
            )

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
        beta_kept = beta_all[keep_idx] if keep_idx.size > 0 else None
        sigma_kept = sigma_all[keep_idx] if keep_idx.size > 0 else None

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=posterior,
            beta_draws=beta_kept,
            sigma_draws=sigma_kept,
        )

    if prior_family == "minnesota_canonical":
        canonical = prior.minnesota_canonical
        if canonical is None:
            raise ValueError(
                "prior_family='minnesota_canonical' requires prior.minnesota_canonical"
            )

        canonical_beta_keep: list[np.ndarray] = []
        canonical_sigma_keep: list[np.ndarray] = []

        canonical_precision = np.ones(int(x.shape[0]), dtype=float) if robust else None

        for it in range(sampler.draws):
            if robust:
                assert canonical_precision is not None
                sqrt_lam = np.sqrt(canonical_precision).reshape(-1, 1)
                x_w = x * sqrt_lam
                y_w = y * sqrt_lam
            else:
                x_w = x
                y_w = y

            beta, sigma = _dl_sample_beta_sigma(
                x=x_w,
                y=y_w,
                m0=niw.m0,
                inv_v0_vec=canonical.inv_v0_vec,
                s0=niw.s0,
                nu0=niw.nu0,
                rng=rng,
            )

            if robust:
                assert model.shocks is not None
                resid = y - x @ beta
                canonical_precision = update_precision_scales(
                    errors=resid, sigma=sigma, spec=model.shocks, rng=rng
                )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                canonical_beta_keep.append(beta.copy())
                canonical_sigma_keep.append(sigma.copy())

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=None,
            beta_draws=np.stack(canonical_beta_keep) if canonical_beta_keep else None,
            sigma_draws=np.stack(canonical_sigma_keep) if canonical_sigma_keep else None,
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

        blasso_precision = np.ones(int(x.shape[0]), dtype=float) if robust else None

        blasso_beta_keep: list[np.ndarray] = []
        blasso_sigma_keep: list[np.ndarray] = []
        blasso_last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            v0 = _blasso_v0_from_state(tau=tau)

            if robust:
                assert blasso_precision is not None
                sqrt_lam = np.sqrt(blasso_precision).reshape(-1, 1)
                x_w = x * sqrt_lam
                y_w = y * sqrt_lam
            else:
                x_w = x
                y_w = y

            mn, vn, sn, nun = posterior_niw(x=x_w, y=y_w, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
            blasso_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(
                mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
            )
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

            if robust:
                assert model.shocks is not None
                resid = y - x @ beta
                blasso_precision = update_precision_scales(
                    errors=resid, sigma=sigma, spec=model.shocks, rng=rng
                )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                blasso_beta_keep.append(beta.copy())
                blasso_sigma_keep.append(sigma.copy())

        if blasso_last_posterior is None:
            raise RuntimeError("sampler.draws produced no posterior")

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=blasso_last_posterior,
            beta_draws=np.stack(blasso_beta_keep) if blasso_beta_keep else None,
            sigma_draws=np.stack(blasso_sigma_keep) if blasso_sigma_keep else None,
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

        dl_precision = np.ones(int(x.shape[0]), dtype=float) if robust else None

        dl_beta_keep: list[np.ndarray] = []
        dl_sigma_keep: list[np.ndarray] = []
        dl_last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            if robust:
                assert dl_precision is not None
                sqrt_lam = np.sqrt(dl_precision).reshape(-1, 1)
                x_w = x * sqrt_lam
                y_w = y * sqrt_lam
            else:
                x_w = x
                y_w = y

            beta, sigma = _dl_sample_beta_sigma(
                x=x_w,
                y=y_w,
                m0=niw.m0,
                inv_v0_vec=inv_v0,
                s0=niw.s0,
                nu0=niw.nu0,
                rng=rng,
            )
            mn, vn, sn, nun = posterior_niw(
                x=x_w, y=y_w, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0
            )
            dl_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            psi, vartheta, zeta, inv_v0 = _dl_update(
                beta=beta,
                psi=psi,
                vartheta=vartheta,
                zeta=zeta,
                abeta=float(spec_d.abeta),
                rng=rng,
            )

            if robust:
                assert model.shocks is not None
                resid = y - x @ beta
                dl_precision = update_precision_scales(
                    errors=resid, sigma=sigma, spec=model.shocks, rng=rng
                )

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                dl_beta_keep.append(beta.copy())
                dl_sigma_keep.append(sigma.copy())

        if dl_last_posterior is None:
            raise RuntimeError("sampler.draws produced no posterior")

        return FitResult(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            posterior=dl_last_posterior,
            beta_draws=np.stack(dl_beta_keep) if dl_beta_keep else None,
            sigma_draws=np.stack(dl_sigma_keep) if dl_sigma_keep else None,
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

    ssvs_gamma = np.asarray(rng.uniform(size=k) < spec.inclusion_prob, dtype=bool)
    ssvs_fixed_mask = np.zeros(k, dtype=bool)
    if model.include_intercept and spec.fix_intercept:
        ssvs_fixed_mask[0] = True
        ssvs_gamma[0] = True

    ssvs_beta_keep: list[np.ndarray] = []
    ssvs_sigma_keep: list[np.ndarray] = []
    ssvs_gamma_keep: list[np.ndarray] = []
    ssvs_last_posterior: PosteriorNIW | None = None
    ssvs_precision = np.ones(int(x.shape[0]), dtype=float) if robust else None

    for it in range(sampler.draws):
        if robust:
            assert ssvs_precision is not None
            sqrt_lam = np.sqrt(ssvs_precision).reshape(-1, 1)
            x_w = x * sqrt_lam
            y_w = y * sqrt_lam
        else:
            x_w = x
            y_w = y

        v0_diag = v0_diag_from_gamma(
            gamma=ssvs_gamma,
            spike_var=spec.spike_var,
            slab_var=spec.slab_var,
            intercept_slab_var=spec.intercept_slab_var,
        )
        v0 = np.diag(v0_diag)

        mn, vn, sn, nun = posterior_niw(x=x_w, y=y_w, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
        ssvs_last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

        beta_draws, sigma_draws = sample_posterior_niw(
            mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
        )
        beta = beta_draws[0]
        sigma = sigma_draws[0]

        ssvs_gamma = sample_gamma_rows(
            beta=beta,
            sigma=sigma,
            gamma=ssvs_gamma,
            spike_var=spec.spike_var,
            slab_var=spec.slab_var,
            inclusion_prob=spec.inclusion_prob,
            fixed_mask=ssvs_fixed_mask,
            rng=rng,
        )

        if robust:
            assert model.shocks is not None
            resid = y - x @ beta
            ssvs_precision = update_precision_scales(
                errors=resid, sigma=sigma, spec=model.shocks, rng=rng
            )

        if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
            ssvs_beta_keep.append(beta.copy())
            ssvs_sigma_keep.append(sigma.copy())
            ssvs_gamma_keep.append(ssvs_gamma.copy())

    if ssvs_last_posterior is None:
        raise RuntimeError("sampler.draws produced no posterior")

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=ssvs_last_posterior,
        beta_draws=np.stack(ssvs_beta_keep) if ssvs_beta_keep else None,
        sigma_draws=np.stack(ssvs_sigma_keep) if ssvs_sigma_keep else None,
        gamma_draws=np.stack(ssvs_gamma_keep) if ssvs_gamma_keep else None,
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

        elb_ss_gamma: np.ndarray | None = None
        elb_ss_fixed_mask: np.ndarray | None = None

        tau: np.ndarray | None = None
        lambda_: float | None = None
        lambda_c: float | None = None
        lambda_L: float | None = None
        c_mask: np.ndarray | None = None

        dl_psi: np.ndarray | None = None
        dl_vartheta: np.ndarray | None = None
        dl_zeta: float | None = None
        dl_inv_v0: np.ndarray | None = None
        canonical = prior.minnesota_canonical if prior_family == "minnesota_canonical" else None

        last_posterior: PosteriorNIW | None = None

        for it in range(sampler.draws):
            y_dm = demean_data(y_lat, mu)
            x, y = design_matrix(y_dm, model.p, include_intercept=False)
            _t_eff, k = x.shape

            m0_ssp, v0_ssp = _strip_intercept_niw_blocks(m0=niw.m0, v0=niw.v0, k_no_intercept=k)

            if prior_family == "niw":
                mn, vn, sn, nun = posterior_niw(
                    x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0
                )
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

            elif prior_family == "ssvs":
                if prior.ssvs is None:
                    raise ValueError("prior.family='ssvs' requires prior.ssvs")
                spec = prior.ssvs

                if elb_ss_gamma is None:
                    elb_ss_gamma = np.asarray(
                        rng.uniform(size=k) < float(spec.inclusion_prob), dtype=bool
                    )
                    elb_ss_fixed_mask = np.zeros(k, dtype=bool)
                if elb_ss_fixed_mask is None:
                    raise RuntimeError("ssvs fixed-mask state missing")

                v0_diag = v0_diag_from_gamma(
                    gamma=elb_ss_gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    intercept_slab_var=None,
                )
                v0 = np.diag(v0_diag)

                mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=m0_ssp, v0=v0, s0=niw.s0, nu0=niw.nu0)
                last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)
                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
                beta_lags = beta_draws[0]
                sigma = sigma_draws[0]

                elb_ss_gamma = sample_gamma_rows(
                    beta=beta_lags,
                    sigma=sigma,
                    gamma=elb_ss_gamma,
                    spike_var=spec.spike_var,
                    slab_var=spec.slab_var,
                    inclusion_prob=spec.inclusion_prob,
                    fixed_mask=elb_ss_fixed_mask,
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
                beta_draws, sigma_draws = sample_posterior_niw(
                    mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
                )
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
                    dl_inv_v0 = 1.0 / (
                        dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6
                    )

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

                mn, vn, sn, nun = posterior_niw(
                    x=x, y=y, m0=m0_ssp, v0=v0_ssp, s0=niw.s0, nu0=niw.nu0
                )
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
            elif prior_family == "minnesota_canonical":
                if canonical is None:
                    raise ValueError(
                        "prior_family='minnesota_canonical' requires prior.minnesota_canonical"
                    )
                beta_lags, sigma = _dl_sample_beta_sigma(
                    x=x,
                    y=y,
                    m0=m0_ssp,
                    inv_v0_vec=_strip_intercept_eqwise_precision(
                        inv_v0_vec=canonical.inv_v0_vec,
                        n=y.shape[1],
                        k_no_intercept=k,
                    ),
                    s0=niw.s0,
                    nu0=niw.nu0,
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
                if elb_ss_gamma is not None:
                    g = elb_ss_gamma
                    if model.include_intercept:
                        g = np.concatenate([np.array([True], dtype=bool), elb_ss_gamma])
                    gamma_keep.append(g.copy())

        if last_posterior is None and prior_family != "minnesota_canonical":
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

    beta_keep = []
    sigma_keep = []
    y_lat_keep = []
    gamma_keep = []

    niw = prior.niw

    elb_gamma: np.ndarray | None = None
    elb_fixed_mask: np.ndarray | None = None
    ssvs_spec = prior.ssvs if prior_family == "ssvs" else None
    blasso = prior.blasso if prior_family == "blasso" else None
    dl = prior.dl if prior_family == "dl" else None
    canonical = prior.minnesota_canonical if prior_family == "minnesota_canonical" else None

    tau = None
    lambda_ = None
    lambda_c = None
    lambda_L = None
    c_mask = None

    dl_psi = None
    dl_vartheta = None
    dl_zeta = None
    dl_inv_v0 = None

    last_posterior = None

    for it in range(sampler.draws):
        x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        if prior_family == "niw":
            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(
                mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
            )
            beta = beta_draws[0]
            sigma = sigma_draws[0]
        elif prior_family == "ssvs":
            if ssvs_spec is None:
                raise ValueError("prior.family='ssvs' requires prior.ssvs")

            t_eff, k = x.shape
            _n = y.shape[1]
            if niw.m0.shape != (k, _n):
                raise ValueError("ssvs requires prior.niw.m0 with shape (K, N)")
            if niw.s0.shape != (_n, _n):
                raise ValueError("ssvs requires prior.niw.s0 with shape (N, N)")

            if elb_gamma is None:
                elb_gamma = np.asarray(rng.uniform(size=k) < ssvs_spec.inclusion_prob, dtype=bool)
                elb_fixed_mask = np.zeros(k, dtype=bool)
                if model.include_intercept and ssvs_spec.fix_intercept:
                    elb_fixed_mask[0] = True
                    elb_gamma[0] = True
            if elb_fixed_mask is None:
                raise RuntimeError("ssvs fixed-mask state missing")

            v0_diag = v0_diag_from_gamma(
                gamma=elb_gamma,
                spike_var=ssvs_spec.spike_var,
                slab_var=ssvs_spec.slab_var,
                intercept_slab_var=ssvs_spec.intercept_slab_var,
            )
            v0 = np.diag(v0_diag)

            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(
                mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
            )
            beta = beta_draws[0]
            sigma = sigma_draws[0]

            elb_gamma = sample_gamma_rows(
                beta=beta,
                sigma=sigma,
                gamma=elb_gamma,
                spike_var=ssvs_spec.spike_var,
                slab_var=ssvs_spec.slab_var,
                inclusion_prob=ssvs_spec.inclusion_prob,
                fixed_mask=elb_fixed_mask,
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
                dl_inv_v0 = 1.0 / (
                    dl_psi * (dl_vartheta * dl_vartheta) * (dl_zeta * dl_zeta) + 1e-6
                )

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

        elif prior_family == "minnesota_canonical":
            if canonical is None:
                raise ValueError(
                    "prior_family='minnesota_canonical' requires prior.minnesota_canonical"
                )
            beta, sigma = _dl_sample_beta_sigma(
                x=x,
                y=y,
                m0=niw.m0,
                inv_v0_vec=canonical.inv_v0_vec,
                s0=niw.s0,
                nu0=niw.nu0,
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

            beta_draws, sigma_draws = sample_posterior_niw(
                mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng
            )
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
            if elb_gamma is not None:
                gamma_keep.append(elb_gamma.copy())

    if last_posterior is None and prior_family != "minnesota_canonical":
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
