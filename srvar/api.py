from __future__ import annotations

import numpy as np

from .bvar import posterior_niw, sample_posterior_niw, simulate_var_forecast
from .data.dataset import Dataset
from .results import FitResult, ForecastResult, PosteriorNIW
from .elb import apply_elb_floor, sample_shadow_value, sample_shadow_value_svrw
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .rng import gamma_rate, inverse_gaussian
from .ssvs import sample_gamma_rows, v0_diag_from_gamma
from .sv import log_e2_star, sample_beta_svrw, sample_h0, sample_h_svrw, sample_sigma_eta2
from .var import design_matrix


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
            y_lat[mask, j] = model.elb.bound - 0.05

    x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)
    t_eff, n = y.shape

    niw = prior.niw
    prior_family = prior.family.lower()
    blasso = prior.blasso if prior_family == "blasso" else None
    if prior_family == "blasso" and blasso is None:
        raise ValueError("prior.family='blasso' requires prior.blasso")

    tau: np.ndarray | None = None
    lambda_: float | None = None
    lambda_c: float | None = None
    lambda_L: float | None = None
    c_mask: np.ndarray | None = None
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
                if lambda_c is None or lambda_L is None or c_mask is None:
                    raise RuntimeError("adaptive lambda state missing")
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


def _fit_no_elb(
    *,
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    prior_family: str,
    rng: np.random.Generator,
) -> FitResult:
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
        y_lat[mask, j] = elb.bound - 0.05

    beta_keep: list[np.ndarray] = []
    sigma_keep: list[np.ndarray] = []
    y_lat_keep: list[np.ndarray] = []
    gamma_keep: list[np.ndarray] = []

    niw = prior.niw

    gamma: np.ndarray | None = None
    fixed_mask: np.ndarray | None = None
    spec = prior.ssvs if prior_family == "ssvs" else None
    blasso = prior.blasso if prior_family == "blasso" else None

    tau: np.ndarray | None = None
    lambda_: float | None = None
    lambda_c: float | None = None
    lambda_L: float | None = None
    c_mask: np.ndarray | None = None

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


def fit(
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    *,
    rng: np.random.Generator | None = None,
) -> FitResult:
    """Fit an SRVAR/BVAR model and return posterior draws.

    This is the primary user-facing entry point for estimating models in this package.

    Supported configurations
    ------------------------
    - Conjugate BVAR with Normal-Inverse-Wishart prior (``prior.family='niw'``)
    - Spike-and-slab variable selection (``prior.family='ssvs'``)
    - Effective lower bound (ELB) data-augmentation Gibbs sampler (``model.elb.enabled``)
    - Stochastic volatility random-walk (SVRW) (``model.volatility.enabled``; requires NIW)

    Parameters
    ----------
    dataset:
        Observed data (T, N). When ELB is enabled, some variables are interpreted as
        censored at the ELB.
    model:
        Model configuration, including lag order ``p`` and optional ELB/SV specs.
    prior:
        Prior configuration. The prior family is determined by ``prior.family``.
    sampler:
        MCMC configuration (draws, burn-in, thinning).
    rng:
        Optional NumPy RNG.

    Returns
    -------
    FitResult
        Fit result with posterior parameters and/or stored posterior draws depending on the
        model configuration.

    Notes
    -----
    For conjugate NIW without ELB/SV, ``FitResult.posterior`` is populated and
    ``beta_draws``/``sigma_draws`` are produced by direct sampling.

    For Gibbs samplers (ELB, SSVS, SV), burn-in and thinning are applied online and the
    returned ``*_draws`` arrays contain only kept draws.

    When ELB is enabled, the latent series is initialized by setting observations at or
    below the bound to a small amount below the bound (currently ``bound - 0.05``). This is
    a numerical initialization choice intended to avoid starting exactly at the truncation
    boundary.
    """
    prior_family = prior.family.lower()
    if prior_family not in {"niw", "ssvs", "blasso"}:
        raise ValueError("only prior.family in {'niw','ssvs','blasso'} is supported")

    if rng is None:
        rng = np.random.default_rng()

    if model.volatility is not None and model.volatility.enabled:
        if prior_family not in {"niw", "blasso"}:
            raise ValueError("stochastic volatility currently requires prior.family in {'niw','blasso'}")
        return _fit_svrw(dataset=dataset, model=model, prior=prior, sampler=sampler, rng=rng)

    if model.elb is None or not model.elb.enabled:
        return _fit_no_elb(dataset=dataset, model=model, prior=prior, sampler=sampler, prior_family=prior_family, rng=rng)

    # Phase 3: ELB data-augmentation Gibbs
    # Initialize latent series: start at observed, but nudge ELB-bound observations slightly below bound
    # Update latent shadow rates at ELB observations
    return _fit_elb_gibbs(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        prior_family=prior_family,
        rng=rng,
    )


def forecast(
    fit: FitResult,
    horizons: list[int],
    *,
    draws: int = 1000,
    quantile_levels: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> ForecastResult:
    """Generate predictive simulations from a fitted model.

    Forecasts are produced by Monte Carlo simulation using posterior parameter draws.

    Parameters
    ----------
    fit:
        Result from :func:`srvar.api.fit`.
    horizons:
        List of forecast horizons (in steps) requested by the caller. Internally,
        simulations are generated out to ``H = max(horizons)``.
    draws:
        Number of predictive simulation paths.
    quantile_levels:
        Quantiles to compute from the simulated draws. Defaults to ``[0.1, 0.5, 0.9]``.
    rng:
        Optional NumPy RNG.

    Returns
    -------
    ForecastResult
        Forecast result containing:

        - ``draws``: simulated observations with shape ``(D, H, N)``
        - ``mean``: mean across draws with shape ``(H, N)``
        - ``quantiles``: dict mapping each requested quantile to an array ``(H, N)``

    Notes
    -----
    If ELB is enabled in the fitted model, returned ``draws`` are the observed (floored)
    draws, and ``latent_draws`` contains the unconstrained latent draws.

    If you call ``forecast(fit, horizons=[1, 3], ...)`` then ``result.mean[0]`` corresponds
    to horizon 1 and ``result.mean[2]`` corresponds to horizon 3.
    """
    if rng is None:
        rng = np.random.default_rng()

    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool):
        raise ValueError("draws must be an integer")
    if int(draws) < 1:
        raise ValueError("draws must be >= 1")
    draws = int(draws)

    if not isinstance(horizons, list) or len(horizons) == 0:
        raise ValueError("horizons must be a non-empty list of positive integers")
    horizons_int: list[int] = []
    for h in horizons:
        if not isinstance(h, (int, np.integer)) or isinstance(h, bool):
            raise ValueError("horizons must contain only integers")
        hi = int(h)
        if hi < 1:
            raise ValueError("horizons must contain only positive integers")
        horizons_int.append(hi)
    horizons = horizons_int

    if not isinstance(quantile_levels, list) or len(quantile_levels) == 0:
        raise ValueError("quantile_levels must be a non-empty list")
    quantiles_float: list[float] = []
    for q in quantile_levels:
        qf = float(q)
        if not np.isfinite(qf) or not (0.0 < qf < 1.0):
            raise ValueError("quantile_levels must be finite and in (0, 1)")
        quantiles_float.append(qf)
    quantile_levels = quantiles_float

    hmax = int(max(horizons))
    p = fit.model.p

    if fit.posterior is None and fit.beta_draws is None:
        raise ValueError(
            "fit does not contain posterior parameters or stored draws; "
            "this can happen if burn_in/thin leaves zero kept draws"
        )

    base_dataset = fit.latent_dataset if fit.latent_dataset is not None else fit.dataset
    if base_dataset.T < p:
        raise ValueError("dataset is too short for requested lag order p")
    y_last = base_dataset.values[-p:, :]

    if fit.beta_draws is not None and fit.sigma_draws is not None:
        # sample with replacement from stored posterior draws
        idx = rng.integers(0, fit.beta_draws.shape[0], size=draws)
        beta_draws = fit.beta_draws[idx]
        sigma_draws = fit.sigma_draws[idx]

        sims = np.empty((draws, hmax, fit.dataset.N), dtype=float)
        for d in range(draws):
            sims[d] = simulate_var_forecast(
                y_last=y_last,
                beta=beta_draws[d],
                sigma=sigma_draws[d],
                horizon=hmax,
                include_intercept=fit.model.include_intercept,
                rng=rng,
            )
    elif fit.beta_draws is not None and fit.h_draws is not None and fit.sigma_eta2_draws is not None:
        idx = rng.integers(0, fit.beta_draws.shape[0], size=draws)
        beta_draws = fit.beta_draws[idx]
        h_draws = fit.h_draws[idx]
        sigma_eta2_draws = fit.sigma_eta2_draws[idx]

        sims = np.empty((draws, hmax, fit.dataset.N), dtype=float)
        for d in range(draws):
            lags = y_last.copy()
            h_curr = h_draws[d, -1, :].copy()
            sig_eta = sigma_eta2_draws[d].copy()
            path = np.empty((hmax, fit.dataset.N), dtype=float)
            for h_step in range(hmax):
                x_parts: list[np.ndarray] = []
                if fit.model.include_intercept:
                    x_parts.append(np.array([1.0], dtype=float))
                for lag in range(1, p + 1):
                    x_parts.append(lags[-lag, :])
                x_row = np.concatenate(x_parts)

                mean = x_row @ beta_draws[d]
                eps = rng.normal(size=fit.dataset.N) * np.exp(0.5 * h_curr)
                y_next = mean + eps

                path[h_step] = y_next
                lags = np.vstack([lags[1:, :], y_next]) if p > 1 else y_next.reshape(1, -1)
                h_curr = h_curr + np.sqrt(sig_eta) * rng.normal(size=fit.dataset.N)

            sims[d] = path
    else:
        if fit.posterior is None:
            raise ValueError("fit has no posterior parameters or stored draws")
        beta_draws, sigma_draws = sample_posterior_niw(
            mn=fit.posterior.mn,
            vn=fit.posterior.vn,
            sn=fit.posterior.sn,
            nun=fit.posterior.nun,
            draws=draws,
            rng=rng,
        )

        sims = np.empty((draws, hmax, fit.dataset.N), dtype=float)
        for d in range(draws):
            sims[d] = simulate_var_forecast(
                y_last=y_last,
                beta=beta_draws[d],
                sigma=sigma_draws[d],
                horizon=hmax,
                include_intercept=fit.model.include_intercept,
                rng=rng,
            )

    mean = sims.mean(axis=0)
    quantiles = {q: np.quantile(sims, q=q, axis=0) for q in quantile_levels}

    latent_sims: np.ndarray | None = None

    # If ELB enabled, return observed draws (apply floor) for constrained variables
    if fit.model.elb is not None and fit.model.elb.enabled:
        latent_sims = sims.copy()
        applies_to_idx = [fit.dataset.variables.index(v) for v in fit.model.elb.applies_to]
        sims = apply_elb_floor(sims, bound=fit.model.elb.bound, indices=applies_to_idx)
        mean = sims.mean(axis=0)
        quantiles = {q: np.quantile(sims, q=q, axis=0) for q in quantile_levels}

    return ForecastResult(
        variables=list(fit.dataset.variables),
        horizons=list(horizons),
        draws=sims,
        latent_draws=latent_sims,
        mean=mean,
        quantiles=quantiles,
    )
