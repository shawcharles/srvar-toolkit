from __future__ import annotations

import numpy as np

from .bvar import posterior_niw, sample_posterior_niw, simulate_var_forecast
from .data.dataset import Dataset
from .results import FitResult, ForecastResult, PosteriorNIW
from .elb import apply_elb_floor, sample_shadow_value, sample_shadow_value_svrw
from .spec import ModelSpec, PriorSpec, SamplerConfig
from .ssvs import sample_gamma_rows, v0_diag_from_gamma
from .sv import log_e2_star, sample_beta_svrw, sample_h0, sample_h_svrw, sample_sigma_eta2
from .var import design_matrix


def fit(
    dataset: Dataset,
    model: ModelSpec,
    prior: PriorSpec,
    sampler: SamplerConfig,
    *,
    rng: np.random.Generator | None = None,
) -> FitResult:
    prior_family = prior.family.lower()
    if prior_family not in {"niw", "ssvs"}:
        raise ValueError("only prior.family in {'niw','ssvs'} is supported")

    if rng is None:
        rng = np.random.default_rng()

    if model.volatility is not None and model.volatility.enabled:
        if prior_family != "niw":
            raise ValueError("stochastic volatility currently requires prior.family='niw'")
        vol = model.volatility

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
        mn, _vn, _sn, _nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
        beta = mn.copy()

        h0 = np.log(np.var(y, axis=0) + 1e-12)
        h = np.tile(h0.reshape(1, -1), (t_eff, 1))
        sigma_eta2 = 0.05 * np.ones(n, dtype=float)

        beta_keep: list[np.ndarray] = []
        h_keep: list[np.ndarray] = []
        h0_keep: list[np.ndarray] = []
        sigma_eta2_keep: list[np.ndarray] = []

        for it in range(sampler.draws):
            x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

            beta = sample_beta_svrw(x=x, y=y, m0=niw.m0, v0=niw.v0, h=h, rng=rng)

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
                h[:, i] = sample_h_svrw(y_star=y_star, h=h[:, i], sigma_eta2=float(sigma_eta2[i]), h0=float(h0[i]), rng=rng)
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

            if it >= sampler.burn_in and ((it - sampler.burn_in) % sampler.thin == 0):
                beta_keep.append(beta.copy())
                h_keep.append(h.copy())
                h0_keep.append(h0.copy())
                sigma_eta2_keep.append(sigma_eta2.copy())

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
            beta_draws=np.stack(beta_keep) if beta_keep else None,
            sigma_draws=None,
            h_draws=np.stack(h_keep) if h_keep else None,
            h0_draws=np.stack(h0_keep) if h0_keep else None,
            sigma_eta2_draws=np.stack(sigma_eta2_keep) if sigma_eta2_keep else None,
        )

    if model.elb is None or not model.elb.enabled:
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

        if prior.ssvs is None:
            raise ValueError("prior.family='ssvs' requires prior.ssvs")

        spec = prior.ssvs
        t_eff, k = x.shape
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

    # Phase 3: ELB data-augmentation Gibbs
    elb = model.elb

    applies_to_idx: list[int] = []
    for name in elb.applies_to:
        try:
            applies_to_idx.append(dataset.variables.index(name))
        except ValueError as e:
            raise ValueError(f"elb.applies_to contains unknown variable: {name}") from e

    # Initialize latent series: start at observed, but nudge ELB-bound observations slightly below bound
    y_lat = dataset.values.copy()

    elb_t_idx: dict[int, np.ndarray] = {}
    for j in applies_to_idx:
        mask = dataset.values[:, j] <= (elb.bound + elb.tol)
        elb_t_idx[j] = np.where(mask)[0]
        y_lat[mask, j] = elb.bound - 0.05

    beta_keep: list[np.ndarray] = []
    sigma_keep: list[np.ndarray] = []

    niw = prior.niw

    gamma: np.ndarray | None = None
    fixed_mask: np.ndarray | None = None
    spec = prior.ssvs if prior_family == "ssvs" else None

    last_posterior: PosteriorNIW | None = None

    for it in range(sampler.draws):
        x, y = design_matrix(y_lat, model.p, include_intercept=model.include_intercept)

        if prior_family == "niw":
            mn, vn, sn, nun = posterior_niw(x=x, y=y, m0=niw.m0, v0=niw.v0, s0=niw.s0, nu0=niw.nu0)
            last_posterior = PosteriorNIW(mn=mn, vn=vn, sn=sn, nun=nun)

            beta_draws, sigma_draws = sample_posterior_niw(mn=mn, vn=vn, sn=sn, nun=nun, draws=1, rng=rng)
            beta = beta_draws[0]
            sigma = sigma_draws[0]
        else:
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

        # Update latent shadow rates at ELB observations
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
        beta_draws=np.stack(beta_keep) if beta_keep else None,
        sigma_draws=np.stack(sigma_keep) if sigma_keep else None,
    )


def forecast(
    fit: FitResult,
    horizons: list[int],
    *,
    draws: int = 1000,
    quantile_levels: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> ForecastResult:
    if rng is None:
        rng = np.random.default_rng()

    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    hmax = int(max(horizons))
    p = fit.model.p

    base_dataset = fit.latent_dataset if fit.latent_dataset is not None else fit.dataset
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

    # If ELB enabled, return observed draws (apply floor) for constrained variables
    if fit.model.elb is not None and fit.model.elb.enabled:
        applies_to_idx = [fit.dataset.variables.index(v) for v in fit.model.elb.applies_to]
        sims = apply_elb_floor(sims, bound=fit.model.elb.bound, indices=applies_to_idx)
        mean = sims.mean(axis=0)
        quantiles = {q: np.quantile(sims, q=q, axis=0) for q in quantile_levels}

    return ForecastResult(
        variables=list(fit.dataset.variables),
        horizons=list(horizons),
        draws=sims,
        mean=mean,
        quantiles=quantiles,
    )
