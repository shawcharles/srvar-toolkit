from __future__ import annotations

import numpy as np

from .bvar import sample_posterior_niw, simulate_var_forecast
from .data.dataset import Dataset
from .elb import apply_elb_floor
from .results import FitResult, ForecastResult
from .samplers import _fit_elb_gibbs, _fit_no_elb, _fit_svrw
from .spec import ModelSpec, PriorSpec, SamplerConfig


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
    below the bound to a small amount below the bound (``bound - elb.init_offset``). This is
    a numerical initialization choice intended to avoid starting exactly at the truncation
    boundary.
    """
    prior_family = prior.family.lower()
    if prior_family not in {"niw", "ssvs", "blasso", "dl"}:
        raise ValueError("only prior.family in {'niw','ssvs','blasso','dl'} is supported")

    if rng is None:
        rng = np.random.default_rng()

    if model.volatility is not None and model.volatility.enabled:
        if prior_family not in {"niw", "blasso", "dl"}:
            raise ValueError("stochastic volatility currently requires prior.family in {'niw','blasso','dl'}")
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

    if fit.model.steady_state is not None and fit.beta_draws is None:
        raise ValueError("steady_state forecasting requires stored beta_draws; reduce burn_in or thin")

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
