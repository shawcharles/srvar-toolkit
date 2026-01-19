from __future__ import annotations

import numpy as np

from .bvar import sample_posterior_niw, simulate_var_forecast
from .data.dataset import Dataset
from .elb import apply_elb_floor
from .results import FitResult, ForecastResult
from .samplers import _fit_elb_gibbs, _fit_fsv, _fit_no_elb, _fit_svcov, _fit_svrw
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

    if model.shocks is not None and model.shocks.family != "gaussian":
        vol = model.volatility
        vol_factor = vol is not None and vol.enabled and vol.covariance == "factor"
        if model.elb is not None and model.elb.enabled and not vol_factor:
            raise ValueError(
                "robust shocks with ELB require factor SV (volatility.covariance='factor')"
            )
        if model.steady_state is not None and not vol_factor:
            raise ValueError(
                "robust shocks with steady_state require factor SV (volatility.covariance='factor')"
            )
        if vol is not None and vol.enabled and not vol_factor:
            raise ValueError(
                "robust shocks are currently supported only with factor SV "
                "(volatility.covariance='factor')"
            )

    if model.volatility is not None and model.volatility.enabled:
        if prior_family not in {"niw", "blasso", "dl"}:
            raise ValueError(
                "stochastic volatility currently requires prior.family in {'niw','blasso','dl'}"
            )
        if model.volatility.covariance == "triangular":
            if prior_family != "niw":
                raise ValueError("triangular SV covariance currently requires prior.family='niw'")
            return _fit_svcov(dataset=dataset, model=model, prior=prior, sampler=sampler, rng=rng)
        if model.volatility.covariance == "factor":
            if prior_family != "niw":
                raise ValueError("FSV currently requires prior.family='niw'")
            return _fit_fsv(dataset=dataset, model=model, prior=prior, sampler=sampler, rng=rng)
        return _fit_svrw(dataset=dataset, model=model, prior=prior, sampler=sampler, rng=rng)

    if model.elb is None or not model.elb.enabled:
        return _fit_no_elb(
            dataset=dataset,
            model=model,
            prior=prior,
            sampler=sampler,
            prior_family=prior_family,
            rng=rng,
        )

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
    stationarity: str = "allow",
    stationarity_tol: float = 1e-10,
    stationarity_max_draws: int | None = None,
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
    stationarity:
        Stationarity policy for VAR coefficient draws used in forecasting:

        - ``"allow"`` (default): do not enforce stability.
        - ``"reject"``: require stable VAR dynamics (companion eigenvalues strictly
          inside the unit circle), rejecting non-stationary coefficient draws.
    stationarity_tol:
        Numerical tolerance for the stability check. A draw is considered stationary if
        ``max(abs(eigvals)) < 1 - stationarity_tol``.
    stationarity_max_draws:
        When `fit` does not contain stored posterior draws (i.e., forecasting samples
        from a conjugate NIW posterior), this caps the **total number of candidate
        posterior draws** attempted to collect `draws` stationary draws under
        ``stationarity="reject"``. If not provided, defaults to ``50 * draws``.

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

    if not isinstance(stationarity, str) or not stationarity:
        raise ValueError("stationarity must be a non-empty string")
    stationarity_l = stationarity.lower()
    if stationarity_l not in {"allow", "reject"}:
        raise ValueError("stationarity must be one of: allow, reject")
    if not np.isfinite(float(stationarity_tol)) or float(stationarity_tol) < 0:
        raise ValueError("stationarity_tol must be finite and >= 0")
    stationarity_tol = float(stationarity_tol)

    if stationarity_max_draws is not None:
        if not isinstance(stationarity_max_draws, (int, np.integer)) or isinstance(
            stationarity_max_draws, bool
        ):
            raise ValueError("stationarity_max_draws must be an integer when provided")
        if int(stationarity_max_draws) < 1:
            raise ValueError("stationarity_max_draws must be >= 1")
        stationarity_max_draws = int(stationarity_max_draws)

    if fit.posterior is None and fit.beta_draws is None:
        raise ValueError(
            "fit does not contain posterior parameters or stored draws; "
            "this can happen if burn_in/thin leaves zero kept draws"
        )

    if fit.model.steady_state is not None and fit.beta_draws is None:
        raise ValueError(
            "steady_state forecasting requires stored beta_draws; reduce burn_in or thin"
        )

    base_dataset = fit.latent_dataset if fit.latent_dataset is not None else fit.dataset
    if base_dataset.T < p:
        raise ValueError("dataset is too short for requested lag order p")
    y_last = base_dataset.values[-p:, :]

    if fit.beta_draws is not None and fit.sigma_draws is not None:
        # sample with replacement from stored posterior draws
        idx: np.ndarray
        if stationarity_l == "reject":
            from .var import is_stationary

            stable = [
                is_stationary(
                    fit.beta_draws[i],
                    n=fit.dataset.N,
                    p=p,
                    include_intercept=fit.model.include_intercept,
                    tol=stationarity_tol,
                )
                for i in range(int(fit.beta_draws.shape[0]))
            ]
            stable_idx = np.flatnonzero(np.asarray(stable, dtype=bool))
            if stable_idx.size < 1:
                raise ValueError("no stationary coefficient draws available in fit.beta_draws")
            sel = rng.integers(0, stable_idx.size, size=draws)
            idx = stable_idx[sel]
        else:
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
                shocks=fit.model.shocks,
            )
    elif (
        fit.beta_draws is not None
        and fit.h_draws is not None
        and fit.sigma_eta2_draws is not None
        and fit.q_draws is not None
    ):
        import scipy.linalg

        idx: np.ndarray
        if stationarity_l == "reject":
            from .var import is_stationary

            stable = [
                is_stationary(
                    fit.beta_draws[i],
                    n=fit.dataset.N,
                    p=p,
                    include_intercept=fit.model.include_intercept,
                    tol=stationarity_tol,
                )
                for i in range(int(fit.beta_draws.shape[0]))
            ]
            stable_idx = np.flatnonzero(np.asarray(stable, dtype=bool))
            if stable_idx.size < 1:
                raise ValueError("no stationary coefficient draws available in fit.beta_draws")
            sel = rng.integers(0, stable_idx.size, size=draws)
            idx = stable_idx[sel]
        else:
            idx = rng.integers(0, fit.beta_draws.shape[0], size=draws)
        beta_draws = fit.beta_draws[idx]
        h_draws = fit.h_draws[idx]
        sigma_eta2_draws = fit.sigma_eta2_draws[idx]
        q_draws = fit.q_draws[idx]
        sv_gamma0_draws = fit.sv_gamma0_draws[idx] if fit.sv_gamma0_draws is not None else None
        sv_phi_draws = fit.sv_phi_draws[idx] if fit.sv_phi_draws is not None else None

        sims = np.empty((draws, hmax, fit.dataset.N), dtype=float)
        for d in range(draws):
            lags = y_last.copy()
            h_curr = h_draws[d, -1, :].copy()
            sig_eta = sigma_eta2_draws[d].copy()
            q = q_draws[d]
            gamma0 = sv_gamma0_draws[d] if sv_gamma0_draws is not None else None
            phi = sv_phi_draws[d] if sv_phi_draws is not None else None
            path = np.empty((hmax, fit.dataset.N), dtype=float)

            for h_step in range(hmax):
                x_parts = []
                if fit.model.include_intercept:
                    x_parts.append(np.array([1.0], dtype=float))
                for lag in range(1, p + 1):
                    x_parts.append(lags[-lag, :])
                x_row = np.concatenate(x_parts)

                mean = x_row @ beta_draws[d]
                z = rng.normal(size=fit.dataset.N)
                eps = z * np.exp(0.5 * h_curr)
                innov = scipy.linalg.solve_triangular(q, eps, lower=False, check_finite=False)
                y_next = mean + innov

                path[h_step] = y_next
                lags = np.vstack([lags[1:, :], y_next]) if p > 1 else y_next.reshape(1, -1)
                if gamma0 is not None and phi is not None:
                    h_curr = (
                        gamma0 + phi * h_curr + np.sqrt(sig_eta) * rng.normal(size=fit.dataset.N)
                    )
                else:
                    h_curr = h_curr + np.sqrt(sig_eta) * rng.normal(size=fit.dataset.N)

            sims[d] = path
    elif (
        fit.beta_draws is not None
        and fit.h_draws is not None
        and fit.sigma_eta2_draws is not None
        and fit.lambda_draws is not None
        and fit.h_factor_draws is not None
        and fit.sigma_eta2_factor_draws is not None
    ):
        idx: np.ndarray
        if stationarity_l == "reject":
            from .var import is_stationary

            stable = [
                is_stationary(
                    fit.beta_draws[i],
                    n=fit.dataset.N,
                    p=p,
                    include_intercept=fit.model.include_intercept,
                    tol=stationarity_tol,
                )
                for i in range(int(fit.beta_draws.shape[0]))
            ]
            stable_idx = np.flatnonzero(np.asarray(stable, dtype=bool))
            if stable_idx.size < 1:
                raise ValueError("no stationary coefficient draws available in fit.beta_draws")
            sel = rng.integers(0, stable_idx.size, size=draws)
            idx = stable_idx[sel]
        else:
            idx = rng.integers(0, fit.beta_draws.shape[0], size=draws)

        beta_draws = fit.beta_draws[idx]
        h_eta_draws = fit.h_draws[idx]
        sigma_eta2_eta_draws = fit.sigma_eta2_draws[idx]
        lam_draws = fit.lambda_draws[idx]
        h_f_draws = fit.h_factor_draws[idx]
        sigma_eta2_f_draws = fit.sigma_eta2_factor_draws[idx]

        sims = np.empty((draws, hmax, fit.dataset.N), dtype=float)
        for d in range(draws):
            lags = y_last.copy()
            h_eta_curr = h_eta_draws[d, -1, :].copy()
            sig_eta2_eta = sigma_eta2_eta_draws[d].copy()
            lam = lam_draws[d]

            h_f_curr = h_f_draws[d, -1, :].copy()
            sig_eta2_f = sigma_eta2_f_draws[d].copy()

            k = int(h_f_curr.shape[0])
            path = np.empty((hmax, fit.dataset.N), dtype=float)

            for h_step in range(hmax):
                x_parts = []
                if fit.model.include_intercept:
                    x_parts.append(np.array([1.0], dtype=float))
                for lag in range(1, p + 1):
                    x_parts.append(lags[-lag, :])
                x_row = np.concatenate(x_parts)

                mean = x_row @ beta_draws[d]

                f_step = rng.normal(size=k) * np.exp(0.5 * h_f_curr)
                eta_step = rng.normal(size=fit.dataset.N) * np.exp(0.5 * h_eta_curr)
                eps = lam @ f_step + eta_step

                if fit.model.shocks is not None and fit.model.shocks.family != "gaussian":
                    fam = str(fit.model.shocks.family).lower()
                    if fam == "student_t":
                        nu = float(fit.model.shocks.df)
                        lam_shock = float(rng.gamma(shape=0.5 * nu, scale=2.0 / nu))
                        eps = eps / float(np.sqrt(lam_shock))
                    elif fam == "mixture_outlier":
                        prob = float(fit.model.shocks.outlier_prob)
                        kappa = float(fit.model.shocks.outlier_variance)
                        is_outlier = bool(rng.uniform() < prob)
                        eps = eps * (float(np.sqrt(kappa)) if is_outlier else 1.0)
                    else:  # pragma: no cover
                        raise ValueError(f"unknown shocks.family: {fit.model.shocks.family}")

                y_next = mean + eps
                path[h_step] = y_next

                lags = np.vstack([lags[1:, :], y_next]) if p > 1 else y_next.reshape(1, -1)

                h_eta_curr = h_eta_curr + np.sqrt(sig_eta2_eta) * rng.normal(size=fit.dataset.N)
                h_f_curr = h_f_curr + np.sqrt(sig_eta2_f) * rng.normal(size=k)

            sims[d] = path
    elif (
        fit.beta_draws is not None and fit.h_draws is not None and fit.sigma_eta2_draws is not None
    ):
        idx: np.ndarray
        if stationarity_l == "reject":
            from .var import is_stationary

            stable = [
                is_stationary(
                    fit.beta_draws[i],
                    n=fit.dataset.N,
                    p=p,
                    include_intercept=fit.model.include_intercept,
                    tol=stationarity_tol,
                )
                for i in range(int(fit.beta_draws.shape[0]))
            ]
            stable_idx = np.flatnonzero(np.asarray(stable, dtype=bool))
            if stable_idx.size < 1:
                raise ValueError("no stationary coefficient draws available in fit.beta_draws")
            sel = rng.integers(0, stable_idx.size, size=draws)
            idx = stable_idx[sel]
        else:
            idx = rng.integers(0, fit.beta_draws.shape[0], size=draws)
        beta_draws = fit.beta_draws[idx]
        h_draws = fit.h_draws[idx]
        sigma_eta2_draws = fit.sigma_eta2_draws[idx]
        sv_gamma0_draws = fit.sv_gamma0_draws[idx] if fit.sv_gamma0_draws is not None else None
        sv_phi_draws = fit.sv_phi_draws[idx] if fit.sv_phi_draws is not None else None

        sims = np.empty((draws, hmax, fit.dataset.N), dtype=float)
        for d in range(draws):
            lags = y_last.copy()
            h_curr = h_draws[d, -1, :].copy()
            sig_eta = sigma_eta2_draws[d].copy()
            gamma0 = sv_gamma0_draws[d] if sv_gamma0_draws is not None else None
            phi = sv_phi_draws[d] if sv_phi_draws is not None else None
            path = np.empty((hmax, fit.dataset.N), dtype=float)
            for h_step in range(hmax):
                x_parts = []
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
                if gamma0 is not None and phi is not None:
                    h_curr = (
                        gamma0 + phi * h_curr + np.sqrt(sig_eta) * rng.normal(size=fit.dataset.N)
                    )
                else:
                    h_curr = h_curr + np.sqrt(sig_eta) * rng.normal(size=fit.dataset.N)

            sims[d] = path
    else:
        if fit.posterior is None:
            raise ValueError("fit has no posterior parameters or stored draws")
        if stationarity_l == "reject":
            from .var import is_stationary

            max_total = (
                int(50 * draws) if stationarity_max_draws is None else int(stationarity_max_draws)
            )
            accepted_beta: list[np.ndarray] = []
            accepted_sigma: list[np.ndarray] = []
            attempted = 0
            while len(accepted_beta) < draws and attempted < max_total:
                need = int(draws - len(accepted_beta))
                # oversample to reduce expected loop iterations, but stay within budget
                batch = min(max(10, 2 * need), max_total - attempted)
                beta_b, sigma_b = sample_posterior_niw(
                    mn=fit.posterior.mn,
                    vn=fit.posterior.vn,
                    sn=fit.posterior.sn,
                    nun=fit.posterior.nun,
                    draws=batch,
                    rng=rng,
                )
                attempted += batch
                for i in range(int(batch)):
                    if is_stationary(
                        beta_b[i],
                        n=fit.dataset.N,
                        p=p,
                        include_intercept=fit.model.include_intercept,
                        tol=stationarity_tol,
                    ):
                        accepted_beta.append(beta_b[i])
                        accepted_sigma.append(sigma_b[i])
                        if len(accepted_beta) == draws:
                            break

            if len(accepted_beta) < draws:
                raise ValueError(
                    f"stationarity='reject' could not generate {draws} stationary draws within "
                    f"{max_total} candidate draws; try increasing stationarity_max_draws or relaxing priors"
                )

            beta_draws = np.stack(accepted_beta)
            sigma_draws = np.stack(accepted_sigma)
        else:
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
                shocks=fit.model.shocks,
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
