import numpy as np

from srvar import Dataset, ElbSpec, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig, ShockSpec, SteadyStateSpec


def test_factor_sv_with_student_t_shocks_fit_forecast_runs() -> None:
    rng = np.random.default_rng(10)
    ds = Dataset.from_arrays(
        values=rng.standard_normal((60, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=2,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
        shocks=ShockSpec(family="student_t", df=7.0),
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(11))
    assert fit_res.beta_draws is not None
    assert fit_res.lambda_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.h_factor_draws is not None

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(12))
    assert fc.draws.shape == (20, 2, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_factor_sv_with_outlier_mixture_shocks_fit_forecast_runs() -> None:
    rng = np.random.default_rng(20)
    ds = Dataset.from_arrays(
        values=rng.standard_normal((60, 3)),
        variables=["y1", "y2", "y3"],
    )

    shocks = ShockSpec(family="mixture_outlier", outlier_prob=0.2, outlier_variance=25.0)
    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=2,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
        shocks=shocks,
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(21))
    fc = forecast(fit_res, horizons=[1], draws=20, rng=np.random.default_rng(22))
    assert fc.draws.shape == (20, 1, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_factor_sv_with_student_t_shocks_and_elb_runs() -> None:
    rng = np.random.default_rng(30)
    t = 60
    bound = 0.0

    r_latent = rng.standard_normal(t)
    r_obs = np.maximum(r_latent, bound)
    y2 = rng.standard_normal(t)
    ds = Dataset.from_arrays(values=np.column_stack([r_obs, y2]), variables=["r", "y2"])

    model = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=bound, applies_to=["r"]),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=1,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
        shocks=ShockSpec(family="student_t", df=7.0),
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(31))
    assert fit_res.latent_dataset is not None

    fc = forecast(fit_res, horizons=[1], draws=20, rng=np.random.default_rng(32))
    assert fc.latent_draws is not None
    assert np.all(np.isfinite(fc.draws))


def test_factor_sv_with_student_t_shocks_and_steady_state_runs() -> None:
    rng = np.random.default_rng(40)
    ds = Dataset.from_arrays(
        values=rng.standard_normal((70, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds.N), v0_mu=0.1),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=2,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
        shocks=ShockSpec(family="student_t", df=7.0),
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(41))
    assert fit_res.mu_draws is not None

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(42))
    assert np.all(np.isfinite(fc.draws))
