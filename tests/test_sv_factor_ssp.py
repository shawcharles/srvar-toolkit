import numpy as np

from srvar import Dataset, ElbSpec, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, MuSSVSSpec, PriorSpec, SamplerConfig, SteadyStateSpec


def test_factor_sv_with_steady_state_fit_forecast_runs_and_produces_mu_draws() -> None:
    ds = Dataset.from_arrays(
        values=np.random.default_rng(123).standard_normal((70, 3)),
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
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))

    assert fit_res.beta_draws is not None
    assert fit_res.beta_draws.shape[1] == k
    assert fit_res.mu_draws is not None
    assert fit_res.mu_draws.shape[0] == fit_res.beta_draws.shape[0]
    assert fit_res.mu_draws.shape[1] == ds.N

    assert fit_res.lambda_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.h_factor_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.sigma_eta2_factor_draws is not None

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(789))
    assert fc.draws.shape == (20, 2, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_factor_sv_with_steady_state_mu_ssvs_produces_mu_gamma_draws() -> None:
    ds = Dataset.from_arrays(
        values=np.random.default_rng(111).standard_normal((70, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(
            mu0=np.zeros(ds.N),
            v0_mu=0.1,
            ssvs=MuSSVSSpec(spike_var=1e-4, slab_var=0.01, inclusion_prob=0.5),
        ),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=2,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(222))
    assert fit_res.mu_draws is not None
    assert fit_res.mu_gamma_draws is not None
    assert fit_res.mu_gamma_draws.shape == fit_res.mu_draws.shape
    assert fit_res.mu_gamma_draws.dtype == bool


def test_factor_sv_with_steady_state_and_elb_returns_latent_series() -> None:
    rng = np.random.default_rng(321)
    t = 60
    bound = 0.0

    r_latent = rng.standard_normal(t)
    r_obs = np.maximum(r_latent, bound)
    y2 = rng.standard_normal(t)
    ds = Dataset.from_arrays(values=np.column_stack([r_obs, y2]), variables=["r", "y2"])

    model = ModelSpec(
        p=2,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds.N), v0_mu=0.1),
        elb=ElbSpec(bound=bound, applies_to=["r"]),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=1,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=40, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(654))
    assert fit_res.latent_dataset is not None
    assert fit_res.latent_draws is not None
    assert fit_res.mu_draws is not None

    r_idx = ds.variables.index("r")
    mask = ds.values[:, r_idx] <= bound + 1e-12
    assert np.all(fit_res.latent_dataset.values[mask, r_idx] <= bound + 1e-10)

    fc = forecast(fit_res, horizons=[1, 2], draws=30, rng=np.random.default_rng(987))
    assert fc.latent_draws is not None
    assert np.all(fc.draws[..., r_idx] >= bound - 1e-12)

