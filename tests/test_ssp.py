import numpy as np

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, MuSSVSSpec, PriorSpec, SamplerConfig, SteadyStateSpec


def _toy_dataset(*, t: int = 70, n: int = 2, seed: int = 123) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((t, n))
    return Dataset.from_arrays(values=y, variables=[f"y{i+1}" for i in range(n)])


def test_ssp_fit_produces_mu_draws_and_forecast_shapes() -> None:
    ds = _toy_dataset(t=70, n=2, seed=1)

    model = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds.N), v0_mu=0.1),
    )
    k = (1 if model.include_intercept else 0) + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=80, burn_in=20, thin=2)

    rng = np.random.default_rng(999)
    fit_res = fit(ds, model, prior, sampler, rng=rng)

    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
    assert fit_res.mu_draws is not None
    assert fit_res.mu_draws.shape[0] == fit_res.beta_draws.shape[0]
    assert fit_res.mu_draws.shape[1] == ds.N
    assert fit_res.beta_draws.shape[1] == k

    fc = forecast(fit_res, horizons=[1, 3], draws=30, rng=np.random.default_rng(2024))
    assert fc.draws.shape == (30, 3, ds.N)


def test_ssp_mu_ssvs_produces_mu_gamma_draws() -> None:
    ds = _toy_dataset(t=70, n=3, seed=2)

    model = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(
            mu0=np.zeros(ds.N),
            v0_mu=0.1,
            ssvs=MuSSVSSpec(spike_var=1e-4, slab_var=0.01, inclusion_prob=0.5),
        ),
    )
    k = 1 + ds.N * model.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=80, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.mu_draws is not None
    assert fit_res.mu_gamma_draws is not None
    assert fit_res.mu_gamma_draws.shape == fit_res.mu_draws.shape
    assert fit_res.mu_gamma_draws.dtype == bool


def test_ssp_elb_and_sv_paths_store_mu_draws() -> None:
    rng = np.random.default_rng(4)
    t, n = 90, 2
    y = rng.standard_normal((t, n))
    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    model_elb = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds.N), v0_mu=0.1),
        elb=ElbSpec(bound=elb_bound, applies_to=["r"]),
    )
    k = 1 + ds.N * model_elb.p
    prior = PriorSpec.niw_default(k=k, n=ds.N)
    sampler = SamplerConfig(draws=60, burn_in=10, thin=2)

    fit_res_elb = fit(ds, model_elb, prior, sampler, rng=np.random.default_rng(999))
    assert fit_res_elb.latent_draws is not None
    assert fit_res_elb.mu_draws is not None

    model_sv = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds.N), v0_mu=0.1),
        volatility=VolatilitySpec(),
    )
    fit_res_sv = fit(ds, model_sv, prior, sampler, rng=np.random.default_rng(999))
    assert fit_res_sv.h_draws is not None
    assert fit_res_sv.mu_draws is not None
