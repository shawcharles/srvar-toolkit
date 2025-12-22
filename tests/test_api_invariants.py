import numpy as np

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _toy_dataset(*, t: int = 70, n: int = 2, seed: int = 123) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((t, n))
    return Dataset.from_arrays(values=y, variables=[f"y{i+1}" for i in range(n)])


def test_invariants_phase2_bvar() -> None:
    ds = _toy_dataset(t=70, n=2, seed=1)

    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.posterior is not None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
    assert fit_res.h_draws is None
    assert fit_res.sigma_eta2_draws is None

    fc = forecast(fit_res, horizons=[1, 3], draws=30, rng=np.random.default_rng(2024))
    assert fc.draws.shape == (30, 3, ds.N)


def test_invariants_phase3_elb() -> None:
    rng = np.random.default_rng(2)
    t, n = 90, 2
    y = rng.standard_normal((t, n))
    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    model = ModelSpec(p=1, include_intercept=True, elb=ElbSpec(bound=elb_bound, applies_to=["r"]))
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.posterior is not None
    assert fit_res.latent_dataset is not None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
    assert fit_res.h_draws is None

    fc = forecast(fit_res, horizons=[1, 3], draws=30, rng=np.random.default_rng(2024))
    assert np.all(fc.draws[:, :, 0] >= elb_bound - 1e-12)


def test_invariants_phase4_sv() -> None:
    ds = _toy_dataset(t=90, n=2, seed=3)

    model = ModelSpec(p=1, include_intercept=True, volatility=VolatilitySpec())
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is None
    assert fit_res.h_draws is not None
    assert fit_res.h0_draws is not None
    assert fit_res.sigma_eta2_draws is not None

    fc = forecast(fit_res, horizons=[1, 3], draws=30, rng=np.random.default_rng(2024))
    assert fc.draws.shape == (30, 3, ds.N)


def test_invariants_phase4_sv_elb() -> None:
    rng = np.random.default_rng(4)
    t, n = 100, 2
    y = rng.standard_normal((t, n))
    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    model = ModelSpec(
        p=1,
        include_intercept=True,
        elb=ElbSpec(bound=elb_bound, applies_to=["r"]),
        volatility=VolatilitySpec(),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.posterior is None
    assert fit_res.latent_dataset is not None
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None

    fc = forecast(fit_res, horizons=[1, 3], draws=30, rng=np.random.default_rng(2024))
    assert np.all(fc.draws[:, :, 0] >= elb_bound - 1e-12)
