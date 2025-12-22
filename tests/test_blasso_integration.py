import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import VolatilitySpec


def test_blasso_elb_fit_and_forecast_respects_floor() -> None:
    rng = np.random.default_rng(10)

    t, n = 90, 2
    y = rng.standard_normal((t, n))
    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)

    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    model = ModelSpec(p=1, include_intercept=True, elb=ElbSpec(bound=elb_bound, applies_to=["r"]))
    k = 1 + n * model.p
    prior = PriorSpec.from_blasso(k=k, n=n, include_intercept=True, mode="global")
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.posterior is not None
    assert fit_res.latent_dataset is not None
    assert fit_res.latent_draws is not None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None

    fc = forecast(fit_res, horizons=[1, 5], draws=40, rng=np.random.default_rng(2025))
    assert fc.draws.shape == (40, 5, 2)
    assert np.all(fc.draws[:, :, 0] >= elb_bound - 1e-12)
    assert fc.latent_draws is not None


def test_blasso_sv_fit_and_forecast_shapes() -> None:
    rng = np.random.default_rng(11)

    t, n = 70, 2
    y = rng.standard_normal((t, n))
    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=1, include_intercept=True, volatility=VolatilitySpec())
    k = 1 + n * model.p
    prior = PriorSpec.from_blasso(k=k, n=n, include_intercept=True, mode="adaptive")
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(111))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None

    fc = forecast(fit_res, horizons=[1, 4], draws=30, rng=np.random.default_rng(2026))
    assert fc.draws.shape == (30, 4, 2)
    assert np.all(np.isfinite(fc.draws))
