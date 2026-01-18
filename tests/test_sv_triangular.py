import numpy as np

from srvar import Dataset, ElbSpec, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_triangular_sv_fit_forecast_runs() -> None:
    ds = Dataset.from_arrays(
        values=np.random.default_rng(123).standard_normal((60, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="triangular", q_prior_var=1.0),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=60, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.q_draws is not None

    d = fit_res.beta_draws.shape[0]
    assert fit_res.q_draws.shape == (d, ds.N, ds.N)
    assert np.allclose(fit_res.q_draws[:, np.arange(ds.N), np.arange(ds.N)], 1.0)
    assert np.allclose(
        fit_res.q_draws[:, np.tril_indices(ds.N, k=-1)[0], np.tril_indices(ds.N, k=-1)[1]], 0.0
    )

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(789))
    assert fc.draws.shape == (20, 2, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_triangular_sv_with_elb_runs() -> None:
    rng = np.random.default_rng(123)
    y = rng.standard_normal((80, 3))
    bound = 0.25
    y[:, 1] = rng.normal(loc=0.8, scale=0.05, size=y.shape[0])
    y[:12, 1] = bound

    ds = Dataset.from_arrays(values=y, variables=["x", "r", "z"])

    model = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=bound, applies_to=["r"], enabled=True),
        volatility=VolatilitySpec(enabled=True, covariance="triangular", q_prior_var=1.0),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=50, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))
    assert fit_res.latent_draws is not None
    assert fit_res.q_draws is not None

    r_idx = ds.variables.index("r")
    elb_times = np.where(ds.values[:, r_idx] <= bound + model.elb.tol)[0]
    assert elb_times.size > 0
    assert np.all(fit_res.latent_draws[:, elb_times, r_idx] <= bound + 1e-8)
