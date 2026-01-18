import numpy as np

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_sv_ar1_diagonal_runs() -> None:
    ds = Dataset.from_arrays(
        values=np.random.default_rng(123).standard_normal((80, 2)),
        variables=["y1", "y2"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(
            enabled=True,
            dynamics="ar1",
            covariance="diagonal",
            phi_prior_mean=0.95,
            phi_prior_var=0.1,
            gamma0_prior_mean=0.0,
            gamma0_prior_var=10.0,
        ),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=80, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.sv_gamma0_draws is not None
    assert fit_res.sv_phi_draws is not None
    assert fit_res.sv_gamma0_draws.shape == fit_res.sv_phi_draws.shape

    fc = forecast(fit_res, horizons=[1, 3], draws=20, rng=np.random.default_rng(789))
    assert fc.draws.shape == (20, 3, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_sv_ar1_triangular_runs() -> None:
    ds = Dataset.from_arrays(
        values=np.random.default_rng(123).standard_normal((70, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(
            enabled=True,
            dynamics="ar1",
            covariance="triangular",
            q_prior_var=1.0,
            phi_prior_mean=0.95,
            phi_prior_var=0.1,
            gamma0_prior_mean=0.0,
            gamma0_prior_var=10.0,
        ),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=60, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))
    assert fit_res.beta_draws is not None
    assert fit_res.q_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.sv_gamma0_draws is not None
    assert fit_res.sv_phi_draws is not None

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(789))
    assert fc.draws.shape == (20, 2, ds.N)
    assert np.all(np.isfinite(fc.draws))
