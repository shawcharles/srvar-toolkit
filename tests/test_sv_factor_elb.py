import numpy as np

from srvar import Dataset, ElbSpec, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_factor_sv_with_elb_fit_forecast_runs_and_returns_latent_series() -> None:
    rng = np.random.default_rng(123)
    t = 50
    bound = 0.0

    # Construct an observed rate series censored at the ELB (floor at bound).
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
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=60, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))
    assert fit_res.latent_dataset is not None
    assert fit_res.latent_draws is not None

    # Shadow-rate draws should only apply to the censored observations (at bound).
    r_idx = ds.variables.index("r")
    mask = ds.values[:, r_idx] <= bound + 1e-12
    assert np.all(fit_res.latent_dataset.values[mask, r_idx] <= bound + 1e-10)
    assert np.allclose(
        fit_res.latent_dataset.values[~mask, r_idx],
        ds.values[~mask, r_idx],
        atol=0.0,
        rtol=0.0,
    )

    # Factor SV state should be present.
    assert fit_res.lambda_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.h_factor_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.sigma_eta2_factor_draws is not None

    # Forecast should return floored (observed) draws and also the latent draws.
    fc = forecast(fit_res, horizons=[1, 2], draws=50, rng=np.random.default_rng(789))
    assert fc.latent_draws is not None
    assert fc.draws.shape == (50, 2, ds.N)
    assert fc.latent_draws.shape == (50, 2, ds.N)
    assert np.all(fc.draws[..., r_idx] >= bound - 1e-12)
