import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_phase3_elb_fit_and_forecast_respects_floor() -> None:
    rng = np.random.default_rng(123)

    t, n = 80, 2

    # Latent process
    latent = np.zeros((t, n), dtype=float)
    for i in range(1, t):
        latent[i, 0] = 0.8 * latent[i - 1, 0] + rng.normal(0.0, 0.4)
        latent[i, 1] = 0.5 * latent[i - 1, 1] + rng.normal(0.0, 0.4)

    bound = 0.0
    observed = latent.copy()
    observed[:, 0] = np.maximum(observed[:, 0], bound)

    ds = Dataset.from_arrays(values=observed, variables=["RATE", "Y"])

    model = ModelSpec(p=1, include_intercept=True, elb=ElbSpec(bound=bound, applies_to=["RATE"]))
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=200, burn_in=50, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.latent_dataset is not None
    assert fit_res.latent_draws is not None
    assert fit_res.latent_draws.ndim == 3
    assert fit_res.latent_draws.shape[1:] == (t, n)

    # At bound observations, latent should be <= bound
    rate_obs = ds.values[:, 0]
    rate_lat = fit_res.latent_dataset.values[:, 0]
    mask = np.isclose(rate_obs, bound)
    assert np.all(rate_lat[mask] <= bound + 1e-12)

    fc = forecast(fit_res, horizons=[1, 3], draws=50, rng=np.random.default_rng(2024))

    # Forecast draws returned are observed (ELB floor applied)
    assert np.all(fc.draws[:, :, 0] >= bound - 1e-12)
    assert fc.latent_draws is not None
