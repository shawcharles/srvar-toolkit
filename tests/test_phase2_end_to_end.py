import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_phase2_fit_forecast_shapes_and_reproducibility() -> None:
    rng = np.random.default_rng(123)

    # Simulate a simple stable VAR(1) for testing
    t, n = 60, 2
    beta = np.array(
        [
            [0.0, 0.0],
            [0.6, 0.0],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    sigma = np.eye(n) * 0.1

    y = np.zeros((t, n), dtype=float)
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        y[i] = x @ beta + rng.multivariate_normal(mean=np.zeros(n), cov=sigma)

    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=500, burn_in=100, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    rng_fc1 = np.random.default_rng(2024)
    fc1 = forecast(fit_res, horizons=[1, 3], draws=50, rng=rng_fc1)

    rng_fc2 = np.random.default_rng(2024)
    fc2 = forecast(fit_res, horizons=[1, 3], draws=50, rng=rng_fc2)

    assert fc1.draws.shape == (50, 3, 2)
    assert fc1.mean.shape == (3, 2)
    assert np.all(np.isfinite(fc1.draws))

    # same seed should reproduce
    assert np.allclose(fc1.mean, fc2.mean)
