import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_minnesota_prior_shapes_and_own_lag_mean() -> None:
    rng = np.random.default_rng(123)
    y = rng.standard_normal((80, 3))

    prior = PriorSpec.niw_minnesota(
        p=2,
        y=y,
        include_intercept=True,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        own_lag_mean=1.0,
    )

    k_expected = 1 + 3 * 2
    assert prior.family.lower() == "niw"
    assert prior.niw.m0.shape == (k_expected, 3)
    assert prior.niw.v0.shape == (k_expected, k_expected)
    assert prior.niw.s0.shape == (3, 3)

    assert np.all(np.diag(prior.niw.v0) > 0)
    assert np.all(np.isfinite(prior.niw.v0))

    # own-lag-1 means at positions base + j
    assert float(prior.niw.m0[1, 0]) == 1.0
    assert float(prior.niw.m0[2, 1]) == 1.0
    assert float(prior.niw.m0[3, 2]) == 1.0

    # other entries are zero
    assert np.allclose(prior.niw.m0[0, :], 0.0)


def test_phase2_fit_forecast_with_minnesota_prior_runs() -> None:
    rng = np.random.default_rng(123)

    t, n = 70, 2
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
    prior = PriorSpec.niw_minnesota(
        p=model.p,
        y=ds.values,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        own_lag_mean=0.0,
    )
    sampler = SamplerConfig(draws=200, burn_in=50, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))
    fc = forecast(fit_res, horizons=[1, 3], draws=50, rng=np.random.default_rng(2024))

    assert fc.draws.shape == (50, 3, 2)
    assert np.all(np.isfinite(fc.draws))
