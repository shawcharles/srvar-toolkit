import numpy as np

from srvar.rng import inverse_gaussian


def test_inverse_gaussian_positive_and_finite() -> None:
    rng = np.random.default_rng(123)
    x = inverse_gaussian(mu=1.5, lam=2.0, rng=rng)
    assert np.isfinite(x)
    assert x > 0


def test_inverse_gaussian_mean_approx_mu() -> None:
    rng = np.random.default_rng(123)
    mu = 1.2
    lam = 3.0
    draws = inverse_gaussian(mu=np.full(50_000, mu), lam=np.full(50_000, lam), rng=rng)
    m = float(np.mean(draws))
    assert np.isfinite(m)
    assert abs(m - mu) < 0.05
