import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import (
    VolatilitySpec,
    log_e2_star,
    sample_h0,
    sample_h_svrw,
    sample_mixture_indicators,
    sample_sigma_eta2,
)


def test_sample_mixture_indicators_reproducible() -> None:
    y_star = np.linspace(-2.0, 2.0, 25)
    h = np.linspace(-1.0, 1.0, 25)

    rng1 = np.random.default_rng(123)
    s1 = sample_mixture_indicators(y_star=y_star, h=h, rng=rng1)

    rng2 = np.random.default_rng(123)
    s2 = sample_mixture_indicators(y_star=y_star, h=h, rng=rng2)

    assert s1.shape == (25,)
    assert np.array_equal(s1, s2)
    assert int(s1.min()) >= 0
    assert int(s1.max()) <= 6


def test_sample_h_svrw_reproducible_and_finite() -> None:
    rng = np.random.default_rng(999)

    t = 40
    e = rng.normal(size=t)
    y_star = log_e2_star(e, epsilon=1e-4)
    h_init = np.zeros(t, dtype=float)

    sigma_eta2 = 0.05
    h0 = 0.0

    rng1 = np.random.default_rng(2024)
    h1 = sample_h_svrw(y_star=y_star, h=h_init, sigma_eta2=sigma_eta2, h0=h0, rng=rng1)

    rng2 = np.random.default_rng(2024)
    h2 = sample_h_svrw(y_star=y_star, h=h_init, sigma_eta2=sigma_eta2, h0=h0, rng=rng2)

    assert h1.shape == (t,)
    assert np.all(np.isfinite(h1))
    assert np.allclose(h1, h2)


def test_sample_h0_and_sigma_eta2_are_positive_and_reproducible() -> None:
    rng = np.random.default_rng(7)

    t = 50
    h = rng.normal(scale=0.2, size=t)

    rng1 = np.random.default_rng(111)
    h0_1 = sample_h0(h1=float(h[0]), sigma_eta2=0.1, prior_mean=1e-6, prior_var=10.0, rng=rng1)

    rng2 = np.random.default_rng(111)
    h0_2 = sample_h0(h1=float(h[0]), sigma_eta2=0.1, prior_mean=1e-6, prior_var=10.0, rng=rng2)

    assert h0_1 == h0_2

    rng3 = np.random.default_rng(222)
    s1 = sample_sigma_eta2(h=h, h0=h0_1, nu0=1.0, s0=0.01, rng=rng3)

    rng4 = np.random.default_rng(222)
    s2 = sample_sigma_eta2(h=h, h0=h0_1, nu0=1.0, s0=0.01, rng=rng4)

    assert s1 == s2
    assert s1 > 0.0


def test_phase4_sv_fit_and_forecast_shapes_and_finiteness() -> None:
    rng = np.random.default_rng(123)

    t, n = 70, 2
    beta = np.array(
        [
            [0.0, 0.0],
            [0.6, 0.0],
            [0.0, 0.4],
        ],
        dtype=float,
    )

    y = np.zeros((t, n), dtype=float)
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        eps = rng.normal(size=n) * 0.3
        y[i] = x @ beta + eps

    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=1, include_intercept=True, volatility=VolatilitySpec())
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.latent_draws is None
    assert fit_res.beta_draws.shape[1:] == (1 + n * model.p, n)
    assert fit_res.h_draws.shape[1:] == (t - model.p, n)
    assert fit_res.sigma_eta2_draws.shape[1:] == (n,)
    assert np.all(np.isfinite(fit_res.h_draws))

    fc = forecast(fit_res, horizons=[1, 4], draws=40, rng=np.random.default_rng(2024))
    assert fc.draws.shape == (40, 4, 2)
    assert np.all(np.isfinite(fc.draws))
    assert fc.latent_draws is None


def test_phase4_sv_elb_fit_and_forecast_respects_floor() -> None:
    rng = np.random.default_rng(321)

    t, n = 80, 2
    beta = np.array(
        [
            [0.0, 0.0],
            [0.7, 0.0],
            [0.0, 0.4],
        ],
        dtype=float,
    )

    y = np.zeros((t, n), dtype=float)
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        eps = rng.normal(size=n) * 0.25
        y[i] = x @ beta + eps

    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)

    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    elb = ElbSpec(bound=elb_bound, applies_to=["r"], tol=1e-8)
    model = ModelSpec(p=1, include_intercept=True, elb=elb, volatility=VolatilitySpec())
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(111))
    assert fit_res.latent_dataset is not None
    assert fit_res.latent_draws is not None

    fc = forecast(fit_res, horizons=[1, 5], draws=40, rng=np.random.default_rng(2025))
    assert fc.draws.shape == (40, 5, 2)
    assert np.all(fc.draws[:, :, 0] >= elb_bound - 1e-12)
    assert fc.latent_draws is not None
