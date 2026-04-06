import numpy as np
import pytest

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.config import ConfigError
from srvar.elb import ElbSpec
from srvar.runner import build_prior
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig, SteadyStateSpec


def _toy_var_dataset(*, t: int = 70, seed: int = 123) -> Dataset:
    rng = np.random.default_rng(seed)
    beta = np.array(
        [
            [0.0, 0.0],
            [0.6, 0.0],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    sigma = np.diag([0.08, 0.1])

    y = np.zeros((t, 2), dtype=float)
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        y[i] = x @ beta + rng.multivariate_normal(mean=np.zeros(2), cov=sigma)

    return Dataset.from_arrays(values=y, variables=["y1", "y2"])


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


def test_minnesota_legacy_alias_matches_historical_constructor() -> None:
    rng = np.random.default_rng(321)
    y = rng.standard_normal((60, 3))

    kwargs = {
        "p": 2,
        "y": y,
        "include_intercept": True,
        "lambda1": 0.2,
        "lambda2": 0.5,
        "lambda3": 1.0,
        "lambda4": 10.0,
        "own_lag_mean": 1.0,
    }
    prior_alias = PriorSpec.niw_minnesota(**kwargs)
    prior_legacy = PriorSpec.niw_minnesota_legacy(**kwargs)

    assert np.allclose(prior_alias.niw.m0, prior_legacy.niw.m0)
    assert np.allclose(prior_alias.niw.v0, prior_legacy.niw.v0)
    assert np.allclose(prior_alias.niw.s0, prior_legacy.niw.s0)
    assert prior_alias.niw.nu0 == prior_legacy.niw.nu0


def test_runner_build_prior_accepts_explicit_minnesota_legacy_method() -> None:
    rng = np.random.default_rng(99)
    ds = Dataset.from_arrays(values=rng.standard_normal((50, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=2, include_intercept=True)

    cfg = {
        "prior": {
            "family": "niw",
            "method": "minnesota_legacy",
            "minnesota": {
                "lambda1": 0.2,
                "lambda2": 0.5,
                "lambda3": 1.0,
                "lambda4": 10.0,
                "own_lag_mean": 1.0,
            },
        }
    }

    prior_from_cfg = build_prior(cfg, dataset=ds, model=model)
    prior_direct = PriorSpec.niw_minnesota_legacy(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        own_lag_mean=1.0,
    )

    assert np.allclose(prior_from_cfg.niw.m0, prior_direct.niw.m0)
    assert np.allclose(prior_from_cfg.niw.v0, prior_direct.niw.v0)
    assert np.allclose(prior_from_cfg.niw.s0, prior_direct.niw.s0)
    assert prior_from_cfg.niw.nu0 == prior_direct.niw.nu0


def test_canonical_minnesota_constructor_preserves_equation_specific_variances() -> None:
    ds = _toy_var_dataset(t=80, seed=7)

    lambda1 = 0.2
    lambda2 = 0.5
    lambda3 = 1.0
    lambda4 = 10.0
    prior = PriorSpec.niw_minnesota_canonical(
        p=2,
        y=ds.values,
        include_intercept=True,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lambda4=lambda4,
        own_lag_mean=1.0,
    )

    assert prior.minnesota_canonical is not None
    sigma2 = prior.minnesota_canonical.sigma2
    k = 1 + ds.N * 2
    variances = (
        1.0 / prior.minnesota_canonical.inv_v0_vec.reshape((k, ds.N), order="F")
    )

    assert np.allclose(variances[0, :], (lambda1 * lambda4) ** 2 * sigma2)
    assert np.isclose(variances[1, 0], lambda1**2)
    assert np.isclose(variances[2, 1], lambda1**2)
    assert np.isclose(variances[2, 0], lambda1**2 * lambda2**2 * sigma2[0] / sigma2[1])
    assert np.isclose(variances[1, 1], lambda1**2 * lambda2**2 * sigma2[1] / sigma2[0])
    assert np.isclose(variances[3, 0], lambda1**2 / (2.0 ** (2.0 * lambda3)))
    assert np.isclose(variances[4, 1], lambda1**2 / (2.0 ** (2.0 * lambda3)))


def test_runner_build_prior_accepts_minnesota_canonical_method() -> None:
    ds = _toy_var_dataset(t=75, seed=8)
    model = ModelSpec(p=2, include_intercept=True)

    cfg = {
        "prior": {
            "family": "niw",
            "method": "minnesota_canonical",
            "minnesota": {
                "lambda1": 0.2,
                "lambda2": 0.5,
                "lambda3": 1.0,
                "lambda4": 10.0,
                "own_lag_mean": 1.0,
            },
        }
    }

    prior_from_cfg = build_prior(cfg, dataset=ds, model=model)
    prior_direct = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        own_lag_mean=1.0,
    )

    assert prior_from_cfg.minnesota_canonical is not None
    assert prior_direct.minnesota_canonical is not None
    assert np.allclose(prior_from_cfg.niw.m0, prior_direct.niw.m0)
    assert np.allclose(prior_from_cfg.niw.v0, prior_direct.niw.v0)
    assert np.allclose(
        prior_from_cfg.minnesota_canonical.inv_v0_vec,
        prior_direct.minnesota_canonical.inv_v0_vec,
    )


def test_tempered_minnesota_constructor_interpolates_legacy_and_canonical() -> None:
    ds = _toy_var_dataset(t=80, seed=77)
    alpha = 0.25
    legacy = PriorSpec.niw_minnesota_legacy(
        p=2,
        y=ds.values,
        include_intercept=True,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        own_lag_mean=1.0,
    )
    canonical = PriorSpec.niw_minnesota_canonical(
        p=2,
        y=ds.values,
        include_intercept=True,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        own_lag_mean=1.0,
    )
    tempered = PriorSpec.niw_minnesota_tempered(
        p=2,
        y=ds.values,
        include_intercept=True,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        alpha=alpha,
        own_lag_mean=1.0,
    )

    assert tempered.minnesota_canonical is not None
    assert tempered.minnesota_canonical.mode == "tempered"
    assert tempered.minnesota_canonical.tempered_alpha == alpha

    k = 1 + ds.N * 2
    legacy_var = np.repeat(np.diag(legacy.niw.v0).reshape(-1, 1), repeats=ds.N, axis=1)
    canonical_var = 1.0 / canonical.minnesota_canonical.inv_v0_vec.reshape((k, ds.N), order="F")
    tempered_var = 1.0 / tempered.minnesota_canonical.inv_v0_vec.reshape((k, ds.N), order="F")

    assert np.allclose(tempered_var, legacy_var * np.power(canonical_var / legacy_var, alpha))
    assert np.allclose(tempered.niw.v0, np.diag(np.mean(tempered_var, axis=1)))


def test_runner_build_prior_accepts_minnesota_tempered_method_for_diagonal_sv() -> None:
    ds = _toy_var_dataset(t=75, seed=18)
    model = ModelSpec(
        p=2,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="diagonal"),
    )

    cfg = {
        "prior": {
            "family": "niw",
            "method": "minnesota_tempered",
            "minnesota": {
                "lambda1": 0.2,
                "lambda2": 0.5,
                "lambda3": 1.0,
                "lambda4": 10.0,
                "own_lag_mean": 1.0,
                "tempered_alpha": 0.25,
            },
        }
    }

    prior_from_cfg = build_prior(cfg, dataset=ds, model=model)
    prior_direct = PriorSpec.niw_minnesota_tempered(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        alpha=0.25,
        own_lag_mean=1.0,
    )

    assert prior_from_cfg.minnesota_canonical is not None
    assert prior_from_cfg.minnesota_canonical.mode == "tempered"
    assert prior_direct.minnesota_canonical is not None
    assert np.allclose(prior_from_cfg.niw.m0, prior_direct.niw.m0)
    assert np.allclose(prior_from_cfg.niw.v0, prior_direct.niw.v0)
    assert np.allclose(
        prior_from_cfg.minnesota_canonical.inv_v0_vec,
        prior_direct.minnesota_canonical.inv_v0_vec,
    )


def test_minnesota_canonical_rejects_unsupported_sv_covariance_in_config_and_api() -> None:
    ds = _toy_var_dataset(t=70, seed=9)
    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="triangular"),
    )
    cfg = {
        "prior": {
            "family": "niw",
            "method": "minnesota_canonical",
            "minnesota": {
                "lambda1": 0.2,
                "lambda2": 0.5,
                "lambda3": 1.0,
                "lambda4": 10.0,
            },
        }
    }

    with pytest.raises(ConfigError, match="currently supports only"):
        build_prior(cfg, dataset=ds, model=model)

    prior = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    sampler = SamplerConfig(draws=20, burn_in=5, thin=1)

    with pytest.raises(ValueError, match="supports only homoskedastic models"):
        fit(ds, model, prior, sampler, rng=np.random.default_rng(10))


def test_minnesota_tempered_rejects_unsupported_models_in_config_and_api() -> None:
    ds = _toy_var_dataset(t=70, seed=19)
    model = ModelSpec(p=1, include_intercept=True)
    cfg = {
        "prior": {
            "family": "niw",
            "method": "minnesota_tempered",
            "minnesota": {
                "lambda1": 0.2,
                "lambda2": 0.5,
                "lambda3": 1.0,
                "lambda4": 10.0,
                "tempered_alpha": 0.25,
            },
        }
    }

    with pytest.raises(ConfigError, match="supports only diagonal stochastic volatility"):
        build_prior(cfg, dataset=ds, model=model)

    prior = PriorSpec.niw_minnesota_tempered(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        alpha=0.25,
    )
    sampler = SamplerConfig(draws=20, burn_in=5, thin=1)

    with pytest.raises(ValueError, match="supports only diagonal stochastic volatility"):
        fit(ds, model, prior, sampler, rng=np.random.default_rng(20))


def test_canonical_minnesota_homoskedastic_fit_forecast_runs() -> None:
    ds = _toy_var_dataset(t=80, seed=10)
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds.values,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    sampler = SamplerConfig(draws=80, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(11))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
    assert fit_res.beta_draws.shape[1:] == (1 + ds.N * model.p, ds.N)
    assert fit_res.sigma_draws.shape[1:] == (ds.N, ds.N)
    assert np.all(np.isfinite(fit_res.beta_draws))
    assert np.all(np.isfinite(fit_res.sigma_draws))

    fc = forecast(fit_res, horizons=[1, 3], draws=30, rng=np.random.default_rng(12))
    assert fc.draws.shape == (30, 3, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_canonical_minnesota_elb_fit_forecast_runs() -> None:
    ds = _toy_var_dataset(t=85, seed=12)
    elb_bound = -0.05
    y = ds.values.copy()
    y[:, 0] = np.minimum(y[:, 0], elb_bound)
    ds_elb = Dataset.from_arrays(values=y, variables=ds.variables)

    model = ModelSpec(
        p=1,
        include_intercept=True,
        elb=ElbSpec(bound=elb_bound, applies_to=["y1"]),
    )
    prior = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds_elb.values,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    sampler = SamplerConfig(draws=70, burn_in=10, thin=2)

    fit_res = fit(ds_elb, model, prior, sampler, rng=np.random.default_rng(13))

    assert fit_res.posterior is None
    assert fit_res.latent_draws is not None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(14))
    assert fc.draws.shape == (20, 2, ds_elb.N)
    assert np.all(fc.draws[:, :, 0] >= elb_bound - 1e-12)
    assert fc.latent_draws is not None


def test_canonical_minnesota_steady_state_fit_runs() -> None:
    ds = _toy_var_dataset(t=80, seed=13)
    model = ModelSpec(
        p=1,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds.N), v0_mu=0.1),
    )
    prior = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds.values,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    sampler = SamplerConfig(draws=70, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(14))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
    assert fit_res.mu_draws is not None
    assert fit_res.mu_draws.shape[0] == fit_res.beta_draws.shape[0]
    assert np.all(np.isfinite(fit_res.mu_draws))


def test_canonical_minnesota_diagonal_sv_fit_forecast_runs() -> None:
    ds = _toy_var_dataset(t=85, seed=14)
    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="diagonal"),
    )
    prior = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds.values,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    sampler = SamplerConfig(draws=80, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(15))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert np.all(np.isfinite(fit_res.beta_draws))
    assert np.all(np.isfinite(fit_res.h_draws))

    fc = forecast(fit_res, horizons=[1, 3], draws=20, rng=np.random.default_rng(16))
    assert fc.draws.shape == (20, 3, ds.N)
    assert np.all(np.isfinite(fc.draws))


def test_tempered_minnesota_diagonal_sv_fit_forecast_runs() -> None:
    ds = _toy_var_dataset(t=85, seed=20)
    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="diagonal"),
    )
    prior = PriorSpec.niw_minnesota_tempered(
        p=model.p,
        y=ds.values,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
        alpha=0.25,
    )
    sampler = SamplerConfig(draws=80, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(21))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert np.all(np.isfinite(fit_res.beta_draws))
    assert np.all(np.isfinite(fit_res.h_draws))

    fc = forecast(fit_res, horizons=[1, 3], draws=20, rng=np.random.default_rng(22))
    assert fc.draws.shape == (20, 3, ds.N)
    assert np.all(np.isfinite(fc.draws))
