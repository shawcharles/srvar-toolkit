import numpy as np
import pytest

from srvar.api import fit, forecast
from srvar.config import ConfigError, build_model
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig, ShockSpec


def test_fit_student_t_shocks_runs_and_forecasts() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(40, 1))
    ds = Dataset.from_arrays(values=y, variables=["y"])

    model = ModelSpec(p=1, include_intercept=True, shocks=ShockSpec(family="student_t", df=7.0))
    prior = PriorSpec.niw_default(k=2, n=1)
    sampler = SamplerConfig(draws=30, burn_in=0, thin=1)

    res = fit(ds, model, prior, sampler, rng=rng)
    assert res.beta_draws is not None
    assert res.sigma_draws is not None
    assert res.beta_draws.shape == (30, 2, 1)
    assert res.sigma_draws.shape == (30, 1, 1)

    fc = forecast(res, horizons=[1, 2], draws=25, rng=rng)
    assert fc.draws.shape == (25, 2, 1)
    assert np.all(np.isfinite(fc.mean))


def test_fit_outlier_mixture_shocks_runs_and_forecasts() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=(40, 1))
    ds = Dataset.from_arrays(values=y, variables=["y"])

    shocks = ShockSpec(family="mixture_outlier", outlier_prob=0.2, outlier_variance=25.0)
    model = ModelSpec(p=1, include_intercept=True, shocks=shocks)
    prior = PriorSpec.niw_default(k=2, n=1)
    sampler = SamplerConfig(draws=30, burn_in=0, thin=1)

    res = fit(ds, model, prior, sampler, rng=rng)
    assert res.beta_draws is not None
    assert res.sigma_draws is not None
    assert res.beta_draws.shape == (30, 2, 1)
    assert res.sigma_draws.shape == (30, 1, 1)

    fc = forecast(res, horizons=[1], draws=25, rng=rng)
    assert fc.draws.shape == (25, 1, 1)
    assert np.all(np.isfinite(fc.mean))


def test_config_rejects_robust_shocks_with_elb() -> None:
    ds = Dataset.from_arrays(values=np.zeros((10, 1)), variables=["y"])
    cfg = {
        "model": {
            "p": 1,
            "elb": {"enabled": True, "bound": 0.0, "applies_to": ["y"]},
            "shocks": {"family": "student_t", "df": 7.0},
        }
    }
    with pytest.raises(ConfigError, match="robust shocks"):
        build_model(cfg, dataset=ds)
