import numpy as np
import pytest

from srvar.api import fit, forecast
from srvar.data.dataset import Dataset
from srvar.results import FitResult, ForecastResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_arviz_converters_import_or_convert() -> None:
    ds = Dataset.from_arrays(values=np.zeros((3, 1), dtype=float), variables=["y"])
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=2, n=1)
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)
    fit_res = FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=np.zeros((1, 2, 1), dtype=float),
        sigma_draws=np.ones((1, 1, 1), dtype=float),
    )

    fc = ForecastResult(
        variables=["y"],
        horizons=[1],
        draws=np.zeros((2, 1, 1), dtype=float),
        mean=np.zeros((1, 1), dtype=float),
        quantiles={},
        latent_draws=None,
    )

    from srvar.arviz import fit_to_inferencedata, forecast_to_inferencedata

    try:
        import arviz as az  # type: ignore
    except Exception:
        with pytest.raises(ImportError, match="arviz is required"):
            fit_to_inferencedata(fit_res)
        with pytest.raises(ImportError, match="arviz is required"):
            forecast_to_inferencedata(fc)
        return

    idata_fit = fit_to_inferencedata(fit_res)
    assert isinstance(idata_fit, az.InferenceData)
    assert hasattr(idata_fit, "posterior")
    assert "beta" in idata_fit.posterior.data_vars

    idata_fc = forecast_to_inferencedata(fc)
    assert isinstance(idata_fc, az.InferenceData)
    assert hasattr(idata_fc, "posterior_predictive")
    assert "draws" in idata_fc.posterior_predictive.data_vars


def test_fit_to_inferencedata_shapes_and_groups() -> None:
    pytest.importorskip("arviz")

    from srvar.arviz import fit_to_inferencedata

    rng = np.random.default_rng(0)
    ds = Dataset.from_arrays(values=rng.standard_normal((40, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=8, burn_in=0, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(1))
    idata = fit_to_inferencedata(fit_res)

    assert hasattr(idata, "posterior")
    assert "chain" in idata.posterior.dims
    assert "draw" in idata.posterior.dims
    assert idata.posterior.sizes["chain"] == 1

    assert hasattr(idata, "observed_data")
    assert "y" in idata.observed_data.data_vars
    assert idata.observed_data["y"].dims == ("time", "variable")


def test_forecast_to_inferencedata_includes_posterior_predictive() -> None:
    pytest.importorskip("arviz")

    from srvar.arviz import forecast_to_inferencedata

    rng = np.random.default_rng(10)
    ds = Dataset.from_arrays(values=rng.standard_normal((40, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=8, burn_in=0, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(11))
    fc = forecast(fit_res, horizons=[1, 2], draws=5, rng=np.random.default_rng(12))

    idata = forecast_to_inferencedata(fc)
    assert hasattr(idata, "posterior_predictive")
    assert "chain" in idata.posterior_predictive.dims
    assert "draw" in idata.posterior_predictive.dims
    assert "draws" in idata.posterior_predictive.data_vars
