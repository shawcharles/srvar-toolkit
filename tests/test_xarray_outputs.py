import numpy as np
import pytest

from srvar.data.dataset import Dataset
from srvar.results import FitResult, ForecastResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.xarray import fit_to_xarray, forecast_to_xarray


def test_xarray_converters_import_or_convert() -> None:
    ds = Dataset.from_arrays(values=np.zeros((3, 1), dtype=float), variables=["y"])
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=2, n=1)
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)
    fit = FitResult(
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
        quantiles={0.5: np.zeros((1, 1), dtype=float)},
    )

    try:
        import xarray as xr  # type: ignore
    except Exception:
        with pytest.raises(ImportError, match="xarray is required"):
            fit_to_xarray(fit)
        with pytest.raises(ImportError, match="xarray is required"):
            forecast_to_xarray(fc)
        return

    ds_fit = fit_to_xarray(fit)
    assert isinstance(ds_fit, xr.Dataset)
    assert ds_fit["beta"].dims == ("draw", "regressor", "variable")
    assert ds_fit["sigma"].dims == ("draw", "variable", "variable2")

    ds_fc = forecast_to_xarray(fc)
    assert isinstance(ds_fc, xr.Dataset)
    assert ds_fc["draws"].dims == ("draw", "horizon", "variable")
    assert ds_fc["quantiles"].dims == ("quantile", "horizon", "variable")
