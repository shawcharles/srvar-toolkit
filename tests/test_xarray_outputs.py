import numpy as np
import pandas as pd
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


def test_fit_to_xarray_aligns_sv_states_to_full_time_index() -> None:
    try:
        import xarray as xr  # type: ignore
    except Exception:
        pytest.skip("xarray not installed")

    t = 6
    p = 2
    n = 2
    time = pd.date_range("2000-01-01", periods=t, freq="MS")

    ds = Dataset.from_arrays(values=np.zeros((t, n), dtype=float), variables=["y1", "y2"], time_index=time)
    model = ModelSpec(p=p, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + n * p, n=n)
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)

    h_draws = np.arange((t - p) * n, dtype=float).reshape(1, t - p, n)
    fit = FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        h_draws=h_draws,
    )

    out = fit_to_xarray(fit)
    assert isinstance(out, xr.Dataset)
    assert out["h"].dims == ("draw", "time", "variable")
    assert out["h"].shape == (1, t, n)

    values = out["h"].values
    assert np.isnan(values[:, :p, :]).all()
    assert np.array_equal(values[:, p:, :], h_draws)


def test_fit_to_xarray_includes_factor_sv_draws_when_present() -> None:
    try:
        import xarray as xr  # type: ignore
    except Exception:
        pytest.skip("xarray not installed")

    t = 7
    p = 2
    n = 3
    k = 2
    draws = 2
    time = pd.date_range("2000-01-01", periods=t, freq="MS")

    ds = Dataset.from_arrays(values=np.zeros((t, n), dtype=float), variables=["a", "b", "c"], time_index=time)
    model = ModelSpec(p=p, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + n * p, n=n)
    sampler = SamplerConfig(draws=draws, burn_in=0, thin=1)

    t_eff = t - p
    lambda_draws = np.ones((draws, n, k), dtype=float)
    factor_draws = np.arange(draws * t_eff * k, dtype=float).reshape(draws, t_eff, k)
    h_factor_draws = np.zeros((draws, t_eff, k), dtype=float)
    h0_factor_draws = np.zeros((draws, k), dtype=float)
    sigma_eta2_factor_draws = np.ones((draws, k), dtype=float)

    fit = FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        lambda_draws=lambda_draws,
        factor_draws=factor_draws,
        h_factor_draws=h_factor_draws,
        h0_factor_draws=h0_factor_draws,
        sigma_eta2_factor_draws=sigma_eta2_factor_draws,
    )

    out = fit_to_xarray(fit)
    assert isinstance(out, xr.Dataset)
    assert out["lambda"].dims == ("draw", "variable", "factor")
    assert out["h_factor"].dims == ("draw", "time", "factor")
    assert out["h0_factor"].dims == ("draw", "factor")
    assert out["sigma_eta2_factor"].dims == ("draw", "factor")
    assert out["factors"].dims == ("draw", "time", "factor")

    assert out["lambda"].shape == (draws, n, k)
    assert out["factors"].shape == (draws, t, k)

    values = out["factors"].values
    assert np.isnan(values[:, :p, :]).all()
    assert np.array_equal(values[:, p:, :], factor_draws)
