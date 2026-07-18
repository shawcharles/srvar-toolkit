import numpy as np
import pandas as pd
import pytest

from srvar.api import fit, forecast
from srvar.data.dataset import Dataset
from srvar.results import FitResult, ForecastResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.xarray import (
    fevd_to_xarray,
    fit_to_xarray,
    forecast_to_xarray,
    historical_decomposition_to_xarray,
    irf_to_xarray,
)


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

    ds = Dataset.from_arrays(
        values=np.zeros((t, n), dtype=float), variables=["y1", "y2"], time_index=time
    )
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

    ds = Dataset.from_arrays(
        values=np.zeros((t, n), dtype=float), variables=["a", "b", "c"], time_index=time
    )
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


def test_xarray_fit_forecast_integration_matrix() -> None:
    xr = pytest.importorskip("xarray")  # noqa: F841

    # 1) Homoskedastic NIW
    rng = np.random.default_rng(0)
    ds = Dataset.from_arrays(values=rng.standard_normal((40, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=8, burn_in=0, thin=1)
    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(1))
    ds_fit = fit_to_xarray(fit_res)
    assert {"y", "beta", "sigma"} <= set(ds_fit.data_vars)

    fc = forecast(fit_res, horizons=[1, 2], draws=5, rng=np.random.default_rng(2))
    ds_fc = forecast_to_xarray(fc)
    assert "draws" in ds_fc.data_vars

    # 2) ELB-only (shadow-rate augmentation)
    from srvar import ElbSpec

    bound = 0.0
    r_lat = rng.standard_normal(45)
    r_obs = np.maximum(r_lat, bound)
    y2 = rng.standard_normal(45)
    ds_elb = Dataset.from_arrays(values=np.column_stack([r_obs, y2]), variables=["r", "y2"])
    model_elb = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=bound, applies_to=["r"]),
    )
    prior_elb = PriorSpec.niw_default(k=1 + ds_elb.N * model_elb.p, n=ds_elb.N)
    sampler_elb = SamplerConfig(draws=12, burn_in=0, thin=1)
    fit_elb = fit(ds_elb, model_elb, prior_elb, sampler_elb, rng=np.random.default_rng(3))
    ds_fit_elb = fit_to_xarray(fit_elb)
    assert "y_latent" in ds_fit_elb.data_vars
    assert "latent_draws" in ds_fit_elb.data_vars

    fc_elb = forecast(fit_elb, horizons=[1], draws=4, rng=np.random.default_rng(4))
    ds_fc_elb = forecast_to_xarray(fc_elb)
    assert "latent_draws" in ds_fc_elb.data_vars

    # 3) Diagonal SVRW
    from srvar import VolatilitySpec

    ds_sv = Dataset.from_arrays(values=rng.standard_normal((50, 2)), variables=["y1", "y2"])
    model_sv = ModelSpec(
        p=2,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="diagonal", dynamics="rw"),
    )
    prior_sv = PriorSpec.niw_default(k=1 + ds_sv.N * model_sv.p, n=ds_sv.N)
    sampler_sv = SamplerConfig(draws=10, burn_in=0, thin=1)
    fit_sv = fit(ds_sv, model_sv, prior_sv, sampler_sv, rng=np.random.default_rng(5))
    ds_fit_sv = fit_to_xarray(fit_sv)
    assert "h" in ds_fit_sv.data_vars
    assert np.isnan(ds_fit_sv["h"].isel(draw=0, variable=0).values[: model_sv.p]).all()

    # 4) Triangular SV (time-invariant correlations)
    model_tri = ModelSpec(
        p=2,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="triangular", dynamics="rw"),
    )
    prior_tri = PriorSpec.niw_default(k=1 + ds_sv.N * model_tri.p, n=ds_sv.N)
    sampler_tri = SamplerConfig(draws=10, burn_in=0, thin=1)
    fit_tri = fit(ds_sv, model_tri, prior_tri, sampler_tri, rng=np.random.default_rng(6))
    ds_fit_tri = fit_to_xarray(fit_tri)
    assert "q" in ds_fit_tri.data_vars
    assert ds_fit_tri["q"].dims == ("draw", "variable", "variable2")

    # 5) “Maximal” FSV: ELB + steady-state + robust shocks
    from srvar.spec import ShockSpec, SteadyStateSpec

    model_fsv = ModelSpec(
        p=2,
        include_intercept=True,
        steady_state=SteadyStateSpec(mu0=np.zeros(ds_elb.N), v0_mu=0.1),
        elb=ElbSpec(bound=bound, applies_to=["r"]),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=1,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
        shocks=ShockSpec(family="student_t", df=7.0),
    )
    prior_fsv = PriorSpec.niw_default(k=1 + ds_elb.N * model_fsv.p, n=ds_elb.N)
    sampler_fsv = SamplerConfig(draws=12, burn_in=0, thin=1)
    fit_fsv = fit(ds_elb, model_fsv, prior_fsv, sampler_fsv, rng=np.random.default_rng(7))
    ds_fit_fsv = fit_to_xarray(fit_fsv)
    assert {"lambda", "loadings", "h_factor"} <= set(ds_fit_fsv.data_vars)
    assert ds_fit_fsv["loadings"].attrs.get("alias_of") == "lambda"

    fc_fsv = forecast(fit_fsv, horizons=[1, 2], draws=4, rng=np.random.default_rng(8))
    ds_fc_fsv = forecast_to_xarray(fc_fsv)
    assert "draws" in ds_fc_fsv.data_vars


def test_structural_results_convert_to_xarray() -> None:
    pytest.importorskip("xarray")

    from srvar.analysis.fevd import fevd_cholesky
    from srvar.analysis.hd import historical_decomposition_cholesky
    from srvar.analysis.irf import irf_cholesky

    rng = np.random.default_rng(123)
    ds = Dataset.from_arrays(values=rng.standard_normal((60, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=12, burn_in=0, thin=1)
    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))

    irf = irf_cholesky(fit_res, horizons=4, draws=5, rng=np.random.default_rng(1))
    ds_irf = irf_to_xarray(irf)
    assert ds_irf["draws"].dims == ("draw", "horizon", "variable", "shock")

    fevd = fevd_cholesky(fit_res, horizons=4, draws=5, rng=np.random.default_rng(2))
    ds_fevd = fevd_to_xarray(fevd)
    assert ds_fevd["draws"].dims == ("draw", "horizon", "variable", "shock")

    hd = historical_decomposition_cholesky(fit_res, draws=5, rng=np.random.default_rng(3))
    ds_hd = historical_decomposition_to_xarray(hd)
    assert ds_hd["contributions_draws"].dims == ("draw", "time", "variable", "shock")
