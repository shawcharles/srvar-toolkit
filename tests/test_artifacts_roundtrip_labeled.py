from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from srvar import Dataset, ElbSpec, VolatilitySpec
from srvar.api import fit, forecast
from srvar.artifacts import load_fit_npz, load_forecast_npz, save_fit_npz, save_forecast_npz
from srvar.results import FitResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.xarray import fit_to_xarray


def _rebuild_fitresult_from_npz(*, original: FitResult, fit_npz: object) -> FitResult:
    return FitResult(
        dataset=fit_npz.dataset,
        model=original.model,
        prior=original.prior,
        sampler=original.sampler,
        posterior=fit_npz.posterior,
        latent_dataset=fit_npz.latent_dataset,
        latent_draws=fit_npz.latent_draws,
        beta_draws=fit_npz.beta_draws,
        sigma_draws=fit_npz.sigma_draws,
        q_draws=fit_npz.q_draws,
        h_draws=fit_npz.h_draws,
        h0_draws=fit_npz.h0_draws,
        sigma_eta2_draws=fit_npz.sigma_eta2_draws,
        sv_gamma0_draws=fit_npz.sv_gamma0_draws,
        sv_phi_draws=fit_npz.sv_phi_draws,
        lambda_draws=fit_npz.lambda_draws,
        factor_draws=fit_npz.factor_draws,
        h_factor_draws=fit_npz.h_factor_draws,
        h0_factor_draws=fit_npz.h0_factor_draws,
        sigma_eta2_factor_draws=fit_npz.sigma_eta2_factor_draws,
        gamma_draws=fit_npz.gamma_draws,
        mu_draws=fit_npz.mu_draws,
        mu_gamma_draws=fit_npz.mu_gamma_draws,
    )


def test_fit_npz_roundtrip_homoskedastic_and_xarray(tmp_path: Path) -> None:
    pytest.importorskip("xarray")

    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=40, freq="MS")
    ds = Dataset.from_arrays(
        values=rng.standard_normal((40, 2)), variables=["y1", "y2"], time_index=time
    )

    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=8, burn_in=0, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(1))
    path = tmp_path / "fit_result.npz"
    save_fit_npz(path, fit_res)

    fit_npz = load_fit_npz(path)
    assert fit_npz.beta_draws is not None
    assert fit_npz.sigma_draws is not None
    assert np.array_equal(fit_npz.beta_draws, fit_res.beta_draws)
    assert np.array_equal(fit_npz.sigma_draws, fit_res.sigma_draws)

    fit_loaded = _rebuild_fitresult_from_npz(original=fit_res, fit_npz=fit_npz)
    ds_xr = fit_to_xarray(fit_loaded)
    assert {"y", "beta", "sigma"} <= set(ds_xr.data_vars)


def test_fit_npz_roundtrip_fsv_elb_and_xarray(tmp_path: Path) -> None:
    pytest.importorskip("xarray")

    rng = np.random.default_rng(2)
    t = 50
    bound = 0.0
    time = pd.date_range("2000-01-01", periods=t, freq="MS")

    r_latent = rng.standard_normal(t)
    r_obs = np.maximum(r_latent, bound)
    y2 = rng.standard_normal(t)
    ds = Dataset.from_arrays(
        values=np.column_stack([r_obs, y2]), variables=["r", "y2"], time_index=time
    )

    model = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=bound, applies_to=["r"]),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=1,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=10, burn_in=0, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(3))
    path = tmp_path / "fit_result_fsv.npz"
    save_fit_npz(path, fit_res)

    fit_npz = load_fit_npz(path)
    assert fit_npz.lambda_draws is not None
    assert fit_npz.h_factor_draws is not None
    assert fit_npz.h_draws is not None
    assert fit_npz.latent_dataset is not None

    fit_loaded = _rebuild_fitresult_from_npz(original=fit_res, fit_npz=fit_npz)
    ds_xr = fit_to_xarray(fit_loaded)
    assert {"y", "y_latent", "lambda", "loadings", "h_factor"} <= set(ds_xr.data_vars)


def test_forecast_npz_roundtrip_preserves_latent_draws(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    t = 40
    bound = 0.0
    time = pd.date_range("2000-01-01", periods=t, freq="MS")

    r_latent = rng.standard_normal(t)
    r_obs = np.maximum(r_latent, bound)
    y2 = rng.standard_normal(t)
    ds = Dataset.from_arrays(
        values=np.column_stack([r_obs, y2]), variables=["r", "y2"], time_index=time
    )

    model = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=bound, applies_to=["r"]),
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=1,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=8, burn_in=0, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(5))
    fc = forecast(fit_res, horizons=[1, 2], draws=5, rng=np.random.default_rng(6))
    assert fc.latent_draws is not None

    path = tmp_path / "forecast_result.npz"
    save_forecast_npz(path, fc)
    fc_loaded = load_forecast_npz(path)

    assert fc_loaded.latent_draws is not None
    assert np.array_equal(fc_loaded.draws, fc.draws)
    assert np.array_equal(fc_loaded.latent_draws, fc.latent_draws)
