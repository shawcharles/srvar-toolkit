import numpy as np
import pytest

from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _toy_fit_with_fsv_loadings() -> FitResult:
    ds = Dataset.from_arrays(values=np.zeros((5, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)
    lam = np.arange(2, dtype=float).reshape(1, 2, 1)
    return FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        lambda_draws=lam,
    )


def test_fitresult_loadings_alias_mirrors_lambda_draws() -> None:
    fit = _toy_fit_with_fsv_loadings()
    assert fit.lambda_draws is not None
    assert fit.loading_draws is not None
    assert np.array_equal(fit.loading_draws, fit.lambda_draws)


def test_xarray_fit_includes_loadings_alias_variable() -> None:
    pytest.importorskip("xarray")
    from srvar.xarray import fit_to_xarray

    fit = _toy_fit_with_fsv_loadings()
    ds = fit_to_xarray(fit)

    assert "lambda" in ds.data_vars
    assert "loadings" in ds.data_vars
    assert ds["loadings"].attrs.get("alias_of") == "lambda"
    assert ds["loadings"].dims == ds["lambda"].dims


def test_arviz_fit_excludes_alias_variables() -> None:
    pytest.importorskip("arviz")
    from srvar.arviz import fit_to_inferencedata

    fit = _toy_fit_with_fsv_loadings()
    idata = fit_to_inferencedata(fit)
    assert "lambda" in idata.posterior.data_vars
    assert "loadings" not in idata.posterior.data_vars

