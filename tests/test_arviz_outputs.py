import numpy as np
import pytest

from srvar.arviz import fit_to_inferencedata, forecast_to_inferencedata
from srvar.data.dataset import Dataset
from srvar.results import FitResult, ForecastResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_arviz_converters_import_or_convert() -> None:
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
        quantiles={},
    )

    try:
        import arviz as az  # type: ignore
    except Exception:
        with pytest.raises(ImportError, match="arviz is required"):
            fit_to_inferencedata(fit)
        with pytest.raises(ImportError, match="arviz is required"):
            forecast_to_inferencedata(fc)
        return

    id_fit = fit_to_inferencedata(fit)
    assert isinstance(id_fit, az.InferenceData)
    assert hasattr(id_fit, "posterior")

    id_fc = forecast_to_inferencedata(fc)
    assert isinstance(id_fc, az.InferenceData)
    assert hasattr(id_fc, "posterior_predictive")
