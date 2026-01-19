from __future__ import annotations

from typing import Any


def _require_arviz() -> Any:
    try:
        import arviz as az  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "arviz is required for InferenceData outputs. Install with `pip install srvar-toolkit[arviz]`."
        ) from exc
    return az


def fit_to_inferencedata(fit: Any) -> Any:
    """Convert a :class:`~srvar.results.FitResult` into an `arviz.InferenceData`."""
    az = _require_arviz()
    from .xarray import fit_to_xarray

    ds = fit_to_xarray(fit)

    posterior_vars = [
        name
        for name in ds.data_vars
        if "draw" in ds[name].dims and not ds[name].attrs.get("alias_of")
    ]
    posterior = ds[posterior_vars].expand_dims(chain=[0])

    obs_vars = [name for name in ["y"] if name in ds.data_vars]
    if obs_vars:
        observed_data = ds[obs_vars]
        return az.InferenceData(posterior=posterior, observed_data=observed_data)
    return az.InferenceData(posterior=posterior)


def forecast_to_inferencedata(fc: Any) -> Any:
    """Convert a :class:`~srvar.results.ForecastResult` into an `arviz.InferenceData`."""
    az = _require_arviz()
    from .xarray import forecast_to_xarray

    ds = forecast_to_xarray(fc)
    pp_vars = [name for name in ["draws", "latent_draws"] if name in ds.data_vars]
    posterior_predictive = ds[pp_vars].expand_dims(chain=[0])
    return az.InferenceData(posterior_predictive=posterior_predictive)
