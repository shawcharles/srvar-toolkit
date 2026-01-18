import numpy as np
import pytest

from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.scenario import conditional_forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _make_fit(
    *, variables: list[str], beta: np.ndarray, sigma: np.ndarray, y_last: np.ndarray
) -> FitResult:
    ds = Dataset.from_arrays(values=np.asarray(y_last, dtype=float), variables=variables)
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + len(variables) * model.p, n=len(variables))
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)
    return FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=np.asarray(beta, dtype=float)[None, :, :],
        sigma_draws=np.asarray(sigma, dtype=float)[None, :, :],
    )


def test_conditional_forecast_ar1_pins_first_step() -> None:
    # y_{t+1} = phi * y_t + eps_{t+1}, eps ~ N(0, 1)
    phi = 0.9
    beta = np.array([[0.0], [phi]], dtype=float)
    sigma = np.array([[1.0]], dtype=float)
    fit = _make_fit(variables=["y"], beta=beta, sigma=sigma, y_last=np.array([[1.0]]))

    fc = conditional_forecast(
        fit,
        horizons=[1, 2],
        constraints={"y": {1: 0.0}},
        draws=2000,
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )

    assert np.max(np.abs(fc.draws[:, 0, 0] - 0.0)) < 1e-10
    assert abs(float(fc.mean[0, 0])) < 1e-10

    # With y_{t+1} pinned to 0, y_{t+2} = eps_{t+2}.
    assert abs(float(fc.mean[1, 0])) < 0.05
    assert abs(float(np.var(fc.draws[:, 1, 0]) - 1.0)) < 0.05


def test_conditional_forecast_independent_variable_unchanged() -> None:
    # 2D diagonal VAR(1) with diagonal covariance. Conditioning on y1 should not affect y2.
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1 lag
            [0.0, 0.25],  # y2 lag
        ],
        dtype=float,
    )
    sigma = np.diag([4.0, 9.0])
    fit = _make_fit(variables=["y1", "y2"], beta=beta, sigma=sigma, y_last=np.array([[1.0, 1.0]]))

    fc = conditional_forecast(
        fit,
        horizons=[1],
        constraints={"y1": {1: 0.0}},
        draws=2000,
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )

    assert np.max(np.abs(fc.draws[:, 0, 0] - 0.0)) < 1e-10
    assert abs(float(fc.mean[0, 1]) - 0.25) < 0.1
    assert abs(float(np.var(fc.draws[:, 0, 1]) - 9.0)) < 0.5


def test_conditional_forecast_validates_constraints() -> None:
    beta = np.array([[0.0], [0.5]], dtype=float)
    sigma = np.array([[1.0]], dtype=float)
    fit = _make_fit(variables=["y"], beta=beta, sigma=sigma, y_last=np.array([[0.0]]))

    with pytest.raises(ValueError, match="unknown variable"):
        _ = conditional_forecast(fit, horizons=[1], constraints={"x": {1: 0.0}}, draws=1)

    with pytest.raises(ValueError, match="exceeds max\\(horizons\\)"):
        _ = conditional_forecast(fit, horizons=[1], constraints={"y": {2: 0.0}}, draws=1)
