import numpy as np
import pytest

from srvar.analysis import irf_sign_restricted
from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _make_fit(
    *, variables: list[str], beta_draws: np.ndarray, sigma_draws: np.ndarray
) -> FitResult:
    ds = Dataset.from_arrays(values=np.zeros((2, len(variables))), variables=variables)
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + len(variables) * model.p, n=len(variables))
    sampler = SamplerConfig(draws=int(beta_draws.shape[0]), burn_in=0, thin=1)
    return FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=np.asarray(beta_draws, dtype=float),
        sigma_draws=np.asarray(sigma_draws, dtype=float),
    )


def test_irf_sign_restricted_enforces_cumulative_signs_in_1d() -> None:
    beta = np.array([[0.0], [0.5]], dtype=float)  # intercept + lag
    sigma = np.array([[1.0]], dtype=float)
    fit = _make_fit(variables=["y"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :])

    res = irf_sign_restricted(
        fit,
        horizons=2,
        restrictions={
            "mp": {
                "y": {"sign": "+", "horizons": [0, 1, 2], "cumulative": True},
            }
        },
        draws=20,
        max_attempts=50,
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )
    assert res.identification == "sign_restricted"
    assert res.variables == ["y"]
    assert res.shocks == ["mp"]
    assert res.horizons == [0, 1, 2]
    assert res.draws.shape == (20, 3, 1, 1)

    cum = np.cumsum(res.draws[:, :, 0, 0], axis=1)
    assert np.all(cum >= 0.0)


def test_irf_sign_restricted_accepts_empty_restrictions() -> None:
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1_{t-1}
            [0.0, 0.25],  # y2_{t-1}
        ],
        dtype=float,
    )
    sigma = np.eye(2, dtype=float)
    fit = _make_fit(
        variables=["y1", "y2"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :]
    )

    res = irf_sign_restricted(
        fit,
        horizons=1,
        restrictions={},
        draws=3,
        max_attempts=1,
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )
    assert res.draws.shape == (3, 2, 2, 2)
    assert len(res.shocks) == 2


def test_irf_sign_restricted_zero_restriction_is_infeasible_in_1d() -> None:
    beta = np.array([[0.0], [0.5]], dtype=float)
    sigma = np.array([[1.0]], dtype=float)
    fit = _make_fit(variables=["y"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :])

    with pytest.raises(ValueError, match="could not generate the requested number"):
        _ = irf_sign_restricted(
            fit,
            horizons=0,
            restrictions={"s": {"y": {0: "0"}}},
            draws=1,
            max_attempts=10,
            zero_tol=0.0,
            stationarity_max_draws=1,
            rng=np.random.default_rng(0),
        )
