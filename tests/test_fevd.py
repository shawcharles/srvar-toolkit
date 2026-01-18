import numpy as np
import pytest

from srvar.analysis import fevd_cholesky, fevd_from_irf, irf_cholesky, irf_reduced_form
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


def test_fevd_identity_on_diagonal_system() -> None:
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1_{t-1}
            [0.0, 0.25],  # y2_{t-1}
        ],
        dtype=float,
    )
    sigma = np.diag([4.0, 9.0])
    fit = _make_fit(
        variables=["y1", "y2"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :]
    )

    fevd = fevd_cholesky(fit, horizons=4, quantile_levels=[0.5])
    assert fevd.horizons == [1, 2, 3, 4]
    assert fevd.draws.shape == (1, 4, 2, 2)

    assert np.allclose(fevd.mean[:, 0, 0], 1.0)
    assert np.allclose(fevd.mean[:, 1, 1], 1.0)
    assert np.allclose(fevd.mean[:, 0, 1], 0.0)
    assert np.allclose(fevd.mean[:, 1, 0], 0.0)


def test_fevd_horizon_indexing_and_sums_to_one() -> None:
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1_{t-1}
            [0.4, 0.2],  # y2_{t-1} cross-effect into y1
        ],
        dtype=float,
    )
    sigma = np.eye(2, dtype=float)
    fit = _make_fit(
        variables=["y1", "y2"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :]
    )

    fevd = fevd_cholesky(fit, horizons=[1, 2], shock_scale="one_sd", quantile_levels=[0.5])

    # One-step ahead FEVD (h=1) uses only IRF horizon 0: y1 has no immediate response to shock y2.
    assert np.isclose(float(fevd.mean[0, 0, 1]), 0.0)
    # Two-step ahead (h=2) includes IRF horizon 1: y1 responds to shock y2 through A[0,1].
    assert float(fevd.mean[1, 0, 1]) > 0.0

    assert np.allclose(fevd.draws.sum(axis=3), 1.0)
    assert np.allclose(fevd.mean.sum(axis=2), 1.0)


def test_fevd_requires_structural_irf() -> None:
    beta = np.array([[0.0], [0.5]], dtype=float)
    sigma = np.ones((1, 1), dtype=float)
    fit = _make_fit(variables=["y"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :])

    irf = irf_reduced_form(fit, horizons=2, quantile_levels=[0.5])
    with pytest.raises(ValueError, match="structural IRF"):
        _ = fevd_from_irf(irf, horizons=1)


def test_fevd_requires_contiguous_irf_horizons() -> None:
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1_{t-1}
            [0.4, 0.2],  # y2_{t-1}
        ],
        dtype=float,
    )
    sigma = np.eye(2, dtype=float)
    fit = _make_fit(
        variables=["y1", "y2"], beta_draws=beta[None, :, :], sigma_draws=sigma[None, :, :]
    )

    irf_sparse = irf_cholesky(fit, horizons=[0, 2], quantile_levels=[0.5])
    with pytest.raises(ValueError, match=r"missing: \[1\]"):
        _ = fevd_from_irf(irf_sparse, horizons=[3])
