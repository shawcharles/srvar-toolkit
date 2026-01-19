import numpy as np
import pytest

from srvar import VolatilitySpec
from srvar.analysis import irf_cholesky, irf_reduced_form
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


def test_irf_cholesky_scaling_on_diagonal_system() -> None:
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

    res_one_sd = irf_cholesky(fit, horizons=4, shock_scale="one_sd", quantile_levels=[0.5])
    assert res_one_sd.draws.shape == (1, 5, 2, 2)
    assert res_one_sd.variables == ["y1", "y2"]
    assert res_one_sd.shocks == ["y1", "y2"]

    assert np.allclose(res_one_sd.mean[:, 0, 0], [2.0, 1.0, 0.5, 0.25, 0.125])
    assert np.allclose(res_one_sd.mean[:, 1, 1], [3.0, 0.75, 0.1875, 0.046875, 0.01171875])
    assert np.allclose(res_one_sd.mean[:, 0, 1], 0.0)
    assert np.allclose(res_one_sd.mean[:, 1, 0], 0.0)

    res_unit = irf_cholesky(fit, horizons=4, shock_scale="unit", quantile_levels=[0.5])
    assert np.allclose(res_unit.mean[:, 0, 0], [1.0, 0.5, 0.25, 0.125, 0.0625])
    assert np.allclose(res_unit.mean[:, 1, 1], [1.0, 0.25, 0.0625, 0.015625, 0.00390625])


def test_irf_cholesky_ordering_relabels_and_reorders_system() -> None:
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

    res = irf_cholesky(
        fit,
        horizons=1,
        ordering=["y2", "y1"],
        shock_scale="one_sd",
        quantile_levels=[0.5],
    )
    assert res.variables == ["y2", "y1"]
    assert res.shocks == ["y2", "y1"]
    assert np.allclose(res.mean[:, 0, 0], [3.0, 0.75])
    assert np.allclose(res.mean[:, 1, 1], [2.0, 1.0])
    assert np.allclose(res.mean[:, 0, 1], 0.0)
    assert np.allclose(res.mean[:, 1, 0], 0.0)


def test_irf_stationarity_reject_filters_unstable_draws() -> None:
    beta_stable = np.array([[0.0], [0.5]], dtype=float)  # intercept + lag
    beta_unstable = np.array([[0.0], [1.5]], dtype=float)
    beta_draws = np.stack([beta_stable, beta_unstable], axis=0)
    sigma_draws = np.repeat(np.ones((1, 1), dtype=float)[None, :, :], repeats=2, axis=0)
    fit = _make_fit(variables=["y"], beta_draws=beta_draws, sigma_draws=sigma_draws)

    res_allow = irf_reduced_form(
        fit, horizons=10, draws=50, stationarity="allow", rng=np.random.default_rng(0)
    )
    res_reject = irf_reduced_form(
        fit, horizons=10, draws=50, stationarity="reject", rng=np.random.default_rng(0)
    )

    # horizon 10 response for stable AR(1) is 0.5^10; allowing unstable draws makes it huge.
    assert np.isclose(float(res_reject.mean[-1, 0, 0]), 0.5**10)
    assert float(res_reject.draws[:, -1, 0, 0].max()) < 0.01
    assert float(res_allow.mean[-1, 0, 0]) > 5.0


def test_irf_stationarity_reject_raises_when_no_stationary_draws() -> None:
    beta_unstable = np.array([[0.0], [1.5]], dtype=float)
    fit = _make_fit(
        variables=["y"],
        beta_draws=beta_unstable[None, :, :],
        sigma_draws=np.ones((1, 1, 1), dtype=float),
    )
    with pytest.raises(ValueError, match="no stationary coefficient draws"):
        _ = irf_reduced_form(fit, horizons=1, stationarity="reject", rng=np.random.default_rng(0))


def test_irf_cholesky_supports_factor_sv_covariance_state() -> None:
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1_{t-1}
            [0.0, 0.25],  # y2_{t-1}
        ],
        dtype=float,
    )

    variables = ["y1", "y2"]
    ds = Dataset.from_arrays(values=np.zeros((3, 2), dtype=float), variables=variables)
    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="factor", dynamics="rw", k_factors=1),
    )
    prior = PriorSpec.niw_default(k=1 + len(variables) * model.p, n=len(variables))
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)

    # Effective-sample volatility states (T - p).
    h_eta = np.array([[0.0, 0.0], [0.0, np.log(9.0)]], dtype=float)[None, :, :]
    h_f = np.array([[0.0], [np.log(4.0)]], dtype=float)[None, :, :]
    lam = np.array([[1.0], [0.5]], dtype=float)[None, :, :]

    fit = FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=beta[None, :, :],
        h_draws=h_eta,
        lambda_draws=lam,
        h_factor_draws=h_f,
    )

    res = irf_cholesky(
        fit,
        horizons=0,
        draws=1,
        shock_scale="one_sd",
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )

    sigma0 = np.array([[5.0, 2.0], [2.0, 10.0]], dtype=float)
    impact = np.linalg.cholesky(sigma0)
    assert np.allclose(res.mean[0], impact)
