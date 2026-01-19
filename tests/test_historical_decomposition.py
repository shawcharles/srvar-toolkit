import numpy as np
import pytest

from srvar.analysis import historical_decomposition_cholesky
from srvar.data.dataset import Dataset
from srvar.elb import ElbSpec
from srvar.results import FitResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import VolatilitySpec


def _make_fit(
    *,
    dataset: Dataset,
    model: ModelSpec,
    beta_draws: np.ndarray,
    sigma_draws: np.ndarray | None = None,
    h_draws: np.ndarray | None = None,
    q_draws: np.ndarray | None = None,
    lambda_draws: np.ndarray | None = None,
    h_factor_draws: np.ndarray | None = None,
) -> FitResult:
    k = int(beta_draws.shape[1])
    n = int(dataset.N)
    prior = PriorSpec.niw_default(k=k, n=n)
    sampler = SamplerConfig(draws=int(beta_draws.shape[0]), burn_in=0, thin=1)
    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=np.asarray(beta_draws, dtype=float),
        sigma_draws=None if sigma_draws is None else np.asarray(sigma_draws, dtype=float),
        h_draws=None if h_draws is None else np.asarray(h_draws, dtype=float),
        q_draws=None if q_draws is None else np.asarray(q_draws, dtype=float),
        lambda_draws=None if lambda_draws is None else np.asarray(lambda_draws, dtype=float),
        h_factor_draws=None if h_factor_draws is None else np.asarray(h_factor_draws, dtype=float),
    )


def test_historical_decomposition_cholesky_reconstructs_ar1() -> None:
    t = 6
    y = np.empty(t, dtype=float)
    y[0] = 1.0
    eps = np.array([0.0, 0.2, -0.1, 0.0, 0.3, -0.2], dtype=float)
    for i in range(1, t):
        y[i] = 0.5 * y[i - 1] + eps[i]

    ds = Dataset.from_arrays(values=y.reshape(-1, 1), variables=["y"])
    beta = np.array([[0.0], [0.5]], dtype=float)  # intercept + lag
    sigma = np.array([[1.0]], dtype=float)
    fit = _make_fit(
        dataset=ds,
        model=ModelSpec(p=1, include_intercept=True),
        beta_draws=beta[None, :, :],
        sigma_draws=sigma[None, :, :],
    )

    hd = historical_decomposition_cholesky(
        fit,
        draws=1,
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )
    assert hd.variables == ["y"]
    assert hd.shocks == ["y"]
    assert hd.baseline_draws.shape == (1, t - 1, 1)
    assert hd.shock_draws.shape == (1, t - 1, 1)
    assert hd.contributions_draws.shape == (1, t - 1, 1, 1)
    assert float(hd.metadata["reconstruction_max_abs_error"]) < 1e-10

    # Structural shocks equal reduced-form residuals under sigma=I.
    assert np.allclose(hd.shock_draws[0, :, 0], eps[1:])

    baseline_expected = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=float)
    assert np.allclose(hd.baseline_draws[0, :, 0], baseline_expected)

    recon = hd.baseline_draws[0, :, 0] + hd.contributions_draws[0, :, 0, 0]
    assert np.allclose(recon, y[1:])


def test_historical_decomposition_cholesky_supports_diagonal_sv() -> None:
    t = 6
    y = np.empty(t, dtype=float)
    y[0] = 1.0

    # Time-varying log-variance h_t (so sd_t = exp(0.5 h_t)).
    h = np.array([0.0, np.log(4.0), 0.0, np.log(9.0), 0.0, np.log(4.0)], dtype=float)
    sd = np.exp(0.5 * h)
    eps = np.array([0.0, 0.5, -0.25, 0.1, 0.0, -0.4], dtype=float)
    for i in range(1, t):
        y[i] = 0.5 * y[i - 1] + sd[i] * eps[i]

    ds = Dataset.from_arrays(values=y.reshape(-1, 1), variables=["y"])
    beta = np.array([[0.0], [0.5]], dtype=float)  # intercept + lag
    fit = _make_fit(
        dataset=ds,
        model=ModelSpec(p=1, include_intercept=True, volatility=VolatilitySpec(enabled=True)),
        beta_draws=beta[None, :, :],
        h_draws=h.reshape(1, t, 1),
    )

    hd = historical_decomposition_cholesky(
        fit,
        draws=1,
        quantile_levels=[0.5],
        rng=np.random.default_rng(0),
    )
    assert float(hd.metadata["reconstruction_max_abs_error"]) < 1e-10
    assert np.allclose(hd.shock_draws[0, :, 0], eps[1:])


def test_historical_decomposition_defaults_to_latent_for_elb() -> None:
    ds = Dataset.from_arrays(values=np.zeros((3, 1)), variables=["y"])
    beta = np.array([[0.0], [0.5]], dtype=float)
    sigma = np.array([[1.0]], dtype=float)
    fit = _make_fit(
        dataset=ds,
        model=ModelSpec(p=1, include_intercept=True, elb=ElbSpec(bound=0.0, applies_to=["y"])),
        beta_draws=beta[None, :, :],
        sigma_draws=sigma[None, :, :],
    )

    with pytest.raises(ValueError, match="fit\\.latent_dataset"):
        _ = historical_decomposition_cholesky(fit, draws=1, rng=np.random.default_rng(0))


def test_historical_decomposition_cholesky_supports_factor_sv() -> None:
    rng = np.random.default_rng(0)
    t = 8
    p = 1
    n = 2
    t_eff = t - p

    a = np.array([[0.5, 0.0], [0.0, 0.25]], dtype=float)
    beta = np.array(
        [
            [0.0, 0.0],  # intercept
            [0.5, 0.0],  # y1_{t-1}
            [0.0, 0.25],  # y2_{t-1}
        ],
        dtype=float,
    )

    lam = np.array([[1.0], [0.5]], dtype=float)

    # Effective-sample log-variance states (T - p).
    h_f = np.linspace(np.log(0.5), np.log(2.0), t_eff, dtype=float)  # (T - p,)
    h_eta = np.column_stack(
        [
            np.linspace(np.log(1.0), np.log(3.0), t_eff, dtype=float),
            np.linspace(np.log(2.0), np.log(4.0), t_eff, dtype=float),
        ]
    )

    y = np.zeros((t, n), dtype=float)
    y[0] = np.array([1.0, -1.0], dtype=float)
    z = rng.standard_normal((t_eff, n))
    for tt in range(t_eff):
        sigma_t = (
            lam @ np.diag(np.exp([h_f[tt]])) @ lam.T + np.diag(np.exp(h_eta[tt, :]))
        )
        l_t = np.linalg.cholesky(sigma_t)
        eps_t = l_t @ z[tt]
        y[tt + 1] = a @ y[tt] + eps_t

    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])
    fit = _make_fit(
        dataset=ds,
        model=ModelSpec(
            p=p,
            include_intercept=True,
            volatility=VolatilitySpec(
                enabled=True,
                covariance="factor",
                dynamics="rw",
                k_factors=1,
            ),
        ),
        beta_draws=beta[None, :, :],
        h_draws=h_eta.reshape(1, t_eff, n),
        lambda_draws=lam.reshape(1, n, 1),
        h_factor_draws=h_f.reshape(1, t_eff, 1),
    )

    hd = historical_decomposition_cholesky(
        fit,
        draws=1,
        quantile_levels=[0.5],
        rng=np.random.default_rng(1),
    )
    assert float(hd.metadata["reconstruction_max_abs_error"]) < 1e-10
    assert np.allclose(hd.shock_draws[0], z, atol=1e-10)
