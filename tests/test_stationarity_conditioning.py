import numpy as np
import pytest

from srvar.api import forecast
from srvar.data.dataset import Dataset
from srvar.results import FitResult, PosteriorNIW
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _make_ar1_fit(*, phi_draws: list[float]) -> FitResult:
    dataset = Dataset.from_arrays(values=np.array([[0.0]]), variables=["y"])
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=2, n=1)
    sampler = SamplerConfig(draws=max(1, len(phi_draws)), burn_in=0, thin=1)

    beta_draws = np.stack([np.array([[1.0], [float(phi)]]) for phi in phi_draws], axis=0)
    sigma = np.zeros((1, 1), dtype=float)
    sigma_draws = np.repeat(sigma[None, :, :], repeats=int(beta_draws.shape[0]), axis=0)

    return FitResult(
        dataset=dataset,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=beta_draws,
        sigma_draws=sigma_draws,
    )


def test_forecast_stationarity_reject_filters_unstable_beta_draws() -> None:
    fit = _make_ar1_fit(phi_draws=[0.5, 1.5])

    res_allow = forecast(
        fit,
        horizons=[5],
        draws=50,
        quantile_levels=[0.5],
        stationarity="allow",
        rng=np.random.default_rng(0),
    )
    res_reject = forecast(
        fit,
        horizons=[5],
        draws=50,
        quantile_levels=[0.5],
        stationarity="reject",
        rng=np.random.default_rng(0),
    )

    # Stable AR(1) with intercept=1 converges to 2.0; at h=5 it equals 1.9375.
    assert np.isclose(float(res_reject.mean[4, 0]), 1.9375)
    assert float(res_reject.draws[:, 4, 0].max()) < 2.0

    # Unstable draws are allowed under the default policy, producing much larger forecasts.
    assert float(res_allow.draws[:, 4, 0].max()) > 5.0


def test_forecast_stationarity_reject_raises_when_no_stationary_draws() -> None:
    fit = _make_ar1_fit(phi_draws=[1.5])
    with pytest.raises(ValueError, match="no stationary coefficient draws"):
        forecast(
            fit,
            horizons=[1],
            draws=1,
            quantile_levels=[0.5],
            stationarity="reject",
            rng=np.random.default_rng(0),
        )


def test_forecast_stationarity_reject_respects_max_draws_in_posterior_sampling(monkeypatch) -> None:
    dataset = Dataset.from_arrays(values=np.array([[0.0]]), variables=["y"])
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=2, n=1)
    sampler = SamplerConfig(draws=1, burn_in=0, thin=1)

    posterior = PosteriorNIW(
        mn=np.zeros((2, 1), dtype=float),
        vn=np.eye(2, dtype=float),
        sn=np.eye(1, dtype=float),
        nun=5.0,
    )
    fit = FitResult(dataset=dataset, model=model, prior=prior, sampler=sampler, posterior=posterior)

    def _fake_sample_posterior_niw(*, mn, vn, sn, nun, draws, rng):
        _ = (mn, vn, sn, nun, rng)
        beta = np.array([[1.0], [1.5]], dtype=float)  # unstable AR(1)
        beta_draws = np.repeat(beta[None, :, :], repeats=int(draws), axis=0)
        sigma_draws = np.zeros((int(draws), 1, 1), dtype=float)
        return beta_draws, sigma_draws

    import srvar.api as api_mod

    monkeypatch.setattr(api_mod, "sample_posterior_niw", _fake_sample_posterior_niw)

    with pytest.raises(ValueError, match="could not generate"):
        forecast(
            fit,
            horizons=[1],
            draws=2,
            quantile_levels=[0.5],
            stationarity="reject",
            stationarity_max_draws=5,
            rng=np.random.default_rng(0),
        )
