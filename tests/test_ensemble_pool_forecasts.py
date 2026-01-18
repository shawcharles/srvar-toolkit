import numpy as np
import pytest

from srvar.ensemble import pool_forecasts
from srvar.results import ForecastResult


def _const_forecast(*, value: float, draws: int, horizons: list[int]) -> ForecastResult:
    h = int(max(horizons))
    sims = np.full((int(draws), h, 1), float(value), dtype=float)
    mean = sims.mean(axis=0)
    quantiles = {0.1: np.quantile(sims, q=0.1, axis=0), 0.9: np.quantile(sims, q=0.9, axis=0)}
    return ForecastResult(
        variables=["y"],
        horizons=list(horizons),
        draws=sims,
        mean=mean,
        quantiles=quantiles,
    )


def test_pool_forecasts_weighted_mixture_mean_and_quantiles() -> None:
    fc0 = _const_forecast(value=0.0, draws=10, horizons=[1, 2])
    fc1 = _const_forecast(value=1.0, draws=30, horizons=[1, 2])

    pooled = pool_forecasts(
        [fc0, fc1],
        weights=[0.25, 0.75],
        draws=2000,
        quantile_levels=[0.1, 0.9],
        rng=np.random.default_rng(0),
    )

    assert pooled.variables == ["y"]
    assert pooled.horizons == [1, 2]
    assert pooled.draws.shape == (2000, 2, 1)

    # Mixture mean should be close to 0.75 for both horizons
    assert pooled.mean.shape == (2, 1)
    assert float(pooled.mean[0, 0]) == pytest.approx(0.75, abs=0.03)
    assert float(pooled.mean[1, 0]) == pytest.approx(0.75, abs=0.03)

    assert set(pooled.quantiles.keys()) == {0.1, 0.9}
    assert float(pooled.quantiles[0.1][0, 0]) == 0.0
    assert float(pooled.quantiles[0.9][0, 0]) == 1.0


def test_pool_forecasts_defaults_quantiles_union() -> None:
    fc0 = _const_forecast(value=0.0, draws=5, horizons=[1])
    fc1 = _const_forecast(value=1.0, draws=5, horizons=[1])
    pooled = pool_forecasts([fc0, fc1], draws=100, rng=np.random.default_rng(0))
    assert set(pooled.quantiles.keys()) == {0.1, 0.9}


def test_pool_forecasts_raises_on_mismatch() -> None:
    fc0 = _const_forecast(value=0.0, draws=5, horizons=[1])
    fc1 = ForecastResult(
        variables=["x"], horizons=[1], draws=fc0.draws.copy(), mean=fc0.mean.copy(), quantiles={}
    )
    with pytest.raises(ValueError, match="variables"):
        pool_forecasts([fc0, fc1])


def test_pool_forecasts_latent_only_when_all_present() -> None:
    fc0 = _const_forecast(value=0.0, draws=5, horizons=[1])
    fc1 = _const_forecast(value=1.0, draws=5, horizons=[1])
    fc1 = ForecastResult(
        variables=fc1.variables,
        horizons=fc1.horizons,
        draws=fc1.draws,
        mean=fc1.mean,
        quantiles=fc1.quantiles,
        latent_draws=fc1.draws.copy(),
    )

    pooled = pool_forecasts([fc0, fc1], draws=100, rng=np.random.default_rng(0))
    assert pooled.latent_draws is None


def test_pool_forecasts_weights_validation() -> None:
    fc0 = _const_forecast(value=0.0, draws=5, horizons=[1])
    fc1 = _const_forecast(value=1.0, draws=5, horizons=[1])
    with pytest.raises(ValueError, match="weights"):
        pool_forecasts([fc0, fc1], weights=[0.1])
    with pytest.raises(ValueError, match="weights"):
        pool_forecasts([fc0, fc1], weights=[1.0, -1.0])
