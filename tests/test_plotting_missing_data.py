import numpy as np

from srvar.plotting import _coverage_curve, _crps_curve, _pit_values
from srvar.results import ForecastResult


def _forecast(draws: np.ndarray) -> ForecastResult:
    sims = np.asarray(draws, dtype=float)
    return ForecastResult(
        variables=["y1", "y2"],
        horizons=[1],
        draws=sims,
        mean=sims.mean(axis=0),
        quantiles={},
    )


def test_coverage_curve_uses_only_finite_cells_when_averaging_across_variables() -> None:
    forecasts = [_forecast(np.zeros((4, 1, 2))), _forecast(np.zeros((4, 1, 2)))]
    y_true = np.asarray([[[0.0, 1.0]], [[np.nan, 2.0]]], dtype=float)

    curve = _coverage_curve(
        forecasts,
        y_true,
        horizon_indices=[0],
        intervals=[0.5],
        var_index=None,
        use_latent=False,
    )

    assert np.isclose(curve[0.5][0], 1.0 / 3.0)


def test_pit_values_drop_missing_realizations() -> None:
    forecasts = [_forecast(np.zeros((4, 1, 2))), _forecast(np.zeros((4, 1, 2)))]
    y_true = np.asarray([[[0.0, 0.0]], [[np.nan, 0.0]]], dtype=float)

    values = _pit_values(
        forecasts,
        y_true,
        horizon_index=0,
        var_index=0,
        use_latent=False,
    )

    assert np.array_equal(values, np.asarray([1.0]))


def test_crps_curve_drops_missing_realizations_in_aggregate_series() -> None:
    forecasts = [_forecast(np.zeros((4, 1, 2))), _forecast(np.zeros((4, 1, 2)))]
    y_true = np.asarray([[[0.0, 1.0]], [[np.nan, 2.0]]], dtype=float)

    curve = _crps_curve(
        forecasts,
        y_true,
        horizon_indices=[0],
        var_index=None,
        use_latent=False,
    )

    assert np.isclose(curve[0], 1.0)
