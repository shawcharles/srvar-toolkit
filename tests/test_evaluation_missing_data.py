import math

import numpy as np

from srvar.evaluation import MetricsAccumulator, compute_metrics_rows
from srvar.results import ForecastResult


def _evaluation_config() -> dict:
    return {
        "coverage": {"enabled": True, "intervals": [0.5], "use_latent": False},
        "crps": {"enabled": True, "use_latent": False},
        "wis": {"enabled": False, "intervals": [], "use_latent": False},
        "pinball": {"enabled": False, "quantiles": [], "use_latent": False},
        "log_score": {"enabled": False, "variance_floor": 1e-12, "use_latent": False},
    }


def _forecast(draws: np.ndarray) -> ForecastResult:
    sims = np.asarray(draws, dtype=float)
    return ForecastResult(
        variables=["y"],
        horizons=[1],
        draws=sims,
        mean=sims.mean(axis=0),
        quantiles={},
    )


def test_compute_metrics_rows_excludes_missing_realizations_from_coverage() -> None:
    forecasts = [_forecast(np.zeros((4, 1, 1))), _forecast(np.zeros((4, 1, 1)))]
    y_true = np.asarray([[[0.0]], [[np.nan]]], dtype=float)

    rows = compute_metrics_rows(
        forecasts=forecasts,
        y_true=y_true,
        variables=["y"],
        evaluation=_evaluation_config(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["variable"] == "y"
    assert row["horizon"] == 1
    assert float(row["crps"]) == 0.0
    assert float(row["rmse"]) == 0.0
    assert float(row["mae"]) == 0.0
    assert float(row["coverage_50"]) == 1.0


def test_metrics_accumulator_excludes_missing_realizations_from_coverage() -> None:
    acc = MetricsAccumulator(variables=["y"], max_h=1, evaluation=_evaluation_config())
    acc.update(forecast=_forecast(np.zeros((4, 1, 1))), y_true=np.asarray([[0.0]], dtype=float))
    acc.update(forecast=_forecast(np.zeros((4, 1, 1))), y_true=np.asarray([[np.nan]], dtype=float))

    rows = acc.rows()

    assert len(rows) == 1
    row = rows[0]
    assert float(row["crps"]) == 0.0
    assert float(row["rmse"]) == 0.0
    assert float(row["mae"]) == 0.0
    assert float(row["coverage_50"]) == 1.0


def test_metrics_rows_return_nan_when_no_finite_coverage_observations_exist() -> None:
    rows = compute_metrics_rows(
        forecasts=[_forecast(np.zeros((4, 1, 1)))],
        y_true=np.asarray([[[np.nan]]], dtype=float),
        variables=["y"],
        evaluation=_evaluation_config(),
    )

    assert len(rows) == 1
    assert math.isnan(float(rows[0]["coverage_50"]))
