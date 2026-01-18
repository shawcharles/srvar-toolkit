import math

import numpy as np
import pytest

from srvar.metrics import wis_draws


def test_wis_draws_perfect_forecast_zero() -> None:
    y = 1.5
    draws = np.full(101, y, dtype=float)
    assert wis_draws(y, draws, intervals=[0.5, 0.8, 0.9]) == 0.0


def test_wis_draws_point_mass_equals_absolute_error() -> None:
    y = 1.0
    draws = np.zeros(200, dtype=float)
    assert np.isclose(wis_draws(y, draws, intervals=[0.5, 0.9]), 1.0)


def test_wis_draws_empty_intervals_is_median_absolute_error() -> None:
    y = 2.0
    draws = np.zeros(101, dtype=float)
    assert np.isclose(wis_draws(y, draws, intervals=[]), 2.0)


def test_wis_draws_nan_returns_nan() -> None:
    assert math.isnan(wis_draws(float("nan"), np.zeros(10, dtype=float), intervals=[0.5]))
    assert math.isnan(wis_draws(0.0, np.array([0.0, np.nan]), intervals=[0.5]))


def test_wis_draws_invalid_intervals_raises() -> None:
    with pytest.raises(ValueError, match="intervals"):
        wis_draws(0.0, np.zeros(10, dtype=float), intervals=[1.0])
    with pytest.raises(ValueError, match="intervals"):
        wis_draws(0.0, np.zeros(10, dtype=float), intervals=[-0.1])
