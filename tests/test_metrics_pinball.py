import math

import numpy as np
import pytest

from srvar.metrics import pinball_draws


def test_pinball_draws_perfect_forecast_zero() -> None:
    y = 1.5
    draws = np.full(101, y, dtype=float)
    assert pinball_draws(y, draws, quantiles=[0.1, 0.5, 0.9]) == 0.0


def test_pinball_draws_point_mass_median_is_half_abs_error() -> None:
    y = 1.0
    draws = np.zeros(200, dtype=float)
    assert np.isclose(pinball_draws(y, draws, quantiles=[0.5]), 0.5)


def test_pinball_draws_point_mass_three_quantiles_average() -> None:
    y = 1.0
    draws = np.zeros(200, dtype=float)
    # losses are [0.1, 0.5, 0.9] -> mean 0.5
    assert np.isclose(pinball_draws(y, draws, quantiles=[0.1, 0.5, 0.9]), 0.5)


def test_pinball_draws_nan_returns_nan() -> None:
    assert math.isnan(pinball_draws(float("nan"), np.zeros(10, dtype=float), quantiles=[0.5]))
    assert math.isnan(pinball_draws(0.0, np.array([0.0, np.nan]), quantiles=[0.5]))


def test_pinball_draws_invalid_quantiles_raise() -> None:
    with pytest.raises(ValueError, match="quantiles"):
        pinball_draws(0.0, np.zeros(10, dtype=float), quantiles=[])
    with pytest.raises(ValueError, match="quantiles"):
        pinball_draws(0.0, np.zeros(10, dtype=float), quantiles=[-0.1])
    with pytest.raises(ValueError, match="quantiles"):
        pinball_draws(0.0, np.zeros(10, dtype=float), quantiles=[1.1])
