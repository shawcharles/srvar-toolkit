import math

import numpy as np
import pytest

from srvar.metrics import log_score_draws


def test_log_score_draws_matches_standard_normal_at_mean() -> None:
    draws = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
    y = 0.0
    expected = -0.5 * math.log(2.0 * math.pi * 1.0)
    assert np.isclose(log_score_draws(y, draws, variance_floor=1e-12), expected)


def test_log_score_draws_matches_standard_normal_one_sd() -> None:
    draws = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
    y = 1.0
    expected = -0.5 * (math.log(2.0 * math.pi * 1.0) + 1.0)
    assert np.isclose(log_score_draws(y, draws, variance_floor=1e-12), expected)


def test_log_score_draws_variance_floor_applied() -> None:
    draws = np.zeros(10, dtype=float)
    y = 0.0
    var_floor = 0.25
    expected = -0.5 * math.log(2.0 * math.pi * var_floor)
    assert np.isclose(log_score_draws(y, draws, variance_floor=var_floor), expected)


def test_log_score_draws_nan_returns_nan() -> None:
    assert math.isnan(log_score_draws(float("nan"), np.zeros(10, dtype=float)))
    assert math.isnan(log_score_draws(0.0, np.array([0.0, np.nan]), variance_floor=1e-12))


def test_log_score_draws_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="draws must be non-empty"):
        log_score_draws(0.0, np.array([], dtype=float))

    with pytest.raises(ValueError, match="variance_floor"):
        log_score_draws(0.0, np.zeros(10, dtype=float), variance_floor=0.0)
