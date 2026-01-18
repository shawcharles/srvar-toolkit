import numpy as np
import pytest

from srvar.stats import diebold_mariano_test, newey_west_long_run_variance


def test_newey_west_long_run_variance_zero_for_zero_series() -> None:
    x = np.zeros(10, dtype=float)
    assert newey_west_long_run_variance(x, max_lag=3) == 0.0


def test_diebold_mariano_identical_losses_stat_zero_p_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = diebold_mariano_test(a, b, horizon=1)
    assert res.nobs == 4
    assert res.mean_diff == 0.0
    assert res.long_run_variance == 0.0
    assert res.statistic == 0.0
    assert res.pvalue == 1.0


def test_diebold_mariano_constant_positive_diff() -> None:
    a = np.ones(20, dtype=float)
    b = np.zeros(20, dtype=float)
    res = diebold_mariano_test(a, b, horizon=1, alternative="two-sided")
    assert np.isposinf(res.statistic)
    assert res.pvalue == 0.0

    res_g = diebold_mariano_test(a, b, horizon=1, alternative="greater")
    assert res_g.pvalue == 0.0

    res_l = diebold_mariano_test(a, b, horizon=1, alternative="less")
    assert res_l.pvalue == 1.0


def test_diebold_mariano_constant_negative_diff() -> None:
    a = np.zeros(20, dtype=float)
    b = np.ones(20, dtype=float)
    res = diebold_mariano_test(a, b, horizon=1, alternative="two-sided")
    assert np.isneginf(res.statistic)
    assert res.pvalue == 0.0

    res_g = diebold_mariano_test(a, b, horizon=1, alternative="greater")
    assert res_g.pvalue == 1.0

    res_l = diebold_mariano_test(a, b, horizon=1, alternative="less")
    assert res_l.pvalue == 0.0


def test_diebold_mariano_default_lag_from_horizon() -> None:
    a = np.arange(10, dtype=float)
    b = np.zeros(10, dtype=float)
    res = diebold_mariano_test(a, b, horizon=3)
    assert res.max_lag == 2


def test_diebold_mariano_drops_nan_pairwise() -> None:
    a = np.array([0.0, 1.0, np.nan, 2.0])
    b = np.array([0.0, 0.0, 0.0, np.nan])
    res = diebold_mariano_test(a, b, horizon=1, small_sample_correction=False)
    assert res.nobs == 2


def test_diebold_mariano_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        diebold_mariano_test(np.zeros(3), np.zeros(4))


def test_diebold_mariano_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError, match="horizon"):
        diebold_mariano_test(np.zeros(3), np.zeros(3), horizon=0)
