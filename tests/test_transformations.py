import numpy as np
import pytest

from srvar.data.transformations import tcode_1d, transform_1d, transform_matrix


def test_transform_1d_level() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = transform_1d(x, "level")
    assert np.allclose(y, x)


def test_transform_1d_diff() -> None:
    x = np.array([1.0, 4.0, 9.0])
    y = transform_1d(x, "diff")
    assert np.allclose(y, np.array([3.0, 5.0]))


def test_transform_matrix_aligns_lengths() -> None:
    x = np.column_stack([
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 20.0, 40.0, 80.0]),
    ])
    y = transform_matrix(x, ["level", "diff"])
    assert y.shape == (3, 2)


def test_transform_unknown_raises() -> None:
    with pytest.raises(ValueError):
        transform_1d(np.array([1.0, 2.0]), "nope")


def test_tcode_1_level() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = tcode_1d(x, 1)
    assert np.allclose(y, x)


def test_tcode_2_first_diff() -> None:
    x = np.array([1.0, 3.0, 6.0, 10.0])
    y = tcode_1d(x, 2)
    assert np.isnan(y[0])
    assert np.allclose(y[1:], np.array([2.0, 3.0, 4.0]))


def test_tcode_3_second_diff() -> None:
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    y = tcode_1d(x, 3)
    assert np.all(np.isnan(y[:2]))
    assert np.allclose(y[2:], np.array([1.0, 1.0, 1.0]))


def test_tcode_4_log() -> None:
    x = np.array([1.0, np.e, np.e**2])
    y = tcode_1d(x, 4)
    assert np.allclose(y, np.array([0.0, 1.0, 2.0]))


def test_tcode_4_nonpositive_warns_and_nan() -> None:
    x = np.array([1.0, 0.0, -1.0, 2.0])
    with pytest.warns(RuntimeWarning):
        y = tcode_1d(x, 4, var_name="X")
    assert not np.isnan(y[0])
    assert np.isnan(y[1])
    assert np.isnan(y[2])
    assert not np.isnan(y[3])


def test_tcode_5_log_diff_scaled_100() -> None:
    x = np.array([100.0, 101.0, 102.01])
    y = tcode_1d(x, 5)
    assert np.isnan(y[0])
    assert np.isclose(y[1], 100.0 * np.log(101.0 / 100.0), rtol=1e-10)
    assert np.isclose(y[2], 100.0 * np.log(102.01 / 101.0), rtol=1e-10)


def test_tcode_6_log_second_diff_scaled_100() -> None:
    x = np.array([100.0, 110.0, 121.0, 133.1])
    y = tcode_1d(x, 6)
    assert np.all(np.isnan(y[:2]))
    expected = 100.0 * (np.log(121.0) - 2.0 * np.log(110.0) + np.log(100.0))
    assert np.isclose(y[2], expected, rtol=1e-10)
