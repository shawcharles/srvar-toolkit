import numpy as np
import pytest

from srvar.data.transformations import transform_1d, transform_matrix


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
