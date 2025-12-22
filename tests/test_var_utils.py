import numpy as np
import pytest

from srvar.var import companion_matrix, design_matrix, is_stationary, lag_matrix


def test_lag_matrix_shapes() -> None:
    y = np.arange(30, dtype=float).reshape(10, 3)
    x = lag_matrix(y, p=2)
    assert x.shape == (8, 6)


def test_design_matrix_intercept_shapes() -> None:
    y = np.arange(20, dtype=float).reshape(10, 2)
    x, yt = design_matrix(y, p=2, include_intercept=True)
    assert x.shape == (8, 1 + 4)
    assert yt.shape == (8, 2)


def test_companion_matrix_shapes_p2() -> None:
    n, p = 2, 2
    beta = np.zeros((1 + n * p, n), dtype=float)
    f = companion_matrix(beta, n=n, p=p, include_intercept=True)
    assert f.shape == (n * p, n * p)


def test_stationarity_trivial() -> None:
    n, p = 2, 2
    beta = np.zeros((1 + n * p, n), dtype=float)
    assert is_stationary(beta, n=n, p=p)


def test_lag_matrix_invalid_p() -> None:
    y = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError):
        lag_matrix(y, p=0)
