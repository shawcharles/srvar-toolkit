import numpy as np
import pytest

from srvar.linalg import solve_psd, symmetrize


def test_symmetrize() -> None:
    a = np.array([[1.0, 2.0], [0.0, 3.0]])
    s = symmetrize(a)
    assert np.allclose(s, np.array([[1.0, 1.0], [1.0, 3.0]]))


def test_solve_psd_matches_direct_solve() -> None:
    a = np.array([[2.0, 0.2], [0.2, 1.0]])
    b = np.array([1.0, 2.0])
    x = solve_psd(a, b)
    assert np.allclose(a @ x, b)


def test_solve_psd_non_square_raises() -> None:
    with pytest.raises(ValueError):
        solve_psd(np.zeros((2, 3)), np.zeros(2))
