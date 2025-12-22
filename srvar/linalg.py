from __future__ import annotations

import numpy as np
import scipy.linalg


def symmetrize(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("a must be a square 2D array")
    return 0.5 * (x + x.T)


def cholesky_jitter(a: np.ndarray, *, jitter: float = 1e-10, max_tries: int = 6) -> np.ndarray:
    x = symmetrize(np.asarray(a, dtype=float))
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("a must be a square 2D array")

    n = x.shape[0]
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(x)
        except np.linalg.LinAlgError:
            x = x + (jitter * (10.0**i)) * np.eye(n, dtype=float)

    raise np.linalg.LinAlgError("cholesky failed even after adding jitter")


def solve_psd(a: np.ndarray, b: np.ndarray, *, jitter: float = 1e-10, max_tries: int = 6) -> np.ndarray:
    x = symmetrize(np.asarray(a, dtype=float))
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("a must be a square 2D array")

    y = np.asarray(b, dtype=float)

    n = x.shape[0]
    for i in range(max_tries):
        try:
            c, lower = scipy.linalg.cho_factor(x, lower=True, check_finite=False)
            return scipy.linalg.cho_solve((c, lower), y, check_finite=False)
        except np.linalg.LinAlgError:
            x = x + (jitter * (10.0**i)) * np.eye(n, dtype=float)

    raise np.linalg.LinAlgError("solve_psd failed even after adding jitter")
