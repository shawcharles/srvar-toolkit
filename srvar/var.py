from __future__ import annotations

import numpy as np


def lag_matrix(y: np.ndarray, p: int) -> np.ndarray:
    v = np.asarray(y, dtype=float)
    if v.ndim != 2:
        raise ValueError("y must be a 2D array of shape (T, N)")
    if p < 1:
        raise ValueError("p must be >= 1")

    t, n = v.shape
    if t <= p:
        raise ValueError("T must be > p")

    xlags = [v[p - lag : t - lag, :] for lag in range(1, p + 1)]
    return np.concatenate(xlags, axis=1)


def design_matrix(y: np.ndarray, p: int, *, include_intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(y, dtype=float)
    if v.ndim != 2:
        raise ValueError("y must be a 2D array of shape (T, N)")

    t, _n = v.shape
    xl = lag_matrix(v, p)
    yt = v[p:t, :]

    if include_intercept:
        x = np.concatenate([np.ones((xl.shape[0], 1), dtype=float), xl], axis=1)
    else:
        x = xl

    return x, yt


def companion_matrix(beta: np.ndarray, n: int, p: int, *, include_intercept: bool = True) -> np.ndarray:
    b = np.asarray(beta, dtype=float)

    k_expected = (1 if include_intercept else 0) + n * p
    if b.shape != (k_expected, n):
        raise ValueError("beta must have shape (K, N) with K = (intercept?1:0) + N*p")

    a_flat = b[1:, :].T if include_intercept else b.T
    top = a_flat

    if p == 1:
        return top

    eye = np.eye(n * (p - 1), dtype=float)
    bottom = np.concatenate([eye, np.zeros((n * (p - 1), n), dtype=float)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def is_stationary(beta: np.ndarray, n: int, p: int, *, include_intercept: bool = True, tol: float = 1e-10) -> bool:
    f = companion_matrix(beta, n=n, p=p, include_intercept=include_intercept)
    eigvals = np.linalg.eigvals(f)
    return bool(np.max(np.abs(eigvals)) < (1.0 - tol))
