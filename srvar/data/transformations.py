from __future__ import annotations

import numpy as np


def transform_1d(x: np.ndarray, code: str) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        raise ValueError("x must be 1D")

    c = code.lower()

    if c == "level":
        return v.copy()
    if c == "diff":
        return np.diff(v, n=1)
    if c == "log":
        return np.log(v)
    if c == "logdiff":
        return np.diff(np.log(v), n=1)
    if c == "pct":
        return 100.0 * (v[1:] / v[:-1] - 1.0)

    raise ValueError(f"unknown transform code: {code}")


def transform_matrix(x: np.ndarray, codes: list[str]) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.ndim != 2:
        raise ValueError("x must be a 2D array")

    if len(codes) != v.shape[1]:
        raise ValueError("len(codes) must equal x.shape[1]")

    cols = [transform_1d(v[:, j], codes[j]) for j in range(v.shape[1])]
    t_min = min(c.shape[0] for c in cols)
    cols = [c[-t_min:] for c in cols]
    return np.column_stack(cols)


def tcode_1d(x: np.ndarray, tcode: int | float, *, var_name: str | None = None) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        raise ValueError("x must be 1D")

    tc = int(tcode)
    if float(tcode) != float(tc):
        raise ValueError("tcode must be an integer")

    if tc == 1:
        return v.copy()

    if tc == 2:
        out = np.full_like(v, np.nan, dtype=float)
        out[1:] = v[1:] - v[:-1]
        return out

    logv = np.where(v > 0.0, np.log(v), np.nan)

    if tc == 4:
        return logv

    if tc == 5:
        out = np.full_like(v, np.nan, dtype=float)
        out[1:] = 100.0 * (logv[1:] - logv[:-1])
        if var_name is not None and var_name.upper() == "UNRATE":
            out = out / 100.0
        return out

    if tc == 6:
        out = np.full_like(v, np.nan, dtype=float)
        out[2:] = 100.0 * (logv[2:] - 2.0 * logv[1:-1] + logv[:-2])
        return out

    raise ValueError(f"unknown tcode: {tcode}")


def tcode_matrix(x: np.ndarray, tcodes: list[int | float], *, var_names: list[str] | None = None) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.ndim != 2:
        raise ValueError("x must be a 2D array")
    if len(tcodes) != v.shape[1]:
        raise ValueError("len(tcodes) must equal x.shape[1]")

    if var_names is not None and len(var_names) != v.shape[1]:
        raise ValueError("len(var_names) must equal x.shape[1]")

    cols: list[np.ndarray] = []
    for j in range(v.shape[1]):
        name = None if var_names is None else var_names[j]
        cols.append(tcode_1d(v[:, j], tcodes[j], var_name=name))
    return np.column_stack(cols)
