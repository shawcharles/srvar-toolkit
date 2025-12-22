from __future__ import annotations

import numpy as np


def gamma_rate(*, shape: float | np.ndarray, rate: float | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shp = np.asarray(shape, dtype=float)
    rte = np.asarray(rate, dtype=float)

    if np.any(~np.isfinite(shp)) or np.any(shp <= 0):
        raise ValueError("shape must be finite and > 0")
    if np.any(~np.isfinite(rte)) or np.any(rte <= 0):
        raise ValueError("rate must be finite and > 0")

    return rng.gamma(shape=shp, scale=1.0 / rte)


def inverse_gaussian(*, mu: float | np.ndarray, lam: float | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = np.asarray(mu, dtype=float)
    l = np.asarray(lam, dtype=float)

    if np.any(~np.isfinite(m)):
        raise ValueError("mu must be finite")
    if np.any(~np.isfinite(l)):
        raise ValueError("lam must be finite")

    m = np.clip(m, 1e-6, 1e6)
    l = np.clip(l, 1e-6, 1e6)

    v = rng.standard_normal(size=np.broadcast(m, l).shape) ** 2

    m_b = np.broadcast_to(m, v.shape)
    l_b = np.broadcast_to(l, v.shape)

    y = m_b + (m_b * m_b * v) / (2.0 * l_b) - (m_b / (2.0 * l_b)) * np.sqrt(4.0 * m_b * l_b * v + (m_b * m_b) * (v * v))

    u = rng.uniform(size=v.shape)
    out = np.where(u <= (m_b / (m_b + y)), y, (m_b * m_b) / y)

    out = np.clip(out, 1e-12, 1e12)
    return out
