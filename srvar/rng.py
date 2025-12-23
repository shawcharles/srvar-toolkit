from __future__ import annotations

import numpy as np
import scipy.stats


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


def gig_rvs(
    *,
    p: float | np.ndarray,
    a: float | np.ndarray,
    b: float | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from the generalized inverse Gaussian distribution GIG(p, a, b).

    This implementation matches the three-parameter GIG(p, a, b) used in the MATLAB
    reference code (see ``functions/gigrnd.m``), with density:

        f(x) ∝ x^(p-1) * exp(-(a*x + b/x)/2),  x > 0.

    SciPy exposes a two-parameter form via ``scipy.stats.geninvgauss(p, omega)``,
    with density:

        f(z) ∝ z^(p-1) * exp(-(omega*z + omega/z)/2),  z > 0,

    which corresponds to the special case GIG(p, omega, omega).

    To sample from GIG(p, a, b), we use the scaling identity:

        Let omega = sqrt(a*b) and Z ~ GIG(p, omega, omega).
        Then X = sqrt(b/a) * Z ~ GIG(p, a, b).

    Notes
    -----
    - The DL prior update step uses GIG draws for both the global scale and the
      Dirichlet weights; any parameterization mismatch here would materially
      change the implied shrinkage.
    """
    pp = np.asarray(p, dtype=float)
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)

    if np.any(~np.isfinite(pp)):
        raise ValueError("p must be finite")
    if np.any(~np.isfinite(aa)) or np.any(aa <= 0):
        raise ValueError("a must be finite and > 0")
    if np.any(~np.isfinite(bb)) or np.any(bb <= 0):
        raise ValueError("b must be finite and > 0")

    shp = np.broadcast(pp, aa, bb).shape
    pp_b = np.broadcast_to(pp, shp).reshape(-1)
    aa_b = np.broadcast_to(aa, shp).reshape(-1)
    bb_b = np.broadcast_to(bb, shp).reshape(-1)

    out = np.empty(pp_b.shape[0], dtype=float)
    for i, (pi, ai, bi) in enumerate(zip(pp_b, aa_b, bb_b, strict=True)):
        omega = float(np.sqrt(ai * bi))
        z = scipy.stats.geninvgauss(pi, omega).rvs(random_state=rng)
        out[i] = float((bi / omega) * z)

    return out.reshape(shp)
