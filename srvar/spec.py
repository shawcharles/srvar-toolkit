from __future__ import annotations

from dataclasses import dataclass

from .elb import ElbSpec
from .sv import VolatilitySpec
from .var import design_matrix

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelSpec:
    p: int
    include_intercept: bool = True
    elb: ElbSpec | None = None
    volatility: VolatilitySpec | None = None

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError("p must be >= 1")


@dataclass(frozen=True, slots=True)
class NIWPrior:
    m0: np.ndarray  # (K, N)
    v0: np.ndarray  # (K, K)
    s0: np.ndarray  # (N, N)
    nu0: float


@dataclass(frozen=True, slots=True)
class SSVSSpec:
    spike_var: float = 1e-4
    slab_var: float = 100.0
    inclusion_prob: float = 0.5
    intercept_slab_var: float | None = None
    fix_intercept: bool = True

    def __post_init__(self) -> None:
        if self.spike_var <= 0:
            raise ValueError("spike_var must be > 0")
        if self.slab_var <= 0:
            raise ValueError("slab_var must be > 0")
        if not (0.0 < self.inclusion_prob < 1.0):
            raise ValueError("inclusion_prob must be in (0, 1)")
        if self.intercept_slab_var is not None and self.intercept_slab_var <= 0:
            raise ValueError("intercept_slab_var must be > 0")


@dataclass(frozen=True, slots=True)
class PriorSpec:
    family: str
    niw: NIWPrior
    ssvs: SSVSSpec | None = None

    @staticmethod
    def niw_default(*, k: int, n: int) -> "PriorSpec":
        m0 = np.zeros((k, n), dtype=float)
        v0 = 10.0 * np.eye(k, dtype=float)
        s0 = np.eye(n, dtype=float)
        nu0 = float(n + 2)
        return PriorSpec(family="niw", niw=NIWPrior(m0=m0, v0=v0, s0=s0, nu0=nu0))

    @staticmethod
    def niw_minnesota(
        *,
        p: int,
        y: np.ndarray,
        n: int | None = None,
        include_intercept: bool = True,
        lambda1: float = 0.1,
        lambda2: float = 0.5,
        lambda3: float = 1.0,
        lambda4: float = 100.0,
        own_lag_means: np.ndarray | list[float] | None = None,
        own_lag_mean: float = 0.0,
        min_sigma2: float = 1e-12,
    ) -> "PriorSpec":
        v = np.asarray(y, dtype=float)
        if v.ndim != 2:
            raise ValueError("y must be a 2D array of shape (T, N)")
        t, n_y = v.shape

        if n is None:
            n = int(n_y)
        if int(n) != int(n_y):
            raise ValueError("n must match y.shape[1]")
        if p < 1:
            raise ValueError("p must be >= 1")
        if t <= p:
            raise ValueError("T must be > p")

        if lambda1 <= 0:
            raise ValueError("lambda1 must be > 0")
        if lambda2 <= 0:
            raise ValueError("lambda2 must be > 0")
        if lambda3 < 0:
            raise ValueError("lambda3 must be >= 0")
        if lambda4 <= 0:
            raise ValueError("lambda4 must be > 0")

        if own_lag_means is not None and own_lag_mean != 0.0:
            raise ValueError("specify at most one of own_lag_means and own_lag_mean")

        sigma2 = np.empty(n, dtype=float)
        for i in range(n):
            xi, yi = design_matrix(v[:, [i]], p, include_intercept=include_intercept)
            b, *_ = np.linalg.lstsq(xi, yi, rcond=None)
            resid = yi - xi @ b
            denom = max(int(resid.shape[0] - xi.shape[1]), 1)
            s2 = float((resid.T @ resid)[0, 0] / denom)
            sigma2[i] = max(s2, float(min_sigma2))

        k = (1 if include_intercept else 0) + n * p
        m0 = np.zeros((k, n), dtype=float)

        base = 1 if include_intercept else 0
        if own_lag_means is not None:
            olm = np.asarray(own_lag_means, dtype=float).reshape(-1)
            if olm.shape != (n,):
                raise ValueError("own_lag_means must have shape (N,)")
            for j in range(n):
                m0[base + j, j] = float(olm[j])
        elif own_lag_mean != 0.0:
            for j in range(n):
                m0[base + j, j] = float(own_lag_mean)

        v0 = np.zeros((k, k), dtype=float)
        if include_intercept:
            v0[0, 0] = float((lambda1 * lambda4) ** 2)

        if n == 1:
            cross_weight = 1.0
        else:
            cross_weight = float((1.0 + (n - 1) * (lambda2**2)) / n)

        base = 1 if include_intercept else 0
        for lag in range(1, p + 1):
            lag_scale = float((lambda1**2) / (lag ** (2.0 * lambda3)))
            for i in range(n):
                idx = base + (lag - 1) * n + i
                v0[idx, idx] = float(lag_scale * cross_weight / sigma2[i])

        s0 = np.diag(sigma2)
        nu0 = float(n + 2)
        return PriorSpec(family="niw", niw=NIWPrior(m0=m0, v0=v0, s0=s0, nu0=nu0))

    @staticmethod
    def ssvs(
        *,
        k: int,
        n: int,
        include_intercept: bool = True,
        m0: np.ndarray | None = None,
        s0: np.ndarray | None = None,
        nu0: float | None = None,
        spike_var: float = 1e-4,
        slab_var: float = 100.0,
        inclusion_prob: float = 0.5,
        intercept_slab_var: float | None = None,
        fix_intercept: bool = True,
    ) -> "PriorSpec":
        if k < 1:
            raise ValueError("k must be >= 1")
        if n < 1:
            raise ValueError("n must be >= 1")

        if m0 is None:
            m0a = np.zeros((k, n), dtype=float)
        else:
            m0a = np.asarray(m0, dtype=float)
            if m0a.shape != (k, n):
                raise ValueError("m0 must have shape (K, N)")

        if s0 is None:
            s0a = np.eye(n, dtype=float)
        else:
            s0a = np.asarray(s0, dtype=float)
            if s0a.shape != (n, n):
                raise ValueError("s0 must have shape (N, N)")

        nu0a = float(n + 2) if nu0 is None else float(nu0)

        niw = NIWPrior(
            m0=m0a,
            v0=np.eye(k, dtype=float),
            s0=s0a,
            nu0=nu0a,
        )
        spec = SSVSSpec(
            spike_var=float(spike_var),
            slab_var=float(slab_var),
            inclusion_prob=float(inclusion_prob),
            intercept_slab_var=None if intercept_slab_var is None else float(intercept_slab_var),
            fix_intercept=bool(fix_intercept and include_intercept),
        )
        return PriorSpec(family="ssvs", niw=niw, ssvs=spec)


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    draws: int = 2000
    burn_in: int = 500
    thin: int = 1

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError("draws must be >= 1")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.thin < 1:
            raise ValueError("thin must be >= 1")
