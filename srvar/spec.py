from __future__ import annotations

from dataclasses import dataclass

from .elb import ElbSpec
from .sv import VolatilitySpec
from .var import design_matrix

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model configuration for VAR/SRVAR estimation.

    Parameters
    ----------
    p:
        VAR lag order.
    include_intercept:
        Whether to include an intercept term in the VAR design matrix.
    elb:
        Optional effective-lower-bound (ELB) configuration. When enabled, specified
        variables are treated as censored at the bound and latent shadow values are
        sampled during estimation.
    volatility:
        Optional stochastic volatility configuration. When enabled, the model uses a
        diagonal stochastic volatility random-walk (SVRW) specification.

    Notes
    -----
    ``ModelSpec`` is intentionally small and immutable. The heavy lifting is done in
    :func:`srvar.api.fit`.
    """
    p: int
    include_intercept: bool = True
    elb: ElbSpec | None = None
    volatility: VolatilitySpec | None = None

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError("p must be >= 1")


@dataclass(frozen=True, slots=True)
class NIWPrior:
    """Normal-Inverse-Wishart (NIW) prior parameters.

    This prior is used for conjugate Bayesian VAR estimation.

    Notes
    -----
    Shapes follow the conventions used throughout the toolkit:

    - ``m0`` has shape ``(K, N)`` where ``K = (1 if include_intercept else 0) + N * p``
    - ``v0`` has shape ``(K, K)``
    - ``s0`` has shape ``(N, N)``
    """
    m0: np.ndarray  # (K, N)
    v0: np.ndarray  # (K, K)
    s0: np.ndarray  # (N, N)
    nu0: float


@dataclass(frozen=True, slots=True)
class SSVSSpec:
    """Hyperparameters for stochastic search variable selection (SSVS).

    The SSVS implementation uses a spike-and-slab Gaussian prior over coefficient
    rows (predictor-specific inclusion indicators).

    Parameters
    ----------
    spike_var:
        Prior variance for excluded predictors (spike component).
    slab_var:
        Prior variance for included predictors (slab component).
    inclusion_prob:
        Prior inclusion probability for each predictor.
    intercept_slab_var:
        Optional slab variance override for the intercept term.
    fix_intercept:
        If True and an intercept is included in the model, the intercept is always
        included.
    """
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
    """Prior specification wrapper.

    This object selects the prior family via ``family`` and carries the required
    parameter blocks.

    Parameters
    ----------
    family:
        Prior family identifier. Currently supported values are ``"niw"`` and
        ``"ssvs"``.
    niw:
        NIW parameter block. This is required for both families because SSVS uses
        a NIW-like structure with a modified coefficient covariance ``V0``.
    ssvs:
        Optional SSVS hyperparameters (required when ``family='ssvs'``).
    """
    family: str
    niw: NIWPrior
    ssvs: SSVSSpec | None = None

    @staticmethod
    def niw_default(*, k: int, n: int) -> "PriorSpec":
        """Construct a simple default NIW prior.

        This uses a zero prior mean for coefficients and relatively weak
        regularization.

        Parameters
        ----------
        k:
            Number of regressors ``K``.
        n:
            Number of endogenous variables ``N``.
        """
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
        """Construct an NIW prior with Minnesota-style shrinkage.

        The Minnesota prior shrinks coefficients toward a random-walk/white-noise
        baseline, with lag-decay and cross-variable shrinkage controlled by
        ``lambda1..lambda4``.

        Parameters
        ----------
        p:
            VAR lag order.
        y:
            Data array used to estimate scaling variances (T, N).
        include_intercept:
            Whether the resulting prior is compatible with a VAR that includes an
            intercept.
        lambda1, lambda2, lambda3, lambda4:
            Standard Minnesota hyperparameters.
        own_lag_means / own_lag_mean:
            Optional prior mean(s) for own first lag.
        min_sigma2:
            Floor for estimated variances used in scaling.

        Returns
        -------
        PriorSpec
            A ``PriorSpec`` instance with ``family='niw'``.
        """
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
    def from_ssvs(
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
        """Construct a prior specification for SSVS estimation.

        Parameters
        ----------
        k:
            Number of regressors ``K``.
        n:
            Number of endogenous variables ``N``.
        include_intercept:
            Whether the corresponding model includes an intercept.
        m0, s0, nu0:
            Optional NIW blocks. If omitted, sensible defaults are used.
        spike_var, slab_var, inclusion_prob:
            SSVS hyperparameters.
        intercept_slab_var:
            Optional slab variance override for the intercept coefficient row.
        fix_intercept:
            Whether to force inclusion of the intercept row.
        """
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
    """MCMC configuration for Gibbs samplers.

    Parameters
    ----------
    draws:
        Total number of iterations to run.
    burn_in:
        Number of initial iterations to discard.
    thin:
        Keep every ``thin``-th draw after burn-in.

    Notes
    -----
    For conjugate NIW estimation (no ELB/SV), draws are sampled directly and then
    burn-in/thinning is applied post hoc. For iterative samplers (ELB, SSVS, SV),
    burn-in/thinning is applied online.
    """
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
