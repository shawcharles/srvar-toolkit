from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .elb import ElbSpec
from .sv import VolatilitySpec
from .var import design_matrix


@dataclass(frozen=True, slots=True)
class MuSSVSSpec:
    spike_var: float = 1e-4
    slab_var: float = 100.0
    inclusion_prob: float = 0.5

    def __post_init__(self) -> None:
        if self.spike_var <= 0 or not np.isfinite(self.spike_var):
            raise ValueError("spike_var must be finite and > 0")
        if self.slab_var <= 0 or not np.isfinite(self.slab_var):
            raise ValueError("slab_var must be finite and > 0")
        if not (0.0 < float(self.inclusion_prob) < 1.0):
            raise ValueError("inclusion_prob must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SteadyStateSpec:
    mu0: np.ndarray  # (N,)
    v0_mu: float | np.ndarray
    ssvs: MuSSVSSpec | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.mu0, dtype=float).reshape(-1)
        if m.ndim != 1 or m.shape[0] < 1:
            raise ValueError("mu0 must be a 1D array of shape (N,)")
        if np.any(~np.isfinite(m)):
            raise ValueError("mu0 must be finite")

        if isinstance(self.v0_mu, (float, int, np.floating, np.integer)) and not isinstance(
            self.v0_mu, bool
        ):
            v_scalar = float(self.v0_mu)
            if not np.isfinite(v_scalar) or v_scalar <= 0:
                raise ValueError("v0_mu must be finite and > 0")
        else:
            v_vec = np.asarray(self.v0_mu, dtype=float).reshape(-1)
            if v_vec.shape != (m.shape[0],):
                raise ValueError("v0_mu must be a scalar or have shape (N,)")
            if np.any(~np.isfinite(v_vec)) or np.any(v_vec <= 0):
                raise ValueError("v0_mu must be finite and > 0")


@dataclass(frozen=True, slots=True)
class ShockSpec:
    """Innovation (shock) distribution configuration.

    Parameters
    ----------
    family:
        Shock family identifier.

        - ``"gaussian"``: standard Gaussian innovations.
        - ``"student_t"``: multivariate Student-t innovations via a scale-mixture of normals.
        - ``"mixture_outlier"``: two-component Gaussian mixture with an outlier variance inflation.
    df:
        Degrees of freedom for ``family="student_t"``. Must be finite and > 2.
    outlier_prob:
        Outlier probability for ``family="mixture_outlier"``. Must be in (0, 1).
    outlier_variance:
        Variance inflation factor for outliers in ``family="mixture_outlier"``. Must be finite
        and > 1.0. When an outlier occurs, innovations have covariance ``outlier_variance * Sigma``.
    """

    family: Literal["gaussian", "student_t", "mixture_outlier"] = "gaussian"
    df: float = 7.0
    outlier_prob: float = 0.05
    outlier_variance: float = 10.0

    def __post_init__(self) -> None:
        fam = str(self.family).lower()
        if fam not in {"gaussian", "student_t", "mixture_outlier"}:
            raise ValueError("family must be one of: gaussian, student_t, mixture_outlier")

        if fam == "student_t":
            nu = float(self.df)
            if not np.isfinite(nu) or nu <= 2.0:
                raise ValueError("df must be finite and > 2 for student_t")

        if fam == "mixture_outlier":
            p = float(self.outlier_prob)
            if not (0.0 < p < 1.0):
                raise ValueError("outlier_prob must be in (0, 1) for mixture_outlier")
            kappa = float(self.outlier_variance)
            if not np.isfinite(kappa) or kappa <= 1.0:
                raise ValueError("outlier_variance must be finite and > 1 for mixture_outlier")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model configuration for VAR/SRVAR estimation.

    Parameters
    ----------
    p:
        VAR lag order.
    include_intercept:
        Whether to include an intercept term in the VAR design matrix.
    steady_state:
        Optional steady-state configuration.
    elb:
        Optional effective-lower-bound (ELB) configuration. When enabled, specified
        variables are treated as censored at the bound and latent shadow values are
        sampled during estimation.
    volatility:
        Optional stochastic volatility configuration. When enabled, the model uses a
        stochastic volatility specification for time-varying variances (random-walk or
        AR(1) dynamics). Residual covariance can be diagonal (independent shocks) or a
        triangular factorization with time-invariant correlations.

    Notes
    -----
    ``ModelSpec`` is intentionally small and immutable. The heavy lifting is done in
    :func:`srvar.api.fit`.
    """

    p: int
    include_intercept: bool = True
    steady_state: SteadyStateSpec | None = None
    elb: ElbSpec | None = None
    volatility: VolatilitySpec | None = None
    shocks: ShockSpec | None = None

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError("p must be >= 1")

        if self.steady_state is not None and not self.include_intercept:
            raise ValueError("steady_state requires include_intercept=True")


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
class BLassoSpec:
    mode: Literal["global", "adaptive"] = "global"
    a0_global: float = 1.0
    b0_global: float = 1.0
    a0_c: float = 1.0
    b0_c: float = 1.0
    a0_L: float = 1.0
    b0_L: float = 1.0
    tau_init: float = 1e4
    lambda_init: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"global", "adaptive"}:
            raise ValueError("mode must be one of: global, adaptive")
        for name in [
            "a0_global",
            "b0_global",
            "a0_c",
            "b0_c",
            "a0_L",
            "b0_L",
            "tau_init",
            "lambda_init",
        ]:
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0")


@dataclass(frozen=True, slots=True)
class DLSpec:
    """Hyperparameters for the Dirichlet–Laplace (DL) shrinkage prior.

    This is a global–local shrinkage prior implemented to match the MATLAB reference
    code in ``functions/parfor_vintages.m`` (model mnemonic "DirLap").

    Parameters
    ----------
    abeta:
        Dirichlet concentration parameter (named ``abeta`` in the MATLAB code).
    dl_scaler:
        Small positive initialization scale used for the latent DL variables
        (named ``DL_scaler`` in the MATLAB code).
    """

    abeta: float = 0.5
    dl_scaler: float = 0.1

    def __post_init__(self) -> None:
        for name in ["abeta", "dl_scaler"]:
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0")


@dataclass(frozen=True, slots=True)
class MinnesotaCanonicalSpec:
    """Equation-wise canonical Minnesota prior metadata.

    Parameters
    ----------
    sigma2:
        Scale variances used in the Minnesota construction (length ``N``).
    inv_v0_vec:
        Equation-wise prior precision vector over ``vec(beta)`` in Fortran / column-major order.
        The block of ``K`` coefficients for equation ``j`` occupies
        ``inv_v0_vec[j*K:(j+1)*K]``.
    mode:
        Equation-wise Minnesota variant label. ``"canonical"`` is the exact canonical
        construction; ``"tempered"`` is the experimental geometric bridge between the
        legacy and canonical variance maps.
    tempered_alpha:
        Blend weight for ``mode="tempered"``. ``0.0`` reproduces the legacy coefficient
        variances and ``1.0`` reproduces the canonical coefficient variances.
    """

    sigma2: np.ndarray
    inv_v0_vec: np.ndarray
    mode: Literal["canonical", "tempered"] = "canonical"
    tempered_alpha: float | None = None

    def __post_init__(self) -> None:
        sigma2 = np.asarray(self.sigma2, dtype=float).reshape(-1)
        inv_v0_vec = np.asarray(self.inv_v0_vec, dtype=float).reshape(-1)
        if sigma2.shape[0] < 1:
            raise ValueError("sigma2 must be a non-empty 1D array")
        if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
            raise ValueError("sigma2 must be finite and > 0")
        if inv_v0_vec.shape[0] < 1:
            raise ValueError("inv_v0_vec must be a non-empty 1D array")
        if np.any(~np.isfinite(inv_v0_vec)) or np.any(inv_v0_vec <= 0):
            raise ValueError("inv_v0_vec must be finite and > 0")
        mode = str(self.mode).lower()
        if mode not in {"canonical", "tempered"}:
            raise ValueError("mode must be one of: canonical, tempered")
        if mode == "canonical":
            if self.tempered_alpha is not None:
                raise ValueError("tempered_alpha must be omitted when mode='canonical'")
            return
        if self.tempered_alpha is None:
            raise ValueError("tempered_alpha is required when mode='tempered'")
        alpha = float(self.tempered_alpha)
        if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise ValueError("tempered_alpha must be finite and in [0, 1]")


def _validate_minnesota_hyperparameters(
    *,
    p: int,
    y: np.ndarray,
    n: int | None,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    lambda4: float,
    own_lag_means: np.ndarray | list[float] | None,
    own_lag_mean: float,
) -> tuple[np.ndarray, int]:
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

    return v, int(n)


def _estimate_minnesota_sigma2(
    *,
    y: np.ndarray,
    p: int,
    include_intercept: bool,
    min_sigma2: float,
) -> np.ndarray:
    v = np.asarray(y, dtype=float)
    n = int(v.shape[1])
    sigma2 = np.empty(n, dtype=float)
    for i in range(n):
        xi, yi = design_matrix(v[:, [i]], p, include_intercept=include_intercept)
        b, *_ = np.linalg.lstsq(xi, yi, rcond=None)
        resid = yi - xi @ b
        denom = max(int(resid.shape[0] - xi.shape[1]), 1)
        s2 = float((resid.T @ resid)[0, 0] / denom)
        sigma2[i] = max(s2, float(min_sigma2))
    return sigma2


def _build_minnesota_prior_mean(
    *,
    n: int,
    p: int,
    include_intercept: bool,
    own_lag_means: np.ndarray | list[float] | None,
    own_lag_mean: float,
) -> np.ndarray:
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

    return m0


def _canonical_minnesota_variances(
    *,
    n: int,
    p: int,
    include_intercept: bool,
    sigma2: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    lambda4: float,
) -> np.ndarray:
    sigma2_arr = np.asarray(sigma2, dtype=float).reshape(-1)
    if sigma2_arr.shape != (n,):
        raise ValueError("sigma2 must have shape (N,)")

    k = (1 if include_intercept else 0) + n * p
    var = np.zeros((k, n), dtype=float)
    if include_intercept:
        var[0, :] = float((lambda1 * lambda4) ** 2) * sigma2_arr

    base = 1 if include_intercept else 0
    for lag in range(1, p + 1):
        lag_scale = float((lambda1**2) / (lag ** (2.0 * lambda3)))
        for pred_idx in range(n):
            row = base + (lag - 1) * n + pred_idx
            for eq_idx in range(n):
                if pred_idx == eq_idx:
                    var[row, eq_idx] = lag_scale
                else:
                    var[row, eq_idx] = (
                        lag_scale * (lambda2**2) * sigma2_arr[eq_idx] / sigma2_arr[pred_idx]
                    )
    return var


def _tempered_minnesota_variances(
    *,
    legacy_var: np.ndarray,
    canonical_var: np.ndarray,
    alpha: float,
) -> np.ndarray:
    legacy_var_arr = np.asarray(legacy_var, dtype=float)
    canonical_var_arr = np.asarray(canonical_var, dtype=float)
    if legacy_var_arr.shape != canonical_var_arr.shape:
        raise ValueError("legacy_var and canonical_var must have the same shape")
    if np.any(~np.isfinite(legacy_var_arr)) or np.any(legacy_var_arr <= 0):
        raise ValueError("legacy_var must be finite and > 0")
    if np.any(~np.isfinite(canonical_var_arr)) or np.any(canonical_var_arr <= 0):
        raise ValueError("canonical_var must be finite and > 0")
    alpha_f = float(alpha)
    if not np.isfinite(alpha_f) or not (0.0 <= alpha_f <= 1.0):
        raise ValueError("alpha must be finite and in [0, 1]")
    return legacy_var_arr * np.power(canonical_var_arr / legacy_var_arr, alpha_f)


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """Prior specification wrapper.

    This object selects the prior family via ``family`` and carries the required
    parameter blocks.

    Parameters
    ----------
    family:
        Prior family identifier. Supported values include ``"niw"``, ``"ssvs"``,
        ``"blasso"``, and ``"dl"``.
    niw:
        NIW parameter block. This is required for both families because SSVS uses
        a NIW-like structure with a modified coefficient covariance ``V0``.
    ssvs:
        Optional SSVS hyperparameters (required when ``family='ssvs'``).
    blasso:
        Optional Bayesian Lasso hyperparameters (required when ``family='blasso'``).
    dl:
        Optional Dirichlet–Laplace hyperparameters (required when ``family='dl'``).
    minnesota_canonical:
        Optional equation-wise Minnesota metadata. When present, the prior is fit
        via the explicit equation-wise Minnesota path rather than the legacy NIW
        matrix-normal path. ``mode="canonical"`` is the exact canonical
        construction; ``mode="tempered"`` is the experimental bridge between the
        legacy and canonical variance maps.
    """

    family: str
    niw: NIWPrior
    ssvs: SSVSSpec | None = None
    blasso: BLassoSpec | None = None
    dl: DLSpec | None = None
    minnesota_canonical: MinnesotaCanonicalSpec | None = None

    @staticmethod
    def niw_default(*, k: int, n: int) -> PriorSpec:
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
    ) -> PriorSpec:
        """Backward-compatible alias for the legacy Minnesota-style NIW prior.

        This method preserves the historical ``srvar-toolkit`` behaviour. For an
        explicit reproducibility path, prefer
        :meth:`PriorSpec.niw_minnesota_legacy`.

        Notes
        -----
        The current implementation is a legacy Minnesota-style NIW construction,
        not a full canonical Minnesota prior. In particular, it uses one scalar
        cross-variable shrinkage weight derived from ``lambda2`` rather than
        equation-specific own-vs-cross shrinkage.
        """
        return PriorSpec.niw_minnesota_legacy(
            p=p,
            y=y,
            n=n,
            include_intercept=include_intercept,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
            min_sigma2=min_sigma2,
        )

    @staticmethod
    def niw_minnesota_legacy(
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
    ) -> PriorSpec:
        """Construct the legacy Minnesota-style NIW prior used by this toolkit.

        This preserves the historical ``srvar-toolkit`` prior semantics. It is a
        Minnesota-style NIW construction with lag decay and a scalar
        cross-variable shrinkage weight, but it is not a full canonical
        Minnesota prior.

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
            Legacy Minnesota-style hyperparameters. ``lambda2`` is collapsed into
            a scalar cross-variable weight shared across equations.
        own_lag_means / own_lag_mean:
            Optional prior mean(s) for own first lag.
        min_sigma2:
            Floor for estimated variances used in scaling.

        Returns
        -------
        PriorSpec
            A ``PriorSpec`` instance with ``family='niw'``.
        """
        v, n = _validate_minnesota_hyperparameters(
            p=p,
            y=y,
            n=n,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
        )
        sigma2 = _estimate_minnesota_sigma2(
            y=v,
            p=p,
            include_intercept=include_intercept,
            min_sigma2=min_sigma2,
        )

        k = (1 if include_intercept else 0) + n * p
        m0 = _build_minnesota_prior_mean(
            n=n,
            p=p,
            include_intercept=include_intercept,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
        )

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
    def niw_minnesota_canonical(
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
    ) -> PriorSpec:
        """Construct an explicit canonical Minnesota prior on the equation-wise path.

        This constructor preserves the standard own-vs-cross variance distinction by
        storing per-equation prior precisions rather than collapsing everything into a
        shared NIW row covariance. The resulting fit path is currently supported for
        homoskedastic models and diagonal stochastic-volatility models.

        Notes
        -----
        The exact equation-wise coefficient prior is carried in
        ``prior.minnesota_canonical.inv_v0_vec``. The companion ``prior.niw.v0``
        block is a row-averaged diagonal summary kept for shape compatibility with the
        rest of the toolkit.
        """
        v, n = _validate_minnesota_hyperparameters(
            p=p,
            y=y,
            n=n,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
        )
        sigma2 = _estimate_minnesota_sigma2(
            y=v,
            p=p,
            include_intercept=include_intercept,
            min_sigma2=min_sigma2,
        )
        var = _canonical_minnesota_variances(
            n=n,
            p=p,
            include_intercept=include_intercept,
            sigma2=sigma2,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
        )
        m0 = _build_minnesota_prior_mean(
            n=n,
            p=p,
            include_intercept=include_intercept,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
        )

        # The exact canonical prior is equation-wise. Keep a row-average summary in
        # NIW.v0 for compatibility with code that expects a (K, K) block.
        v0_summary = np.diag(np.mean(var, axis=1))
        niw = NIWPrior(
            m0=m0,
            v0=v0_summary,
            s0=np.diag(sigma2),
            nu0=2.0,
        )
        canonical = MinnesotaCanonicalSpec(
            sigma2=sigma2,
            inv_v0_vec=1.0 / var.reshape(-1, order="F"),
        )
        return PriorSpec(
            family="niw",
            niw=niw,
            minnesota_canonical=canonical,
        )

    @staticmethod
    def niw_minnesota_tempered(
        *,
        p: int,
        y: np.ndarray,
        n: int | None = None,
        include_intercept: bool = True,
        lambda1: float = 0.1,
        lambda2: float = 0.5,
        lambda3: float = 1.0,
        lambda4: float = 100.0,
        alpha: float = 0.25,
        own_lag_means: np.ndarray | list[float] | None = None,
        own_lag_mean: float = 0.0,
        min_sigma2: float = 1e-12,
    ) -> PriorSpec:
        """Construct an experimental tempered bridge between legacy and canonical Minnesota.

        This keeps the equation-wise sampling path used by
        :meth:`PriorSpec.niw_minnesota_canonical`, but geometrically blends the
        legacy and canonical coefficient variances:

        - ``alpha=0.0`` reproduces the legacy variance map
        - ``alpha=1.0`` reproduces the canonical variance map

        Notes
        -----
        This is an explicit experimental mitigation path. It does not redefine the
        semantics of :meth:`PriorSpec.niw_minnesota_canonical`.
        """
        alpha_f = float(alpha)
        if not np.isfinite(alpha_f) or not (0.0 <= alpha_f <= 1.0):
            raise ValueError("alpha must be finite and in [0, 1]")

        legacy = PriorSpec.niw_minnesota_legacy(
            p=p,
            y=y,
            n=n,
            include_intercept=include_intercept,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
            min_sigma2=min_sigma2,
        )
        canonical = PriorSpec.niw_minnesota_canonical(
            p=p,
            y=y,
            n=n,
            include_intercept=include_intercept,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            lambda4=lambda4,
            own_lag_means=own_lag_means,
            own_lag_mean=own_lag_mean,
            min_sigma2=min_sigma2,
        )
        canonical_meta = canonical.minnesota_canonical
        if canonical_meta is None:
            raise RuntimeError("canonical Minnesota metadata missing")

        n_eq = canonical.niw.m0.shape[1]
        legacy_diag = np.diag(np.asarray(legacy.niw.v0, dtype=float))
        legacy_var = np.repeat(legacy_diag.reshape(-1, 1), repeats=n_eq, axis=1)
        canonical_var = 1.0 / np.asarray(canonical_meta.inv_v0_vec, dtype=float).reshape(
            legacy_var.shape,
            order="F",
        )
        tempered_var = _tempered_minnesota_variances(
            legacy_var=legacy_var,
            canonical_var=canonical_var,
            alpha=alpha_f,
        )
        v0_summary = np.diag(np.mean(tempered_var, axis=1))
        niw = NIWPrior(
            m0=np.asarray(canonical.niw.m0, dtype=float).copy(),
            v0=v0_summary,
            s0=np.asarray(canonical.niw.s0, dtype=float).copy(),
            nu0=float(canonical.niw.nu0),
        )
        tempered = MinnesotaCanonicalSpec(
            sigma2=np.asarray(canonical_meta.sigma2, dtype=float).copy(),
            inv_v0_vec=1.0 / tempered_var.reshape(-1, order="F"),
            mode="tempered",
            tempered_alpha=alpha_f,
        )
        return PriorSpec(
            family="niw",
            niw=niw,
            minnesota_canonical=tempered,
        )

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
    ) -> PriorSpec:
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
            Optional slab variance override for the intercept term.
        fix_intercept:
            Whether to force inclusion of the intercept row.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if n < 1:
            raise ValueError("n must be >= 1")

        m0a: np.ndarray
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

    @staticmethod
    def from_blasso(
        *,
        k: int,
        n: int,
        mode: Literal["global", "adaptive"] = "global",
        include_intercept: bool = True,
        m0: np.ndarray | None = None,
        s0: np.ndarray | None = None,
        nu0: float | None = None,
        a0_global: float = 1.0,
        b0_global: float = 1.0,
        a0_c: float = 1.0,
        b0_c: float = 1.0,
        a0_L: float = 1.0,
        b0_L: float = 1.0,
        tau_init: float = 1e4,
        lambda_init: float = 2.0,
    ) -> PriorSpec:
        if k < 1:
            raise ValueError("k must be >= 1")
        if n < 1:
            raise ValueError("n must be >= 1")

        m0a: np.ndarray
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
        spec = BLassoSpec(
            mode=mode,
            a0_global=float(a0_global),
            b0_global=float(b0_global),
            a0_c=float(a0_c),
            b0_c=float(b0_c),
            a0_L=float(a0_L),
            b0_L=float(b0_L),
            tau_init=float(tau_init),
            lambda_init=float(lambda_init),
        )
        _ = bool(include_intercept)
        return PriorSpec(family="blasso", niw=niw, blasso=spec)

    @staticmethod
    def from_dl(
        *,
        k: int,
        n: int,
        include_intercept: bool = True,
        m0: np.ndarray | None = None,
        s0: np.ndarray | None = None,
        nu0: float | None = None,
        abeta: float = 0.5,
        dl_scaler: float = 0.1,
    ) -> PriorSpec:
        if k < 1:
            raise ValueError("k must be >= 1")
        if n < 1:
            raise ValueError("n must be >= 1")

        m0a: np.ndarray
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
        spec = DLSpec(abeta=float(abeta), dl_scaler=float(dl_scaler))
        _ = bool(include_intercept)
        return PriorSpec(family="dl", niw=niw, dl=spec)


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
