from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data.dataset import Dataset
from .spec import ModelSpec, PriorSpec, SamplerConfig


@dataclass(frozen=True, slots=True)
class PosteriorNIW:
    """NIW posterior parameter block.

    This is returned in :class:`~srvar.results.FitResult` when the model is
    conjugate and closed-form NIW posterior parameters are available.
    """

    mn: np.ndarray  # (K, N)
    vn: np.ndarray  # (K, K)
    sn: np.ndarray  # (N, N)
    nun: float


@dataclass(frozen=True, slots=True)
class FitResult:
    """Output of :func:`srvar.api.fit`.

    Depending on the model configuration, this object may contain:

    - Closed-form NIW posterior parameters (``posterior``)
    - Stored posterior draws of coefficients/covariances (``beta_draws``, ``sigma_draws``)
    - Latent shadow-rate series/draws when ELB is enabled
    - Stochastic volatility state draws when volatility is enabled
    - SSVS inclusion indicator draws when ``prior.family='ssvs'``

    """

    dataset: Dataset
    model: ModelSpec
    prior: PriorSpec
    sampler: SamplerConfig
    posterior: PosteriorNIW | None
    latent_dataset: Dataset | None = None
    latent_draws: np.ndarray | None = None  # (D, T, N)
    beta_draws: np.ndarray | None = None  # (D, K, N)
    sigma_draws: np.ndarray | None = None  # (D, N, N)
    q_draws: np.ndarray | None = None  # (D, N, N)
    h_draws: np.ndarray | None = None  # (D, T, N)
    h0_draws: np.ndarray | None = None  # (D, N)
    sigma_eta2_draws: np.ndarray | None = None  # (D, N)
    sv_gamma0_draws: np.ndarray | None = None  # (D, N)
    sv_phi_draws: np.ndarray | None = None  # (D, N)
    # Factor stochastic volatility (FSV) extras (when model.volatility.covariance == "factor")
    lambda_draws: np.ndarray | None = None  # (D, N, k)
    factor_draws: np.ndarray | None = None  # (D, T, k)
    h_factor_draws: np.ndarray | None = None  # (D, T, k)
    h0_factor_draws: np.ndarray | None = None  # (D, k)
    sigma_eta2_factor_draws: np.ndarray | None = None  # (D, k)
    gamma_draws: np.ndarray | None = None  # (D, K)
    mu_draws: np.ndarray | None = None  # (D, N)
    mu_gamma_draws: np.ndarray | None = None  # (D, N)


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Output of :func:`srvar.api.forecast`."""

    variables: list[str]
    horizons: list[int]
    draws: np.ndarray  # (D, H, N)
    mean: np.ndarray  # (H, N)
    quantiles: dict[float, np.ndarray]  # q -> (H, N)
    latent_draws: np.ndarray | None = None  # (D, H, N)


@dataclass(frozen=True, slots=True)
class IRFResult:
    """Impulse response function (IRF) draws and summaries."""

    variables: list[str]
    shocks: list[str]
    horizons: list[int]
    draws: np.ndarray  # (D, H, N, N): response variable × shock
    mean: np.ndarray  # (H, N, N)
    quantiles: dict[float, np.ndarray]  # q -> (H, N, N)
    identification: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FEVDResult:
    """Forecast error variance decomposition (FEVD) draws and summaries."""

    variables: list[str]
    shocks: list[str]
    horizons: list[int]
    draws: np.ndarray  # (D, H, N, N): response variable × shock share
    mean: np.ndarray  # (H, N, N)
    quantiles: dict[float, np.ndarray]  # q -> (H, N, N)
    identification: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HistoricalDecompositionResult:
    """Historical decomposition (HD) draws and summaries.

    This decomposes observed (or latent) series into a baseline path plus shock contributions
    implied by a structural identification scheme.
    """

    variables: list[str]
    shocks: list[str]
    time_index: Any
    baseline_draws: np.ndarray  # (D, T, N)
    shock_draws: np.ndarray  # (D, T, N): structural shocks
    contributions_draws: np.ndarray  # (D, T, N, N): variable × shock contributions
    mean: np.ndarray  # (T, N, N)
    quantiles: dict[float, np.ndarray]  # q -> (T, N, N)
    identification: str
    metadata: dict[str, Any] = field(default_factory=dict)
