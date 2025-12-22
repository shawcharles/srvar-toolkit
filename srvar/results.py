from __future__ import annotations

from dataclasses import dataclass

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
    h_draws: np.ndarray | None = None  # (D, T, N)
    h0_draws: np.ndarray | None = None  # (D, N)
    sigma_eta2_draws: np.ndarray | None = None  # (D, N)
    gamma_draws: np.ndarray | None = None  # (D, K)


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Output of :func:`srvar.api.forecast`.
    """
    variables: list[str]
    horizons: list[int]
    draws: np.ndarray  # (D, H, N)
    mean: np.ndarray  # (H, N)
    quantiles: dict[float, np.ndarray]  # q -> (H, N)
    latent_draws: np.ndarray | None = None  # (D, H, N)
