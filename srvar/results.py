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

    Attributes
    ----------
    mn:
        Posterior mean of VAR coefficients with shape ``(K, N)``.
    vn:
        Posterior coefficient covariance with shape ``(K, K)``.
    sn:
        Posterior scale matrix for the inverse-Wishart with shape ``(N, N)``.
    nun:
        Posterior degrees of freedom.
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

    Attributes
    ----------
    dataset:
        Original observed dataset.
    model, prior, sampler:
        Specifications used for estimation.
    posterior:
        NIW posterior parameters when available.
    latent_dataset:
        Final latent dataset state (useful for ELB models).
    latent_draws:
        Latent shadow-rate draws with shape ``(D, T, N)``.
    beta_draws, sigma_draws:
        Posterior draws of VAR parameters.
    h_draws, h0_draws, sigma_eta2_draws:
        Stochastic volatility draws.
    gamma_draws:
        SSVS inclusion indicator draws.
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

    Attributes
    ----------
    variables:
        Variable names.
    horizons:
        Horizons requested by the caller.
    draws:
        Predictive simulation draws with shape ``(D, H, N)`` where
        ``H = max(horizons)``.
    mean:
        Mean forecast path with shape ``(H, N)``.
    quantiles:
        Mapping from quantile level to an array with shape ``(H, N)``.
    latent_draws:
        If the fitted model uses ELB shadow-rate augmentation, this contains the
        unconstrained latent predictive draws.
    """
    variables: list[str]
    horizons: list[int]
    draws: np.ndarray  # (D, H, N)
    mean: np.ndarray  # (H, N)
    quantiles: dict[float, np.ndarray]  # q -> (H, N)
    latent_draws: np.ndarray | None = None  # (D, H, N)
