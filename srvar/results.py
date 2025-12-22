from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.dataset import Dataset
from .spec import ModelSpec, PriorSpec, SamplerConfig


@dataclass(frozen=True, slots=True)
class PosteriorNIW:
    mn: np.ndarray  # (K, N)
    vn: np.ndarray  # (K, K)
    sn: np.ndarray  # (N, N)
    nun: float


@dataclass(frozen=True, slots=True)
class FitResult:
    dataset: Dataset
    model: ModelSpec
    prior: PriorSpec
    sampler: SamplerConfig
    posterior: PosteriorNIW | None
    latent_dataset: Dataset | None = None
    beta_draws: np.ndarray | None = None  # (D, K, N)
    sigma_draws: np.ndarray | None = None  # (D, N, N)
    h_draws: np.ndarray | None = None  # (D, T, N)
    h0_draws: np.ndarray | None = None  # (D, N)
    sigma_eta2_draws: np.ndarray | None = None  # (D, N)


@dataclass(frozen=True, slots=True)
class ForecastResult:
    variables: list[str]
    horizons: list[int]
    draws: np.ndarray  # (D, H, N)
    mean: np.ndarray  # (H, N)
    quantiles: dict[float, np.ndarray]  # q -> (H, N)
