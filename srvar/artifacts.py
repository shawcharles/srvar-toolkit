from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.dataset import Dataset
from .results import FitResult, ForecastResult


def save_fit_npz(path: str | Path, fit_res: FitResult) -> None:
    p = Path(path)
    payload: dict[str, Any] = {
        "variables": np.asarray(fit_res.dataset.variables, dtype=object),
        "time_index": np.asarray(fit_res.dataset.time_index.to_numpy(), dtype="datetime64[ns]"),
        "values": fit_res.dataset.values,
        "beta_draws": fit_res.beta_draws,
        "sigma_draws": fit_res.sigma_draws,
        "q_draws": fit_res.q_draws,
        "latent_draws": fit_res.latent_draws,
        "h_draws": fit_res.h_draws,
        "h0_draws": fit_res.h0_draws,
        "sigma_eta2_draws": fit_res.sigma_eta2_draws,
        "sv_gamma0_draws": fit_res.sv_gamma0_draws,
        "sv_phi_draws": fit_res.sv_phi_draws,
        "gamma_draws": fit_res.gamma_draws,
        "mu_draws": fit_res.mu_draws,
        "mu_gamma_draws": fit_res.mu_gamma_draws,
    }
    np.savez_compressed(p, allow_pickle=True, **payload)


def save_forecast_npz(path: str | Path, fc: ForecastResult) -> None:
    p = Path(path)
    payload: dict[str, Any] = {
        "variables": np.asarray(fc.variables, dtype=object),
        "horizons": np.asarray(fc.horizons, dtype=int),
        "draws": fc.draws,
        "mean": fc.mean,
        "latent_draws": fc.latent_draws,
    }
    payload.update({f"q_{q}": arr for q, arr in fc.quantiles.items()})
    np.savez_compressed(p, allow_pickle=True, **payload)


def _optional_npz_array(npz: Any, key: str) -> np.ndarray | None:
    if key not in npz:
        return None
    arr = npz[key]
    if (
        isinstance(arr, np.ndarray)
        and arr.shape == ()
        and arr.dtype == object
        and arr.item() is None
    ):
        return None
    return arr


@dataclass(frozen=True, slots=True)
class FitNPZ:
    dataset: Dataset
    beta_draws: np.ndarray | None
    sigma_draws: np.ndarray | None
    q_draws: np.ndarray | None
    latent_draws: np.ndarray | None
    h_draws: np.ndarray | None
    h0_draws: np.ndarray | None
    sigma_eta2_draws: np.ndarray | None
    sv_gamma0_draws: np.ndarray | None
    sv_phi_draws: np.ndarray | None
    gamma_draws: np.ndarray | None
    mu_draws: np.ndarray | None
    mu_gamma_draws: np.ndarray | None


def load_fit_npz(path: str | Path) -> FitNPZ:
    p = Path(path)
    with np.load(p, allow_pickle=True) as npz:
        variables = [str(v) for v in np.asarray(npz["variables"], dtype=object).tolist()]
        time_index = pd.DatetimeIndex(pd.to_datetime(npz["time_index"]))
        values = np.asarray(npz["values"], dtype=float)
        ds = Dataset.from_arrays(values=values, variables=variables, time_index=time_index)

        return FitNPZ(
            dataset=ds,
            beta_draws=_optional_npz_array(npz, "beta_draws"),
            sigma_draws=_optional_npz_array(npz, "sigma_draws"),
            q_draws=_optional_npz_array(npz, "q_draws"),
            latent_draws=_optional_npz_array(npz, "latent_draws"),
            h_draws=_optional_npz_array(npz, "h_draws"),
            h0_draws=_optional_npz_array(npz, "h0_draws"),
            sigma_eta2_draws=_optional_npz_array(npz, "sigma_eta2_draws"),
            sv_gamma0_draws=_optional_npz_array(npz, "sv_gamma0_draws"),
            sv_phi_draws=_optional_npz_array(npz, "sv_phi_draws"),
            gamma_draws=_optional_npz_array(npz, "gamma_draws"),
            mu_draws=_optional_npz_array(npz, "mu_draws"),
            mu_gamma_draws=_optional_npz_array(npz, "mu_gamma_draws"),
        )


def load_forecast_npz(path: str | Path) -> ForecastResult:
    p = Path(path)
    with np.load(p, allow_pickle=True) as npz:
        variables = [str(v) for v in np.asarray(npz["variables"], dtype=object).tolist()]
        horizons = [int(h) for h in np.asarray(npz["horizons"], dtype=int).tolist()]
        draws = np.asarray(npz["draws"], dtype=float)
        mean = np.asarray(npz["mean"], dtype=float)
        latent_draws = _optional_npz_array(npz, "latent_draws")
        latent = None if latent_draws is None else np.asarray(latent_draws, dtype=float)

        quantiles: dict[float, np.ndarray] = {}
        for key in npz.files:
            if not key.startswith("q_"):
                continue
            try:
                q = float(key[2:])
            except ValueError:
                continue
            quantiles[q] = np.asarray(npz[key], dtype=float)

        return ForecastResult(
            variables=variables,
            horizons=horizons,
            draws=draws,
            mean=mean,
            quantiles=quantiles,
            latent_draws=latent,
        )
