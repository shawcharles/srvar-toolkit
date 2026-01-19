from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.dataset import Dataset
from .results import FitResult, ForecastResult, PosteriorNIW


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
        "latent_values": (
            fit_res.latent_dataset.values if fit_res.latent_dataset is not None else None
        ),
        "posterior_mn": fit_res.posterior.mn if fit_res.posterior is not None else None,
        "posterior_vn": fit_res.posterior.vn if fit_res.posterior is not None else None,
        "posterior_sn": fit_res.posterior.sn if fit_res.posterior is not None else None,
        "posterior_nun": fit_res.posterior.nun if fit_res.posterior is not None else None,
        "h_draws": fit_res.h_draws,
        "h0_draws": fit_res.h0_draws,
        "sigma_eta2_draws": fit_res.sigma_eta2_draws,
        "sv_gamma0_draws": fit_res.sv_gamma0_draws,
        "sv_phi_draws": fit_res.sv_phi_draws,
        "lambda_draws": fit_res.lambda_draws,
        "factor_draws": fit_res.factor_draws,
        "h_factor_draws": fit_res.h_factor_draws,
        "h0_factor_draws": fit_res.h0_factor_draws,
        "sigma_eta2_factor_draws": fit_res.sigma_eta2_factor_draws,
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
    posterior: PosteriorNIW | None
    beta_draws: np.ndarray | None
    sigma_draws: np.ndarray | None
    q_draws: np.ndarray | None
    latent_dataset: Dataset | None
    latent_draws: np.ndarray | None
    h_draws: np.ndarray | None
    h0_draws: np.ndarray | None
    sigma_eta2_draws: np.ndarray | None
    sv_gamma0_draws: np.ndarray | None
    sv_phi_draws: np.ndarray | None
    lambda_draws: np.ndarray | None
    factor_draws: np.ndarray | None
    h_factor_draws: np.ndarray | None
    h0_factor_draws: np.ndarray | None
    sigma_eta2_factor_draws: np.ndarray | None
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

        latent_values = _optional_npz_array(npz, "latent_values")
        latent_dataset = None
        if latent_values is not None:
            latent_dataset = Dataset.from_arrays(
                values=np.asarray(latent_values, dtype=float),
                variables=variables,
                time_index=time_index,
            )

        mn = _optional_npz_array(npz, "posterior_mn")
        vn = _optional_npz_array(npz, "posterior_vn")
        sn = _optional_npz_array(npz, "posterior_sn")
        nun = _optional_npz_array(npz, "posterior_nun")
        posterior_parts = (mn, vn, sn, nun)
        if any(p is not None for p in posterior_parts) and not all(p is not None for p in posterior_parts):
            raise ValueError(
                "fit_result.npz contains a partial posterior_* block; expected all of "
                "posterior_mn/posterior_vn/posterior_sn/posterior_nun or none"
            )
        posterior = None
        if all(p is not None for p in posterior_parts):
            assert mn is not None and vn is not None and sn is not None and nun is not None
            posterior = PosteriorNIW(
                mn=np.asarray(mn, dtype=float),
                vn=np.asarray(vn, dtype=float),
                sn=np.asarray(sn, dtype=float),
                nun=float(np.asarray(nun, dtype=float).reshape(())),
            )

        return FitNPZ(
            dataset=ds,
            posterior=posterior,
            beta_draws=_optional_npz_array(npz, "beta_draws"),
            sigma_draws=_optional_npz_array(npz, "sigma_draws"),
            q_draws=_optional_npz_array(npz, "q_draws"),
            latent_dataset=latent_dataset,
            latent_draws=_optional_npz_array(npz, "latent_draws"),
            h_draws=_optional_npz_array(npz, "h_draws"),
            h0_draws=_optional_npz_array(npz, "h0_draws"),
            sigma_eta2_draws=_optional_npz_array(npz, "sigma_eta2_draws"),
            sv_gamma0_draws=_optional_npz_array(npz, "sv_gamma0_draws"),
            sv_phi_draws=_optional_npz_array(npz, "sv_phi_draws"),
            lambda_draws=_optional_npz_array(npz, "lambda_draws"),
            factor_draws=_optional_npz_array(npz, "factor_draws"),
            h_factor_draws=_optional_npz_array(npz, "h_factor_draws"),
            h0_factor_draws=_optional_npz_array(npz, "h0_factor_draws"),
            sigma_eta2_factor_draws=_optional_npz_array(npz, "sigma_eta2_factor_draws"),
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


def load_run_dir(
    out_dir: str | Path,
    *,
    config_filename: str = "config.yml",
    fit_filename: str = "fit_result.npz",
) -> FitResult:
    """Load a :class:`~srvar.results.FitResult` from a `srvar run` output directory.

    This function:

    1) Loads the stored draws/state from ``fit_result.npz``.
    2) Reconstructs ``ModelSpec``, ``PriorSpec``, and ``SamplerConfig`` from the saved
       ``config.yml`` (without re-loading the original CSV).

    Notes
    -----
    - The returned object is suitable for downstream analysis (IRFs/FEVD/HD) and forecasting.
    - If the saved config and saved dataset are inconsistent (e.g., variable list changed),
      config parsing may fail.
    """
    out = Path(out_dir)
    cfg_path = out / str(config_filename)
    fit_path = out / str(fit_filename)

    if not cfg_path.exists():
        raise FileNotFoundError(f"run directory is missing config file: {cfg_path}")
    if not fit_path.exists():
        raise FileNotFoundError(f"run directory is missing fit artifact: {fit_path}")

    from .config import build_model, build_prior, build_sampler, load_config

    cfg = load_config(cfg_path)
    fit_npz = load_fit_npz(fit_path)

    ds = fit_npz.dataset
    model = build_model(cfg, dataset=ds)
    prior = build_prior(cfg, dataset=ds, model=model)
    sampler, _rng = build_sampler(cfg)

    return FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=fit_npz.posterior,
        latent_dataset=fit_npz.latent_dataset,
        latent_draws=fit_npz.latent_draws,
        beta_draws=fit_npz.beta_draws,
        sigma_draws=fit_npz.sigma_draws,
        q_draws=fit_npz.q_draws,
        h_draws=fit_npz.h_draws,
        h0_draws=fit_npz.h0_draws,
        sigma_eta2_draws=fit_npz.sigma_eta2_draws,
        sv_gamma0_draws=fit_npz.sv_gamma0_draws,
        sv_phi_draws=fit_npz.sv_phi_draws,
        lambda_draws=fit_npz.lambda_draws,
        factor_draws=fit_npz.factor_draws,
        h_factor_draws=fit_npz.h_factor_draws,
        h0_factor_draws=fit_npz.h0_factor_draws,
        sigma_eta2_factor_draws=fit_npz.sigma_eta2_factor_draws,
        gamma_draws=fit_npz.gamma_draws,
        mu_draws=fit_npz.mu_draws,
        mu_gamma_draws=fit_npz.mu_gamma_draws,
    )
