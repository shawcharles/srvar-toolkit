from __future__ import annotations

from typing import Any

import numpy as np

from .results import (
    FEVDResult,
    FitResult,
    ForecastResult,
    HistoricalDecompositionResult,
    IRFResult,
)


def _require_xarray() -> Any:
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "xarray is required for labeled outputs. Install with `pip install srvar-toolkit[xarray]`."
        ) from exc
    return xr


def _regressor_names(*, variables: list[str], p: int, include_intercept: bool) -> list[str]:
    names: list[str] = []
    if include_intercept:
        names.append("const")
    for lag in range(1, int(p) + 1):
        for v in variables:
            names.append(f"{v}_lag{lag}")
    return names


def fit_to_xarray(fit: FitResult) -> Any:
    """Convert a :class:`~srvar.results.FitResult` to an `xarray.Dataset`.

    Returned variables depend on what the fit contains (e.g., ELB/SV models include
    additional latent/state draws).

    Notes
    -----
    Some time-varying draws (e.g., stochastic volatility states, factor SV states)
    are defined on the effective sample ``T - p``. These are aligned to the full
    dataset time index; the first ``p`` observations are filled with missing values.
    """
    xr = _require_xarray()

    variables = list(fit.dataset.variables)
    time = fit.dataset.time_index
    p = int(fit.model.p)
    time_eff = time[p:] if p <= len(time) else time[0:0]

    ds = xr.Dataset(
        data_vars={
            "y": xr.DataArray(
                np.asarray(fit.dataset.values, dtype=float),
                dims=("time", "variable"),
                coords={"time": time, "variable": variables},
            )
        },
        coords={"time": time, "variable": variables},
        attrs={
            "p": p,
            "include_intercept": bool(fit.model.include_intercept),
        },
    )

    if fit.latent_dataset is not None:
        ds["y_latent"] = xr.DataArray(
            np.asarray(fit.latent_dataset.values, dtype=float),
            dims=("time", "variable"),
            coords={"time": time, "variable": variables},
        )

    regressor = _regressor_names(
        variables=variables, p=int(fit.model.p), include_intercept=bool(fit.model.include_intercept)
    )

    if fit.beta_draws is not None:
        draws = np.asarray(fit.beta_draws, dtype=float)
        ds["beta"] = xr.DataArray(
            draws,
            dims=("draw", "regressor", "variable"),
            coords={
                "draw": np.arange(draws.shape[0], dtype=int),
                "regressor": regressor,
                "variable": variables,
            },
        )

    if fit.sigma_draws is not None:
        sig = np.asarray(fit.sigma_draws, dtype=float)
        ds["sigma"] = xr.DataArray(
            sig,
            dims=("draw", "variable", "variable2"),
            coords={
                "draw": np.arange(sig.shape[0], dtype=int),
                "variable": variables,
                "variable2": variables,
            },
        )

    if fit.q_draws is not None:
        q = np.asarray(fit.q_draws, dtype=float)
        ds["q"] = xr.DataArray(
            q,
            dims=("draw", "variable", "variable2"),
            coords={
                "draw": np.arange(q.shape[0], dtype=int),
                "variable": variables,
                "variable2": variables,
            },
        )

    if fit.latent_draws is not None:
        lat = np.asarray(fit.latent_draws, dtype=float)
        ds["latent_draws"] = xr.DataArray(
            lat,
            dims=("draw", "time", "variable"),
            coords={
                "draw": np.arange(lat.shape[0], dtype=int),
                "time": time,
                "variable": variables,
            },
        )

    if fit.h_draws is not None:
        h = np.asarray(fit.h_draws, dtype=float)
        ds["h"] = xr.DataArray(
            h,
            dims=("draw", "time", "variable"),
            coords={
                "draw": np.arange(h.shape[0], dtype=int),
                "time": time_eff,
                "variable": variables,
            },
        )

    if fit.h0_draws is not None:
        h0 = np.asarray(fit.h0_draws, dtype=float)
        ds["h0"] = xr.DataArray(
            h0,
            dims=("draw", "variable"),
            coords={"draw": np.arange(h0.shape[0], dtype=int), "variable": variables},
        )

    if fit.sigma_eta2_draws is not None:
        se = np.asarray(fit.sigma_eta2_draws, dtype=float)
        ds["sigma_eta2"] = xr.DataArray(
            se,
            dims=("draw", "variable"),
            coords={"draw": np.arange(se.shape[0], dtype=int), "variable": variables},
        )

    if fit.sv_gamma0_draws is not None:
        g0 = np.asarray(fit.sv_gamma0_draws, dtype=float)
        ds["sv_gamma0"] = xr.DataArray(
            g0,
            dims=("draw", "variable"),
            coords={"draw": np.arange(g0.shape[0], dtype=int), "variable": variables},
        )

    if fit.sv_phi_draws is not None:
        phi = np.asarray(fit.sv_phi_draws, dtype=float)
        ds["sv_phi"] = xr.DataArray(
            phi,
            dims=("draw", "variable"),
            coords={"draw": np.arange(phi.shape[0], dtype=int), "variable": variables},
        )

    if fit.gamma_draws is not None:
        g = np.asarray(fit.gamma_draws, dtype=bool)
        ds["gamma"] = xr.DataArray(
            g,
            dims=("draw", "regressor"),
            coords={"draw": np.arange(g.shape[0], dtype=int), "regressor": regressor},
        )

    if fit.mu_draws is not None:
        mu = np.asarray(fit.mu_draws, dtype=float)
        ds["mu"] = xr.DataArray(
            mu,
            dims=("draw", "variable"),
            coords={"draw": np.arange(mu.shape[0], dtype=int), "variable": variables},
        )

    if fit.mu_gamma_draws is not None:
        mg = np.asarray(fit.mu_gamma_draws, dtype=bool)
        ds["mu_gamma"] = xr.DataArray(
            mg,
            dims=("draw", "variable"),
            coords={"draw": np.arange(mg.shape[0], dtype=int), "variable": variables},
        )

    if fit.lambda_draws is not None:
        lam = np.asarray(fit.lambda_draws, dtype=float)
        k = int(lam.shape[2])
        ds["lambda"] = xr.DataArray(
            lam,
            dims=("draw", "variable", "factor"),
            coords={
                "draw": np.arange(lam.shape[0], dtype=int),
                "variable": variables,
                "factor": np.arange(k, dtype=int),
            },
        )

    if fit.factor_draws is not None:
        f = np.asarray(fit.factor_draws, dtype=float)
        ds["factors"] = xr.DataArray(
            f,
            dims=("draw", "time", "factor"),
            coords={
                "draw": np.arange(f.shape[0], dtype=int),
                "time": time_eff,
                "factor": np.arange(f.shape[2], dtype=int),
            },
        )

    if fit.h_factor_draws is not None:
        hf = np.asarray(fit.h_factor_draws, dtype=float)
        ds["h_factor"] = xr.DataArray(
            hf,
            dims=("draw", "time", "factor"),
            coords={
                "draw": np.arange(hf.shape[0], dtype=int),
                "time": time_eff,
                "factor": np.arange(hf.shape[2], dtype=int),
            },
        )

    if fit.h0_factor_draws is not None:
        h0f = np.asarray(fit.h0_factor_draws, dtype=float)
        ds["h0_factor"] = xr.DataArray(
            h0f,
            dims=("draw", "factor"),
            coords={
                "draw": np.arange(h0f.shape[0], dtype=int),
                "factor": np.arange(h0f.shape[1], dtype=int),
            },
        )

    if fit.sigma_eta2_factor_draws is not None:
        se_f = np.asarray(fit.sigma_eta2_factor_draws, dtype=float)
        ds["sigma_eta2_factor"] = xr.DataArray(
            se_f,
            dims=("draw", "factor"),
            coords={
                "draw": np.arange(se_f.shape[0], dtype=int),
                "factor": np.arange(se_f.shape[1], dtype=int),
            },
        )

    return ds


def forecast_to_xarray(fc: ForecastResult) -> Any:
    """Convert a :class:`~srvar.results.ForecastResult` to an `xarray.Dataset`."""
    xr = _require_xarray()

    variables = list(fc.variables)
    draws = np.asarray(fc.draws, dtype=float)
    h = int(draws.shape[1])
    horizons_full = np.arange(1, h + 1, dtype=int)

    ds = xr.Dataset(
        data_vars={
            "draws": xr.DataArray(
                draws,
                dims=("draw", "horizon", "variable"),
                coords={
                    "draw": np.arange(draws.shape[0], dtype=int),
                    "horizon": horizons_full,
                    "variable": variables,
                },
            ),
            "mean": xr.DataArray(
                np.asarray(fc.mean, dtype=float),
                dims=("horizon", "variable"),
                coords={"horizon": horizons_full, "variable": variables},
            ),
        },
        coords={"horizon": horizons_full, "variable": variables},
        attrs={"horizons_requested": list(fc.horizons)},
    )

    if fc.quantiles:
        qs = sorted(float(q) for q in fc.quantiles.keys())
        q_arr = np.stack([np.asarray(fc.quantiles[q], dtype=float) for q in qs], axis=0)
        ds["quantiles"] = xr.DataArray(
            q_arr,
            dims=("quantile", "horizon", "variable"),
            coords={"quantile": qs, "horizon": horizons_full, "variable": variables},
        )

    if fc.latent_draws is not None:
        lat = np.asarray(fc.latent_draws, dtype=float)
        ds["latent_draws"] = xr.DataArray(
            lat,
            dims=("draw", "horizon", "variable"),
            coords={
                "draw": np.arange(lat.shape[0], dtype=int),
                "horizon": horizons_full,
                "variable": variables,
            },
        )

    return ds


def irf_to_xarray(irf: IRFResult) -> Any:
    """Convert an :class:`~srvar.results.IRFResult` to an `xarray.Dataset`."""
    xr = _require_xarray()

    variables = list(irf.variables)
    shocks = list(irf.shocks)
    horizons = np.asarray(irf.horizons, dtype=int)
    draws = np.asarray(irf.draws, dtype=float)

    ds = xr.Dataset(
        data_vars={
            "draws": xr.DataArray(
                draws,
                dims=("draw", "horizon", "variable", "shock"),
                coords={
                    "draw": np.arange(draws.shape[0], dtype=int),
                    "horizon": horizons,
                    "variable": variables,
                    "shock": shocks,
                },
            ),
            "mean": xr.DataArray(
                np.asarray(irf.mean, dtype=float),
                dims=("horizon", "variable", "shock"),
                coords={"horizon": horizons, "variable": variables, "shock": shocks},
            ),
        },
        coords={"horizon": horizons, "variable": variables, "shock": shocks},
        attrs={"identification": str(irf.identification)},
    )

    if irf.quantiles:
        qs = sorted(float(q) for q in irf.quantiles.keys())
        q_arr = np.stack([np.asarray(irf.quantiles[q], dtype=float) for q in qs], axis=0)
        ds["quantiles"] = xr.DataArray(
            q_arr,
            dims=("quantile", "horizon", "variable", "shock"),
            coords={"quantile": qs, "horizon": horizons, "variable": variables, "shock": shocks},
        )

    return ds


def fevd_to_xarray(fevd: FEVDResult) -> Any:
    """Convert a :class:`~srvar.results.FEVDResult` to an `xarray.Dataset`."""
    xr = _require_xarray()

    variables = list(fevd.variables)
    shocks = list(fevd.shocks)
    horizons = np.asarray(fevd.horizons, dtype=int)
    draws = np.asarray(fevd.draws, dtype=float)

    ds = xr.Dataset(
        data_vars={
            "draws": xr.DataArray(
                draws,
                dims=("draw", "horizon", "variable", "shock"),
                coords={
                    "draw": np.arange(draws.shape[0], dtype=int),
                    "horizon": horizons,
                    "variable": variables,
                    "shock": shocks,
                },
            ),
            "mean": xr.DataArray(
                np.asarray(fevd.mean, dtype=float),
                dims=("horizon", "variable", "shock"),
                coords={"horizon": horizons, "variable": variables, "shock": shocks},
            ),
        },
        coords={"horizon": horizons, "variable": variables, "shock": shocks},
        attrs={"identification": str(fevd.identification)},
    )

    if fevd.quantiles:
        qs = sorted(float(q) for q in fevd.quantiles.keys())
        q_arr = np.stack([np.asarray(fevd.quantiles[q], dtype=float) for q in qs], axis=0)
        ds["quantiles"] = xr.DataArray(
            q_arr,
            dims=("quantile", "horizon", "variable", "shock"),
            coords={"quantile": qs, "horizon": horizons, "variable": variables, "shock": shocks},
        )

    return ds


def historical_decomposition_to_xarray(hd: HistoricalDecompositionResult) -> Any:
    """Convert a :class:`~srvar.results.HistoricalDecompositionResult` to an `xarray.Dataset`."""
    xr = _require_xarray()

    variables = list(hd.variables)
    shocks = list(hd.shocks)
    time = hd.time_index

    baseline = np.asarray(hd.baseline_draws, dtype=float)
    contrib = np.asarray(hd.contributions_draws, dtype=float)
    shock_draws = np.asarray(hd.shock_draws, dtype=float)

    ds = xr.Dataset(
        data_vars={
            "baseline_draws": xr.DataArray(
                baseline,
                dims=("draw", "time", "variable"),
                coords={
                    "draw": np.arange(baseline.shape[0], dtype=int),
                    "time": time,
                    "variable": variables,
                },
            ),
            "shock_draws": xr.DataArray(
                shock_draws,
                dims=("draw", "time", "shock"),
                coords={
                    "draw": np.arange(shock_draws.shape[0], dtype=int),
                    "time": time,
                    "shock": shocks,
                },
            ),
            "contributions_draws": xr.DataArray(
                contrib,
                dims=("draw", "time", "variable", "shock"),
                coords={
                    "draw": np.arange(contrib.shape[0], dtype=int),
                    "time": time,
                    "variable": variables,
                    "shock": shocks,
                },
            ),
        },
        coords={"time": time, "variable": variables, "shock": shocks},
        attrs={"identification": str(hd.identification)},
    )

    if hd.quantiles:
        qs = sorted(float(q) for q in hd.quantiles.keys())
        q_arr = np.stack([np.asarray(hd.quantiles[q], dtype=float) for q in qs], axis=0)
        ds["quantiles"] = xr.DataArray(
            q_arr,
            dims=("quantile", "time", "variable", "shock"),
            coords={"quantile": qs, "time": time, "variable": variables, "shock": shocks},
        )

    return ds
