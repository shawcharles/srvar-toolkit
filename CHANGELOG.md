# Changelog

All notable changes to the Python SRVAR toolkit will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- `srvar fetch-fred` command to fetch FRED series to a cached CSV (config-driven).
- `--dry-run` flag for `fetch-fred` (prints planned fetch/output without network calls).
- `--validate-series` flag for `fetch-fred` (preflight series existence check via FRED metadata).
- Transformation support in the fetch pipeline, including `processing.transform_order`.
- Runtime warnings for non-positive values when applying log-based tcodes (4/5/6).
- Unit tests covering `tcode_1d` and the `fetch_fred` helpers (mocked, no network).
- Steady-state VAR parameterization (SSP) with Gibbs sampling of the steady-state mean `mu`.
- Optional spike-and-slab selection on `mu` (mu-SSVS).
- YAML-only configuration support for SSP via `model.steady_state`.
- SSP example script (`examples/ssp_fit_forecast.py`).
- SSP test coverage (`tests/test_ssp.py`).
- Robust shock models for homoskedastic VARs via `model.shocks` (Student‑t and outlier-mixture innovations).
- Full-covariance stochastic volatility via factor SV (`model.volatility.covariance: "factor"`, `k_factors`) with RW dynamics (v1: `prior.family: "niw"`).
- Optional `xarray` conversion utilities for labeled outputs (`srvar.xarray`).
- Optional ArviZ conversion utilities for `InferenceData` outputs (`srvar.arviz`).
- ELB-censored backtest evaluation via `evaluation.elb_censor` (censor realized values and optionally forecast draws).
- Streaming backtest evaluation for `metrics.csv` via `output.store_forecasts_in_memory` (reduces RAM for long runs).
- Synthetic memory benchmark script (`scripts/benchmark_backtest_memory.py`).

### Changed

- Refactored `srvar.samplers` into smaller modules and re-exported the public API.
- `forecast()` now requires stored `beta_draws` when `steady_state` is enabled.
- Split config/backtest/evaluation/artifacts logic into `srvar.config`, `srvar.backtest`, `srvar.evaluation`, `srvar.artifacts` (keeping `srvar.runner` as a thin façade).
- Backtest evaluation flags now control both computation and outputs (`evaluation.coverage.enabled`, `evaluation.crps.enabled`, `evaluation.pit.enabled`).

### Fixed

- NIW posterior sampling now correctly handles the univariate case (`N=1`) when drawing from the inverse-Wishart distribution.

## [0.1.0] - 2025-12-22

### Added

- Conjugate NIW Bayesian VAR (BVAR) estimation.
- ELB / shadow-rate data augmentation.
- Diagonal stochastic volatility (SVRW) via KSC mixture + precision-based state sampling.
- Combined SV + ELB model.
- Forecasting API.
- Example scripts in `examples/`.
