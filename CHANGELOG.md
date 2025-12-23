# Changelog

All notable changes to the Python SRVAR toolkit will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- `srvar fetch-fred` command to fetch FRED series to a cached CSV (config-driven).
- `--dry-run` flag for `fetch-fred` (prints planned fetch/output without network calls).
- `--validate-series` flag for `fetch-fred` (preflight series existence check via FRED metadata).
- Table-2 style transformation support in the fetch pipeline, including `processing.transform_order`.
- Runtime warnings for non-positive values when applying log-based tcodes (4/5/6).
- Unit tests covering `tcode_1d` and the `fetch_fred` helpers (mocked, no network).
- Steady-state VAR parameterization (SSP) with Gibbs sampling of the steady-state mean `mu`.
- Optional spike-and-slab selection on `mu` (mu-SSVS).
- YAML-only configuration support for SSP via `model.steady_state`.
- SSP example script (`examples/ssp_fit_forecast.py`).
- SSP test coverage (`tests/test_ssp.py`).

### Changed

- Refactored `srvar.samplers` into smaller modules and re-exported the public API.
- `forecast()` now requires stored `beta_draws` when `steady_state` is enabled.

## [0.1.0] - 2025-12-22

### Added

- Conjugate NIW Bayesian VAR (BVAR) estimation.
- ELB / shadow-rate data augmentation.
- Diagonal stochastic volatility (SVRW) via KSC mixture + precision-based state sampling.
- Combined SV + ELB model.
- Forecasting API.
- Example scripts in `examples/`.
