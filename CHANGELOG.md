# Changelog

All notable changes to the Python SRVAR toolkit will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

## [0.3.0] - 2026-04-06

### Added

- Explicit legacy Minnesota-style NIW prior path via `PriorSpec.niw_minnesota_legacy(...)` and
  `prior.method: "minnesota_legacy"` in YAML configs. `PriorSpec.niw_minnesota(...)` and
  `prior.method: "minnesota"` remain as backward-compatible aliases.
- Explicit canonical Minnesota prior path via `PriorSpec.niw_minnesota_canonical(...)` and
  `prior.method: "minnesota_canonical"` for homoskedastic models and diagonal stochastic
  volatility.
- Explicit experimental tempered Minnesota bridge via `PriorSpec.niw_minnesota_tempered(...)`
  and `prior.method: "minnesota_tempered"` for diagonal stochastic-volatility models.
- Reproducible Minnesota backtest comparison harness via
  `scripts/compare_minnesota_backtests.py`, which runs paired legacy/canonical backtests and
  writes a combined `metrics_comparison.csv`.
- Consolidated Minnesota benchmark summary script via
  `scripts/summarize_minnesota_comparisons.py`, which scans paired comparison bundles and writes
  repo-level `summary.csv` and `summary.md` tables.
- Variable-level Minnesota comparison summary script via
  `scripts/summarize_metrics_comparison_by_variable.py`, which aggregates one
  `metrics_comparison.csv` file across horizons and writes `variable_summary.csv` and
  `variable_summary.md`.
- Forecast-dispersion comparison script via `scripts/compare_forecast_dispersion.py`, plus a
  `--save-forecasts` option on `scripts/compare_minnesota_backtests.py` for diagnostic reruns
  that need per-origin predictive draw artifacts.
- Forecast-mean-vs-realized comparison script via `scripts/compare_forecast_means_to_realized.py`
  for origin-by-origin diagnostic summaries from saved forecast bundles.
- `scripts/compare_forecast_means_to_realized.py` now supports `--cases` and optional detail
  outputs for narrow origin-level deep dives on selected variable/horizon pairs.
- Single-origin Minnesota fit diagnostic via `scripts/diagnose_minnesota_origin.py`, which
  reproduces one scheduled backtest origin as paired baseline/candidate fits and writes fit
  artifacts plus state, forecast, and coefficient comparison tables.
- Posterior coefficient-draw comparison script via `scripts/compare_fit_coefficients.py` for
  selected `VARIABLE:REGRESSOR` cases from paired fit runs.
- Prior-scale diagnostic via `scripts/diagnose_minnesota_prior_scales.py` for comparing legacy
  and canonical Minnesota coefficient variances at one scheduled backtest origin.
- Tempered-origin experiment via `scripts/experiment_tempered_minnesota_origin.py`, which runs a
  three-way legacy/canonical/tempered Minnesota comparison for one scheduled origin.
- Local quarterly benchmark prep/config via `scripts/prepare_term_nfci_benchmark.py` and
  `config/term_nfci_backtest.yaml` for a second fully local Minnesota comparison run.
- Homoskedastic companion benchmark config via `config/term_nfci_backtest_homoskedastic.yaml`
  to compare canonical vs legacy Minnesota on the same local panel without stochastic volatility.
- Richer three-variable local benchmark prep/config via
  `scripts/prepare_term_nfci_wuxia_benchmark.py` and `config/term_nfci_wuxia_backtest.yaml`.
- Local transformed quarterly 15-variable macro benchmark prep/config via
  `scripts/prepare_vintage_macro15_benchmark.py` and
  `config/vintage_macro15_backtest_homoskedastic.yaml`.
- Diagonal-SV companion config for the local transformed 15-variable vintage benchmark via
  `config/vintage_macro15_backtest_diagonal_sv.yaml`.

### Changed

- Source docs and example configs now label the shipped Minnesota-style NIW construction as a
  legacy, non-canonical compatibility path, and document the support boundary for the explicit
  canonical path.

### Fixed

- Backtest metrics and plotting diagnostics now exclude missing realized values from evaluation
  denominators instead of treating them as misses.
- `config/backtest_demo_config.yaml` now parses as valid YAML again; the sample had a top-level
  indentation error before the `output` block.

## [0.2.0] - 2026-01-19

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
- Robust shock models for homoskedastic VARs and factor SV via `model.shocks` (Student‑t and outlier-mixture innovations).
- Full-covariance stochastic volatility via factor SV (`model.volatility.covariance: "factor"`, `k_factors`) with RW dynamics (v1: `prior.family: "niw"`), including ELB shadow-rate data augmentation and steady-state support.
- Structural analysis stack (`srvar.analysis`, `srvar.identification`) including Cholesky IRFs, sign-restricted IRFs, FEVD, and historical decomposition (supports factor SV covariance draws).
- Conditional / scenario forecasting utilities (`srvar.scenario`).
- Optional `xarray` conversion utilities for labeled outputs (`srvar.xarray`).
- Optional ArviZ conversion utilities for `InferenceData` outputs (`srvar.arviz`).
- Labeled output hardening for factor SV loadings (`FitResult.loading_draws` and `ds_fit["loadings"]` alias of `ds_fit["lambda"]`).
- Expository notebooks in `examples/notebooks/` (quickstart, ELB, FSV, structural analysis, and backtesting/evaluation conventions).
- Factor SV demo config (`config/fsv_demo_config.yaml`) and example script (`examples/fsv_fit_forecast.py`).
- `srvar.artifacts.load_run_dir(out_dir)` to reconstruct a `FitResult` from `config.yml` + `fit_result.npz` (including factor SV draws and optional latent dataset / NIW posterior blocks).
- ELB-censored backtest evaluation via `evaluation.elb_censor` (censor realized values and optionally forecast draws).
- Streaming backtest evaluation for `metrics.csv` via `output.store_forecasts_in_memory` (reduces RAM for long runs).
- Additional scoring rules and comparison utilities:
  - weighted interval score (WIS), pinball (quantile) loss, and Gaussian log score / LPD approximation (`srvar.metrics`, `srvar.evaluation`)
  - Diebold–Mariano test and Giacomini–White CPA test (`srvar.stats`)
- Simple forecast pooling / ensembles (`srvar.ensemble`).
- Synthetic memory benchmark script (`scripts/benchmark_backtest_memory.py`).

### Changed

- Refactored `srvar.samplers` into smaller modules and re-exported the public API.
- `forecast()` now requires stored `beta_draws` when `steady_state` is enabled.
- Split config/backtest/evaluation/artifacts logic into `srvar.config`, `srvar.backtest`, `srvar.evaluation`, `srvar.artifacts` (keeping `srvar.runner` as a thin façade).
- Backtest evaluation flags now control both computation and outputs (`evaluation.coverage.enabled`, `evaluation.crps.enabled`, `evaluation.pit.enabled`).

### Fixed

- NIW posterior sampling now correctly handles the univariate case (`N=1`) when drawing from the inverse-Wishart distribution.
- `srvar.xarray.fit_to_xarray` now handles SV/FSV time-varying draws defined on the effective sample (`T - p`).

## [0.1.0] - 2025-12-22

### Added

- Conjugate NIW Bayesian VAR (BVAR) estimation.
- ELB / shadow-rate data augmentation.
- Diagonal stochastic volatility (SVRW) via KSC mixture + precision-based state sampling.
- Combined SV + ELB model.
- Forecasting API.
- Example scripts in `examples/`.
