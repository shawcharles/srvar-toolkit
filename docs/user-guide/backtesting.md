# Backtesting (rolling/expanding)

This page documents the `srvar backtest` command, which runs a reproducible rolling/expanding refit + forecast evaluation loop from a YAML config.

## Why backtest?

A backtest answers questions like:

- How stable are forecasts across time?
- Are predictive intervals well calibrated (coverage / PIT)?
- How does forecast accuracy evolve with horizon (CRPS/RMSE/MAE)?

## CLI usage

From the repository root:

```bash
# Validate the config (including backtest/evaluation blocks)
srvar validate config/backtest_demo_config.yaml

# Run backtest
srvar backtest config/backtest_demo_config.yaml

# Override output directory
srvar backtest config/backtest_demo_config.yaml --out outputs/my_backtest

# Useful flags
srvar backtest config/backtest_demo_config.yaml --quiet
srvar backtest config/backtest_demo_config.yaml --no-color
srvar backtest config/backtest_demo_config.yaml --verbose

# Paired legacy-vs-canonical Minnesota comparison
python scripts/compare_minnesota_backtests.py config/backtest_demo_config.yaml \
  --out-root outputs/minnesota_comparison
```

The comparison script writes paired `baseline/` and `candidate/` backtest directories plus a
`metrics_comparison.csv` file built from the two `metrics.csv` tables.

To consolidate multiple paired comparison bundles into one table, run:

```bash
python scripts/summarize_minnesota_comparisons.py \
  --root outputs/minnesota_comparison \
  --out-csv outputs/minnesota_comparison/summary.csv \
  --out-md outputs/minnesota_comparison/summary.md
```

This writes one row per benchmark with baseline/candidate metric means and deltas.

To inspect one paired comparison bundle variable by variable across horizons, run:

```bash
python scripts/summarize_metrics_comparison_by_variable.py \
  outputs/minnesota_comparison/vintage_macro15_homoskedastic/metrics_comparison.csv
```

This writes `variable_summary.csv` and `variable_summary.md` alongside the input comparison file.

If the paired backtests were run with saved forecast artifacts, you can also compare predictive
dispersion directly:

```bash
python scripts/compare_minnesota_backtests.py config/vintage_macro15_backtest_diagonal_sv.yaml \
  --out-root outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv \
  --save-forecasts

python scripts/compare_forecast_dispersion.py \
  outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/baseline/forecasts \
  outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/candidate/forecasts
```

This writes `forecast_dispersion_summary.csv` and `forecast_dispersion_summary.md` with
variable/horizon averages for predictive standard deviation and central interval widths.

To compare forecast means against realized outcomes origin by origin from saved forecast bundles:

```bash
python scripts/compare_forecast_means_to_realized.py \
  outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/baseline \
  outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/candidate \
  --variables EXUSUK PPIACO CPIAUCSL M2SL
```

This writes `forecast_mean_summary.csv` and `forecast_mean_summary.md` with mean forecast centers,
signed errors, absolute errors, and counts of origins where the candidate is closer to realized.

For a narrow origin-level deep dive, filter specific cases and write the raw rows:

```bash
python scripts/compare_forecast_means_to_realized.py \
  outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/baseline \
  outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/candidate \
  --cases EXUSUK:1 EXUSUK:2 CPIAUCSL:4 PPIACO:2 M2SL:2 \
  --out-detail-csv outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/case_detail.csv \
  --out-detail-md outputs/minnesota_diagnostics/vintage_macro15_diagonal_sv/case_detail.md
```

To reproduce one scheduled origin as paired baseline/candidate fits, save the fit artifacts,
and compare coefficient means plus end-of-sample volatility states directly:

```bash
python scripts/diagnose_minnesota_origin.py \
  config/vintage_macro15_backtest_diagonal_sv.yaml \
  --out-root outputs/minnesota_origin_diagnostics/vintage_macro15_diagonal_sv_2009q1 \
  --origin-date 2009-01-01 \
  --variables EXUSUK PPIACO CPIAUCSL M2SL \
  --horizons 1 2 4
```

This writes paired `baseline/` and `candidate/` run directories with `fit_result.npz` and
`forecast_result.npz`, plus `state_comparison.csv`, `forecast_comparison.csv`,
`beta_mean_comparison.csv`, and compact Markdown summaries for the state table, forecast table,
and top coefficient deltas.

To compare the full posterior draw distributions for selected coefficients from those paired fit
artifacts, run:

```bash
python scripts/compare_fit_coefficients.py \
  outputs/minnesota_origin_diagnostics/vintage_macro15_diagonal_sv_2009q1/baseline \
  outputs/minnesota_origin_diagnostics/vintage_macro15_diagonal_sv_2009q1/candidate \
  --cases \
    EXUSUK:const EXUSUK:HOUST_lag1 EXUSUK:GS10_lag1 EXUSUK:FEDFUNDS_lag1 \
    PPIACO:const PPIACO:HOUST_lag2 PPIACO:GS10_lag1 PPIACO:CPIAUCSL_lag1
```

This writes `fit_coefficient_summary.csv` and `fit_coefficient_summary.md` with posterior means,
central quantiles, sign probabilities, and simple overlap/flip diagnostics. Optional detail
outputs can be requested to export the long-format draw table.

The three comparison scripts above read saved artifacts strictly by default. If you need to
compare artifacts written before the safe format migration, append `--allow-legacy-pickle` to the
relevant command. Do this only once you have verified the artifacts' source and integrity: the
flag can execute pickle code. For example:

```bash
python scripts/compare_forecast_dispersion.py \
  outputs/trusted_old_bundle/baseline/forecasts \
  outputs/trusted_old_bundle/candidate/forecasts \
  --allow-legacy-pickle

python scripts/compare_forecast_means_to_realized.py \
  outputs/trusted_old_bundle/baseline \
  outputs/trusted_old_bundle/candidate \
  --allow-legacy-pickle

python scripts/compare_fit_coefficients.py \
  outputs/trusted_old_bundle/baseline \
  outputs/trusted_old_bundle/candidate \
  --allow-legacy-pickle
```

To inspect the prior-scale driver directly at one scheduled origin, compare the legacy and
canonical Minnesota coefficient variances:

```bash
python scripts/diagnose_minnesota_prior_scales.py \
  config/vintage_macro15_backtest_diagonal_sv.yaml \
  --out-root outputs/minnesota_prior_scale_diagnostics/vintage_macro15_diagonal_sv_2009q1 \
  --origin-date 2009-01-01 \
  --cases \
    EXUSUK:const EXUSUK:HOUST_lag1 EXUSUK:GS10_lag1 EXUSUK:FEDFUNDS_lag1 EXUSUK:EXUSUK_lag1 \
    PPIACO:const PPIACO:HOUST_lag2 PPIACO:GS10_lag1 PPIACO:CPIAUCSL_lag1 PPIACO:PPIACO_lag1
```

This writes `prior_scale_comparison.csv` plus compact Markdown summaries showing how much looser
or tighter the canonical prior is for each selected coefficient, along with the corresponding
`sigma2` scale terms and the closed-form expected ratio implied by the legacy-vs-canonical
construction.

For a bounded mitigation experiment, you can also run a three-way origin comparison with a
tempered bridge prior between legacy and canonical Minnesota:

```bash
python scripts/experiment_tempered_minnesota_origin.py \
  config/vintage_macro15_backtest_diagonal_sv.yaml \
  --out-root outputs/minnesota_tempered_origin/vintage_macro15_diagonal_sv_2009q1_alpha05 \
  --origin-date 2009-01-01 \
  --alpha 0.5 \
  --variables EXUSUK PPIACO CPIAUCSL M2SL \
  --horizons 1 2 4
```

This writes paired `baseline/`, `canonical/`, and `tempered/` fit/forecast artifacts plus
three-way `forecast_comparison.csv`, `state_comparison.csv`, and `beta_comparison.csv` tables.
`alpha=0` reproduces the legacy variance map, while `alpha=1` reproduces the canonical variance
map. This diagnostic experiment currently requires a diagonal-SV backtest config.

For a larger fully local benchmark dataset, first build the quarterly term-spread/NFCI panel and
then run the tracked config:

```bash
python scripts/prepare_term_nfci_benchmark.py --out data/cache/term_nfci_quarterly.csv
srvar backtest config/term_nfci_backtest.yaml --out outputs/term_nfci_backtest
srvar backtest config/term_nfci_backtest_homoskedastic.yaml \
  --out outputs/term_nfci_backtest_homoskedastic
python scripts/prepare_term_nfci_wuxia_benchmark.py --out data/cache/term_nfci_wuxia_quarterly.csv
srvar backtest config/term_nfci_wuxia_backtest.yaml --out outputs/term_nfci_wuxia_backtest
python scripts/prepare_vintage_macro15_benchmark.py \
  --vintage 2022Q3 \
  --out data/cache/vintage_macro15_quarterly.csv
srvar backtest config/vintage_macro15_backtest_homoskedastic.yaml \
  --out outputs/vintage_macro15_backtest_homoskedastic
srvar backtest config/vintage_macro15_backtest_diagonal_sv.yaml \
  --out outputs/vintage_macro15_backtest_diagonal_sv
```

The vintage-based benchmark uses the repo’s local quarterly vintage workbooks, fixes the source
vintage at `2022Q3`, and applies simple benchmark transforms before backtesting. Tracked
companion configs are provided for both homoskedastic and diagonal-SV comparisons.

## YAML schema (high level)

Backtesting uses the standard model blocks plus two additional sections:

- `backtest`: defines origins, refit policy, horizons, and forecast draw settings
- `evaluation`: defines which diagnostics/plots/exports to produce

See the comment-rich template:

- `config/backtest_demo_config.yaml`

### `backtest`

Common keys:

- `mode`: `expanding` or `rolling`
- `min_obs`: minimum training sample size at the first origin
- `step`: how far to advance the origin each iteration
- `horizons`: list of horizons to evaluate
- `draws`: predictive simulation draws per origin
- `quantile_levels`: quantiles to compute from draws

### `evaluation`

Common keys:

- `coverage`: empirical interval coverage by horizon
- `pit`: PIT histograms for selected variables/horizons
- `crps`: CRPS-by-horizon diagnostic
- `elb_censor`: ELB-censored evaluation (floor realized values and optionally forecasts)
- `metrics_table`: if true, writes `metrics.csv`

For details on scoring conventions (ELB censoring, latent-vs-observed scoring, and horizon semantics), see {doc}`evaluation`.

## Outputs

Backtest artifacts are written under `output.out_dir` (or `--out`):

- `config.yml`: exact config used
- `metrics.csv`: per-variable, per-horizon summary metrics
- `coverage_all.png`: coverage vs horizon averaged across variables
- `coverage_<var>.png`: coverage vs horizon for each variable
- `pit_<var>_h<h>.png`: PIT histograms for selected variables/horizons
- `crps_by_horizon.png`: CRPS aggregated by horizon
- `backtest_summary.json`: summary metadata (mode, origins, horizons, elapsed time)

Notes:
- `metrics.csv` includes horizons `1..max(backtest.horizons)` (even if `backtest.horizons` is sparse).

## ELB-censored evaluation

Some macro forecast evaluations treat interest rates as **censored at an effective lower bound (ELB)** when scoring forecasts (e.g., to match shadow-rate VAR conventions in the literature).

Backtesting supports an **evaluation-time** ELB censoring step via `evaluation.elb_censor`. This is distinct from `model.elb`:

- `model.elb`: affects estimation/forecasting (latent shadow rate + observed floor)
- `evaluation.elb_censor`: affects *only* how realized values and/or forecast draws are scored

Example:

```yaml
evaluation:
  elb_censor:
    enabled: true
    bound: 0.25
    variables: ["FEDFUNDS"]
    censor_realized: true
    censor_forecasts: false
```

Notes:

- When `censor_realized: true`, realized values are replaced by `max(y, bound)` for the selected variables.
- When `censor_forecasts: true`, forecast draws are floored at `bound` for the selected variables *before* metrics/plots are computed.

## Disabling diagnostics

The `enabled` flags control both **computation** and **outputs**:

- `evaluation.coverage.enabled: false` skips coverage computation and omits `coverage_*` columns/plots.
- `evaluation.crps.enabled: false` skips CRPS computation and writes `crps=NaN` in `metrics.csv`.
- `evaluation.pit.enabled: false` skips PIT plots.

## Memory and scaling

Backtests can be memory-intensive if you retain all per-origin forecasts in RAM (especially with many origins, draws, variables, and horizons).

Control retention with:

```yaml
output:
  # If true, keeps all per-origin forecasts in memory (required for plots).
  # If false, metrics are computed in a streaming way (plots not supported yet).
  store_forecasts_in_memory: false
```

Recommended patterns:
- Heavy runs: `save_plots: false` + `store_forecasts_in_memory: false` (fast + memory-light; writes `metrics.csv`).
- Diagnostic runs: `save_plots: true` + `store_forecasts_in_memory: true` (plots + full retention).

## Interpreting the diagnostics

### Coverage

Coverage plots compare **empirical** coverage (y-axis) against the **nominal interval** in the legend.

- Above nominal: intervals are conservative (too wide)
- Below nominal: intervals are too narrow (overconfident)

### PIT

PIT histograms should be approximately uniform under calibration.

- U-shaped: predictive distribution too narrow
- Inverted-U: too wide
- Skew: biased forecasts

### CRPS

CRPS is a proper scoring rule for probabilistic forecasts.

- Lower is better
- Plot typically increases with horizon
