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
```

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
- `metrics_table`: if true, writes `metrics.csv`

## Outputs

Backtest artifacts are written under `output.out_dir` (or `--out`):

- `config.yml`: exact config used
- `metrics.csv`: per-variable, per-horizon summary metrics
- `coverage_all.png`: coverage vs horizon averaged across variables
- `coverage_<var>.png`: coverage vs horizon for each variable
- `pit_<var>_h<h>.png`: PIT histograms for selected variables/horizons
- `crps_by_horizon.png`: CRPS aggregated by horizon
- `backtest_summary.json`: summary metadata (mode, origins, horizons, elapsed time)

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
