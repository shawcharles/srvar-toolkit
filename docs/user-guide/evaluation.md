# Evaluation and scoring conventions

This page documents how `srvar backtest` evaluates probabilistic forecasts and how to match common macro/interest-rate scoring conventions (including ELB censoring).

## What is evaluated?

For each forecast origin, the backtest produces a predictive distribution for each series and horizon:

- point forecast: `ForecastResult.mean[h-1, j]`
- predictive draws: `ForecastResult.draws[:, h-1, j]`
- optionally (ELB models): latent shadow draws `ForecastResult.latent_draws[:, h-1, j]`

Realized outcomes are taken from the held-out data as `y_true[i, h-1, j]`.

## Metrics in `metrics.csv`

`metrics.csv` is written when `evaluation.metrics_table: true`.

For each variable `j` and horizon `h`, the toolkit reports:

- `rmse`: root mean squared error of the predictive mean
- `mae`: mean absolute error of the predictive mean
- `crps`: mean CRPS over origins (draw-based; `NaN` when disabled)
- `coverage_<p>`: empirical coverage of the central `p%` interval (only when enabled)

Notes:
- Metrics are aggregated **across forecast origins** (one row per variable-horizon).
- Horizons in `metrics.csv` are **`1..max(backtest.horizons)`** (even if `backtest.horizons` is sparse like `[1, 3, 6, 12, 24]`).

## Coverage

Enable/disable via:

```yaml
evaluation:
  coverage:
    enabled: true
    intervals: [0.5, 0.8, 0.9]
    use_latent: false
```

For an interval level `c` (e.g. `0.8`), coverage uses the central interval:

- lower quantile: `qlo = 0.5 - 0.5*c`
- upper quantile: `qhi = 0.5 + 0.5*c`

and reports the mean hit rate across origins:

`1{ qlo <= y_true <= qhi }`.

## PIT

Enable via:

```yaml
evaluation:
  pit:
    enabled: true
    bins: 10
    variables: ["FEDFUNDS"]
    horizons: [1, 12]
    use_latent: false
```

For each selected variable/horizon, the PIT at an origin is:

`u = mean(draws <= y_true)`.

PIT histograms should look approximately uniform for calibrated forecasts.

## CRPS

Enable/disable via:

```yaml
evaluation:
  crps:
    enabled: true
    use_latent: false
```

When disabled, the toolkit **skips CRPS computation** and writes `crps=NaN` in `metrics.csv`.

## ELB-censored evaluation (interest-rate scoring)

Many shadow-rate VAR evaluations treat interest rates as **censored at an effective lower bound (ELB)** when scoring forecasts (e.g. to match the “observed rate” convention in the literature).

To apply this convention at evaluation time, use `evaluation.elb_censor`:

```yaml
evaluation:
  elb_censor:
    enabled: true
    bound: 0.25
    variables: ["FEDFUNDS"]
    censor_realized: true
    censor_forecasts: false
```

Behavior:
- `censor_realized: true`: replaces realized values with `max(y_true, bound)` for the selected variables.
- `censor_forecasts: true`: floors forecast draws at `bound` for the selected variables **before** computing metrics/plots.

This is distinct from `model.elb`:
- `model.elb` changes the **estimation model** (latent augmentation; returns `latent_draws`).
- `evaluation.elb_censor` changes only the **scoring inputs**.

## Latent vs observed scoring (`use_latent`)

When ELB is enabled in the model, forecasts contain:
- `draws`: observed (floored) predictive draws
- `latent_draws`: latent shadow predictive draws

Each diagnostic can choose which to use:

```yaml
evaluation:
  coverage: { use_latent: false }
  crps: { use_latent: false }
  pit: { use_latent: false }
```

Guidance:
- Use `use_latent: false` to score the distribution of the **observed, censored rate** (typical for policy-rate evaluation).
- Use `use_latent: true` when you explicitly want to evaluate the **shadow rate** distribution.

## Memory and streaming evaluation

Backtests can be memory-heavy if you store all per-origin forecast draws.

For long runs, set:

```yaml
output:
  save_plots: false
  store_forecasts_in_memory: false
```

This enables **streaming** evaluation for `metrics.csv` (no need to retain all forecasts in RAM). Plots currently require `store_forecasts_in_memory: true`.

