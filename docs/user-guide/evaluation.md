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
- `wis`: mean weighted interval score (WIS) over origins (draw-based; only when enabled)
- `pinball`: mean pinball (quantile) loss over origins (draw-based; only when enabled)
- `log_score`: mean Gaussian log score over origins (draw-based; only when enabled)
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

## WIS (weighted interval score)

Enable/disable via:

```yaml
evaluation:
  wis:
    enabled: true
    intervals: [0.5, 0.8, 0.9]
    use_latent: false
```

When enabled, the toolkit writes a `wis` column in `metrics.csv` (mean WIS over origins). When disabled,
the `wis` column is omitted to keep the default `metrics.csv` schema stable.

## Pinball loss (quantile score)

Enable/disable via:

```yaml
evaluation:
  pinball:
    enabled: true
    quantiles: [0.1, 0.5, 0.9]
    use_latent: false
```

When enabled, the toolkit writes a `pinball` column in `metrics.csv` (mean pinball loss over origins).
When disabled, the `pinball` column is omitted to keep the default `metrics.csv` schema stable.

## Log score (Gaussian LPD)

Enable/disable via:

```yaml
evaluation:
  log_score:
    enabled: true
    variance_floor: 1e-12
    use_latent: false
```

When enabled, the toolkit writes a `log_score` column in `metrics.csv` (mean log score over origins).
When disabled, the `log_score` column is omitted to keep the default `metrics.csv` schema stable.

The current implementation uses a **Gaussian approximation** implied by the predictive draws:

`y ~ Normal(mean(draws), var(draws))`

and reports `log p(y_true)` under that approximation. The `variance_floor` avoids degeneracy when
draws are (nearly) constant.

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

## Model comparison (Diebold–Mariano)

For comparing two loss series (e.g., squared errors or CRPS over forecast origins), use:

```python
from srvar.stats import diebold_mariano_test

res = diebold_mariano_test(loss_model_a, loss_model_b, horizon=12)
print(res.statistic, res.pvalue)
```

This uses a Newey–West/HAC variance estimate (default lag `horizon-1`) and an optional
Harvey–Leybourne–Newbold small-sample correction.

## Model comparison (Giacomini–White)

To match the “CPA test” convention used in some macro forecast-comparison papers (including the
MATLAB replication code included with this repo), use:

```python
from srvar.stats import giacomini_white_test

res = giacomini_white_test(loss_model_a, loss_model_b, horizon=12, choice="conditional")
print(res.statistic, res.pvalue, res.significance_code)
```

`choice="unconditional"` uses a constant instrument (closer to a DM-style test), while
`choice="conditional"` uses a constant plus a lagged loss differential as instruments (the
standard conditional predictive ability setup).

## Forecast combinations (pooling)

For simple forecast combinations (ensembles), you can pool predictive draws across models:

```python
from srvar.ensemble import pool_forecasts

pooled = pool_forecasts([fc_model_a, fc_model_b], weights=[0.5, 0.5], draws=5000)
```

This returns a new `ForecastResult` whose predictive distribution is a weighted mixture of the
input predictive distributions.
