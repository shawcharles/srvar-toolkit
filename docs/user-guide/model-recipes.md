# Model recipes

This page is a cookbook of common model configurations (API + YAML). It is intentionally example-driven; see {doc}`configuration-reference` for the full schema.

## 1) Baseline BVAR (NIW + Minnesota shrinkage)

YAML:

```yaml
model:
  p: 12
  include_intercept: true

prior:
  family: "niw"
  method: "minnesota"

sampler:
  draws: 2000
  burn_in: 500
  thin: 2
  seed: 123
```

Good starting points:
- `config/minimal_config.yaml`
- `config/demo_config.yaml`

## 2) Minnesota + stochastic volatility (linear SV benchmark)

Use SV when forecast uncertainty changes over time.

YAML (AR(1) log-vol + triangular covariance):

```yaml
model:
  p: 12
  include_intercept: true
  volatility:
    enabled: true
    dynamics: "ar1"
    covariance: "triangular"
    q_prior_var: 1.0
```

Example:
- `config/carriero2025_backtest_15var_linear_sv.yaml`

## 3) Shadow-rate VAR (ELB augmentation)

Enable ELB when an observed rate is censored at an effective lower bound.

```yaml
model:
  p: 12
  include_intercept: true
  elb:
    enabled: true
    bound: 0.25
    applies_to: ["FEDFUNDS"]
```

If you also want SV:

```yaml
model:
  elb: { enabled: true, bound: 0.25, applies_to: ["FEDFUNDS"] }
  volatility: { enabled: true, dynamics: "ar1", covariance: "triangular" }
```

Example:
- `config/carriero2025_backtest_15var_shadow.yaml`

## 4) Steady-state VAR (SSP)

SSP replaces the intercept with a steady-state mean `mu`.

```yaml
model:
  p: 2
  include_intercept: true
  steady_state:
    mu0: [0.0, 0.0]   # length N
    v0_mu: 0.1        # scalar or length N
```

Example:
- `config/ssp_demo_config.yaml`

## 5) Variable selection and shrinkage (SSVS / BLASSO / DL)

SSVS (spike-and-slab selection over predictors):

```yaml
prior:
  family: "ssvs"
  ssvs:
    spike_var: 0.0001
    slab_var: 100.0
    inclusion_prob: 0.5
    fix_intercept: true
```

Bayesian LASSO:

```yaml
prior:
  family: "blasso"
  blasso:
    mode: "global"
    tau_init: 10000
    lambda_init: 2.0
```

Dirichlet–Laplace:

```yaml
prior:
  family: "dl"
  dl:
    abeta: 0.5
    dl_scaler: 0.1
```

## 6) Backtest scaling recipe (memory-friendly)

For large backtests, prefer streaming metrics and disable plots:

```yaml
output:
  save_plots: false
  store_forecasts_in_memory: false
```

This writes `metrics.csv` without retaining per-origin forecast draws in RAM.

## 7) Structural IRFs (Cholesky)

Compute Cholesky-identified impulse responses from posterior draws:

```python
from srvar.analysis import irf_cholesky

irf = irf_cholesky(
    fit_res,
    horizons=24,          # includes horizon 0
    shock_scale="one_sd", # or "unit" for unit-impact normalization
)
```

Notes:
- `ordering=[...]` changes the recursive identification ordering.
- For stochastic volatility models, the impact matrix uses the last volatility state in each draw.

## 8) Conditional / scenario forecasting (hard constraints)

Generate predictive paths conditional on a future path for selected variables:

```python
from srvar.scenario import conditional_forecast

fc_cond = conditional_forecast(
    fit_res,
    horizons=[1, 4, 8, 12],
    constraints={
        # Horizons are 1-indexed steps ahead: 1 means t+1.
        "FEDFUNDS": {1: 0.25, 2: 0.25, 3: 0.25},
    },
    draws=2000,
)
```

Notes:
- This currently supports homoskedastic (time-invariant covariance) VARs.
- When ELB is enabled, constraints are applied to the latent (unfloored) process used for simulation.
