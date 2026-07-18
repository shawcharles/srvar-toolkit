# Configuration reference

This page is a **complete key-by-key reference** for the YAML configs used by:

- `srvar run` / `srvar validate` (fit + optional forecast)
- `srvar backtest` (rolling/expanding refit + forecast evaluation)

For a narrative introduction, see {doc}`configuration`. For scoring conventions, see {doc}`evaluation`.

## Top-level layout

```yaml
data: ...
model: ...
prior: ...
sampler: ...
forecast: ...    # optional
backtest: ...    # optional
evaluation: ...  # optional (used by backtest)
output: ...
plots: ...       # optional (used by `srvar run`)
```

## `data`

```yaml
data:
  csv_path: "path/to/data.csv"     # required
  date_column: "date"              # required (parsed to DatetimeIndex)
  variables: ["y1", "y2"]          # required (ordered)
  date_format: "%Y-%m-%d"          # optional (pandas format string)
  dropna: true                     # optional (default: true)
```

`dropna: true` removes CSV rows with missing values. Infinite values are never valid training
data. With `dropna: false`, target values may be missing for later backtest evaluation, but every
scheduled training slice must contain only finite values. Remove or impute non-finite values, use
`dropna: true`, or move the backtest origin so its training slice is complete.

For a config containing `backtest`, `srvar validate` validates the prior against the first
scheduled training slice, so target-only missing future rows are allowed. `srvar run` always fits
the full dataset and therefore still requires every value to be finite.

## `model`

```yaml
model:
  p: 4                             # required (VAR lag order)
  include_intercept: true          # optional (default: true)
```

### `model.elb` (shadow-rate augmentation)

```yaml
model:
  elb:
    enabled: true                  # optional (default: true)
    bound: 0.25                    # required when enabled
    applies_to: ["FEDFUNDS"]       # required when enabled
    tol: 1.0e-8                    # optional (default: 1e-8)
    init_offset: 0.05              # optional (default: 0.05)
```

Notes:
- ELB uses **latent shadow-rate sampling** during estimation.
- In forecasts, `ForecastResult.draws` are **observed** (floored) draws and `ForecastResult.latent_draws` are **latent** shadow draws.

### `model.volatility` (stochastic volatility)

```yaml
model:
  volatility:
    enabled: true                  # optional (default: true)

    # State dynamics: "rw" (random walk) or "ar1" (AR(1)).
    dynamics: "rw"                 # optional (default: "rw")

    # Residual covariance:
    # - "diagonal": independent shocks
    # - "triangular": time-invariant correlation structure (CCC-style)
    # - "factor": factor stochastic volatility (full, time-varying covariance; v1: NIW + RW only)
    covariance: "diagonal"         # optional (default: "diagonal")
    q_prior_var: 1.0               # optional (default: 1.0; required positive for triangular)

    # Factor-SV-only settings (used when covariance="factor")
    k_factors: 1                   # optional (default: 1; must be <= N)
    loading_prior_var: 1.0         # optional (default: 1.0; must be > 0)
    store_factor_draws: false      # optional (default: false)

    # KSC mixture stabilization constant used in log(e^2 + epsilon)
    epsilon: 1.0e-4                # optional (default: 1e-4)

    # Priors for initial log-variance state h0 and innovation variances.
    h0_prior_mean: 1.0e-6          # optional (default: 1e-6)
    h0_prior_var: 10.0             # optional (default: 10.0)
    sigma_eta_prior_nu0: 1.0       # optional (default: 1.0)
    sigma_eta_prior_s0: 0.01       # optional (default: 0.01)

    # AR(1)-only priors (used when dynamics="ar1")
    phi_prior_mean: 0.95           # optional (default: 0.95)
    phi_prior_var: 0.1             # optional (default: 0.1)
    gamma0_prior_mean: 0.0         # optional (default: 0.0)
    gamma0_prior_var: 10.0         # optional (default: 10.0)
```

### `model.steady_state` (SSP steady-state parameterization)

```yaml
model:
  steady_state:
    enabled: true                  # optional (default: true)
    mu0: [0.0, 0.0]                # required when enabled (length N)
    v0_mu: 0.1                     # required when enabled (scalar or length N)
    ssvs:
      enabled: false               # optional (default: true)
      spike_var: 0.0001            # optional (default: 1e-4)
      slab_var: 100.0              # optional (default: 100.0)
      inclusion_prob: 0.5          # optional (default: 0.5)
```

Notes:
- `steady_state` requires `include_intercept: true`.
- When enabled, the intercept is parameterized via a steady-state mean `mu`.

### `model.shocks` (robust innovations)

By default, the VAR uses Gaussian innovations. You can switch to robust shock models via
`model.shocks`.

Student‑t innovations:

```yaml
model:
  shocks:
    enabled: true                  # optional (default: true)
    family: "student_t"            # required
    df: 7.0                        # required; must be > 2
```

Outlier-mixture innovations:

```yaml
model:
  shocks:
    enabled: true                  # optional (default: true)
    family: "mixture_outlier"      # required
    outlier_prob: 0.05             # optional (default: 0.05)
    outlier_variance: 10.0         # optional (default: 10.0; must be > 1)
```

Notes / current limitations:
- Robust shocks are supported for **homoskedastic VARs** and for **factor SV**
  (`model.volatility.covariance: "factor"`).
- Robust shocks are not yet supported for diagonal/triangular SV.
- Robust shocks with `model.elb` or `model.steady_state` require factor SV.

## `prior`

```yaml
prior:
  family: "niw"                    # required: "niw" | "ssvs" | "blasso" | "dl"
  method: "minnesota_legacy"       # optional (NIW only): "default" | "minnesota" | "minnesota_legacy" | "minnesota_canonical" | "minnesota_tempered"
```

### NIW defaults

```yaml
prior:
  family: "niw"
  method: "default"
```

### NIW legacy Minnesota-style shrinkage

```yaml
prior:
  family: "niw"
  method: "minnesota_legacy"
  minnesota:
    lambda1: 0.1                   # optional
    lambda2: 0.5                   # optional
    lambda3: 1.0                   # optional
    lambda4: 100.0                 # optional
    own_lag_mean: 1.0              # optional (scalar)
    own_lag_means: [1, 1, 1]       # optional (length N; overrides own_lag_mean)
    min_sigma2: 1.0e-6             # optional (floor for residual variance estimates)
```

Notes:
- `method: "minnesota"` remains supported as a backward-compatible alias for `method: "minnesota_legacy"`.
- `method: "minnesota_legacy"` is the compatibility path and remains the default NIW shrinkage option.

### NIW canonical Minnesota shrinkage

```yaml
prior:
  family: "niw"
  method: "minnesota_canonical"
  minnesota:
    lambda1: 0.1
    lambda2: 0.5
    lambda3: 1.0
    lambda4: 100.0
    own_lag_mean: 1.0
    own_lag_means: [1, 1, 1]
    min_sigma2: 1.0e-6
```

Notes:
- `method: "minnesota_canonical"` uses equation-specific own-vs-cross shrinkage.
- It currently supports homoskedastic models and diagonal stochastic volatility only.
- Triangular SV and factor SV must continue to use the legacy NIW path.

### NIW tempered Minnesota bridge

```yaml
prior:
  family: "niw"
  method: "minnesota_tempered"
  minnesota:
    lambda1: 0.1
    lambda2: 0.5
    lambda3: 1.0
    lambda4: 100.0
    tempered_alpha: 0.25
    own_lag_mean: 1.0
    own_lag_means: [1, 1, 1]
    min_sigma2: 1.0e-6
```

Notes:
- `method: "minnesota_tempered"` is an explicit experimental bridge between the legacy and canonical variance maps.
- `minnesota.tempered_alpha` must be in `[0, 1]`; `0` reproduces the legacy coefficient variances and `1` reproduces the canonical coefficient variances.
- It currently supports diagonal stochastic volatility only.
- It does not redefine the semantics of `method: "minnesota_canonical"`.

### SSVS (variable selection)

```yaml
prior:
  family: "ssvs"
  ssvs:
    spike_var: 0.0001              # optional
    slab_var: 100.0                # optional
    inclusion_prob: 0.5            # optional
    intercept_slab_var: 100.0      # optional
    fix_intercept: true            # optional
```

### Bayesian LASSO (BLASSO)

```yaml
prior:
  family: "blasso"
  blasso:
    mode: "global"                 # optional: "global" | "adaptive"
    tau_init: 10000                # optional
    lambda_init: 2.0               # optional
    a0_global: 1.0                 # optional
    b0_global: 1.0                 # optional
    a0_c: 1.0                      # optional
    b0_c: 1.0                      # optional
    a0_L: 1.0                      # optional
    b0_L: 1.0                      # optional
```

### Dirichlet–Laplace (DL)

```yaml
prior:
  family: "dl"
  dl:
    abeta: 0.5                     # optional
    dl_scaler: 0.1                 # optional
```

## `sampler`

```yaml
sampler:
  draws: 2000                      # optional (default: 2000)
  burn_in: 500                     # optional (default: 500)
  thin: 1                          # optional (default: 1)
  seed: 42                         # optional (if omitted: random seed)
```

## `forecast` (optional; used by `srvar run`)

```yaml
forecast:
  enabled: true                    # optional (default: true)
  horizons: [1, 4, 8, 12]          # required when enabled
  draws: 1000                      # optional (default: 1000)
  quantile_levels: [0.1, 0.5, 0.9] # optional (default shown)

  # Optional stationarity conditioning for VAR coefficient draws:
  # - "allow": do not enforce stability (default)
  # - "reject": require companion eigenvalues < 1 - stationarity_tol
  stationarity: "allow"            # optional (default: "allow")
  stationarity_tol: 1.0e-10        # optional (default: 1e-10)
  stationarity_max_draws: null     # optional (NIW posterior sampling only; default: 50 * draws)
```

Notes:
- Internally, simulations run out to `max(horizons)` and arrays are indexed as `horizon - 1`.

## `backtest` (optional; used by `srvar backtest`)

```yaml
backtest:
  mode: "expanding"                # optional (default: expanding) or "rolling"
  window: 120                      # required when mode="rolling"
  min_obs: 120                     # required
  step: 1                          # optional (default: 1)
  horizons: [1, 3, 6, 12, 24]       # required
  draws: 500                       # optional (default: 500)
  quantile_levels: [0.1, 0.5, 0.9] # optional (default shown)
  origin_start: "2010-01-01"       # optional (requires date index)
  origin_end: "2017-12-01"         # optional (requires date index)

  # Optional stationarity conditioning for VAR coefficient draws (applied at each origin):
  stationarity: "allow"            # optional (default: "allow")
  stationarity_tol: 1.0e-10        # optional (default: 1e-10)
  stationarity_max_draws: null     # optional (NIW posterior sampling only; default: 50 * draws)
```

Notes:
- Internally, forecasts are simulated out to `max(horizons)`.
- `metrics.csv` contains horizons `1..max(horizons)`; plots use the explicit `horizons` list.

## `evaluation` (optional; used by `srvar backtest`)

```yaml
evaluation:
  metrics_table: true
  elb_censor:
    enabled: false
    bound: 0.25
    variables: ["FEDFUNDS"]
    censor_realized: true
    censor_forecasts: false
  coverage:
    enabled: true
    intervals: [0.5, 0.8, 0.9]
    use_latent: false
  crps:
    enabled: true
    use_latent: false
  pit:
    enabled: false
    bins: 10
    variables: ["FEDFUNDS"]
    horizons: [1]
    use_latent: false
```

Notes:
- `coverage.enabled` controls whether `coverage_*` columns/plots are produced.
- `crps.enabled: false` skips CRPS computation and writes `crps=NaN` in `metrics.csv`.
- `pit.horizons` must be a subset of `backtest.horizons` (validated even when PIT is disabled).

## `output`

`srvar run`:

```yaml
output:
  out_dir: "outputs/run"
  save_fit: true                   # optional (default: true)
  save_forecast: true              # optional (default: true)
  save_plots: false                # optional (default: false)
```

`srvar backtest`:

```yaml
output:
  out_dir: "outputs/backtest"
  save_plots: true                 # optional (default: true)
  save_forecasts: false            # optional (default: false)

  # Memory/performance:
  # - true: keep all per-origin forecasts in memory (required for plots)
  # - false: stream metrics without retaining forecast draws (plots not supported)
  store_forecasts_in_memory: true  # optional (default: save_plots)
```

## `plots` (optional; used by `srvar run`)

```yaml
plots:
  variables: ["r"]                 # optional (default: all)
  bands: [0.1, 0.9]                # optional (two floats)
```
