# Quickstart

## Fit a BVAR and forecast

```python
import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig

rng = np.random.default_rng(123)

y = rng.standard_normal((80, 2))
ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

model = ModelSpec(p=2, include_intercept=True)
prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
sampler = SamplerConfig(draws=500, burn_in=100, thin=1)

fit_res = fit(ds, model, prior, sampler, rng=rng)
fc = forecast(fit_res, horizons=[1, 4], draws=200, rng=rng)

print(fc.mean)
```

## CLI + YAML (config-driven run)

The toolkit also supports config-driven runs via the `srvar` CLI.

From the repository root:

```bash
# Validate a config (checks schema, variable names, and basic compatibility)
srvar validate config/demo_config.yaml

# Run fit (+ optional forecast/plots depending on the config)
srvar run config/demo_config.yaml

# Override output directory
srvar run config/demo_config.yaml --out outputs/my_run

# Backtesting (refit + forecast over multiple origins)
srvar backtest config/backtest_demo_config.yaml

# Fetch macro data directly from FRED into a cached CSV
srvar fetch-fred config/fetch_fred_demo_config.yaml

# Preview the planned fetch/output (no network calls)
srvar fetch-fred config/fetch_fred_demo_config.yaml --dry-run

# Preflight-check that series IDs exist (network call)
srvar fetch-fred config/fetch_fred_demo_config.yaml --validate-series
```

Note: `srvar fetch-fred` requires installing the optional `fred` extra (it depends on `fredapi`) and setting a FRED API key (by default via `FRED_API_KEY`).

### What gets written

The run writes outputs to `output.out_dir` (or the `--out` override), for example:

- `config.yml`
- `fit_result.npz`
- `forecast_result.npz` (if forecasting enabled)
- `shadow_rate_*.png`, `volatility_*.png`, `forecast_fan_*.png` (if plot saving enabled)

The backtest writes additional evaluation artifacts (when enabled in the config):

- `metrics.csv`
- `coverage_all.png`, `coverage_<var>.png`
- `pit_<var>_h<h>.png`
- `crps_by_horizon.png`
- `backtest_summary.json`

### Useful flags

- `--quiet`: suppress console output
- `--no-color`: disable ANSI colors in console output
- `--verbose`: show more detailed progress output

For more runnable scripts, see `examples/README.md` in the repository:
https://github.com/shawcharles/srvar-toolkit/blob/main/examples/README.md
