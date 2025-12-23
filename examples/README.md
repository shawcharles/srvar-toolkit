# Examples

This folder contains runnable scripts demonstrating common `srvar-toolkit` workflows.

Most examples use `examples/data/example.csv` so you can run them as-is.

## Recommended order

1. `quickstart_check_version.py`
2. `bvar_basic_fit_forecast.py`
3. `bvar_minnesota_fit_forecast.py`
4. `elb_fit_forecast.py`
5. `sv_fit_forecast.py`
6. `elb_sv_fit_forecast.py`
7. `elb_sv_fit_forecast_plots.py`
8. `blasso.py`
9. `dl_fit_forecast.py`

## Scripts

- `quickstart_check_version.py`
  Prints the installed `srvar-toolkit` version.

- `var_utils.py`
  Demonstrates low-level VAR utilities (design matrix construction and a stationarity check).

- `bvar_basic_fit_forecast.py`
  Loads `examples/data/example.csv`, fits a conjugate BVAR, and produces a small posterior predictive forecast.

- `bvar_minnesota_fit_forecast.py`
  Loads `examples/data/example.csv`, fits a Minnesota-prior BVAR, and forecasts forward.

- `elb_fit_forecast.py`
  Loads `examples/data/example.csv`, fits a shadow-rate (ELB) model, and forecasts forward.

- `sv_fit_forecast.py`
  Loads `examples/data/example.csv`, fits a stochastic-volatility VAR (SVRW), and forecasts forward.

- `elb_sv_fit_forecast.py`
  Loads `examples/data/example.csv`, fits a combined ELB + stochastic volatility model, and forecasts forward.

- `elb_sv_fit_forecast_plots.py`
  Fits an ELB + SV model on a synthetic, time-indexed dataset and writes plots into `outputs/example_plots/`.

- `blasso.py`
  Fits a Bayesian LASSO (BLASSO) VAR with both global and adaptive shrinkage, and prints basic coefficient summaries.

- `dl_fit_forecast.py`
  Loads `examples/data/example.csv`, fits a Dirichlet–Laplace (DL) shrinkage VAR, and produces a small posterior predictive forecast.
