# Examples

This folder contains runnable scripts demonstrating common `srvar-toolkit` workflows.

## Recommended order

1. `quickstart_check_version.py`
2. `bvar_basic_fit_forecast.py`
3. `bvar_minnesota_fit_forecast.py`
4. `elb_fit_forecast.py`
5. `sv_fit_forecast.py`
6. `elb_sv_fit_forecast.py`
7. `elb_sv_fit_forecast_plots.py`

## Scripts

- `quickstart_check_version.py`
  Prints the installed `srvar-toolkit` version.

- `var_utils.py`
  Minimal utilities demo (design matrix construction and a stationarity check).

- `bvar_basic_fit_forecast.py`
  Fits a simple conjugate BVAR and produces a small posterior predictive forecast.

- `bvar_minnesota_fit_forecast.py`
  Fits a BVAR with a Minnesota-style prior and produces a small forecast.

- `elb_fit_forecast.py`
  Fits a shadow-rate (ELB) model on a toy dataset and forecasts forward.

- `sv_fit_forecast.py`
  Fits a stochastic-volatility VAR (SVRW) on a toy dataset and forecasts forward.

- `elb_sv_fit_forecast.py`
  Fits a combined ELB + stochastic volatility model and forecasts forward.

- `elb_sv_fit_forecast_plots.py`
  Fits an ELB + SV model on a synthetic dataset with a time index and saves plots.

- `replication_choose_var1.py`
  A larger replication-style script (more research-oriented). Useful as a template for experiments.

## Note on legacy filenames

You may also see `phase*.py` scripts. Those were used during an earlier refactor and are kept for backward compatibility, but the scripts listed above are the recommended entrypoints.
