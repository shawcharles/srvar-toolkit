# Notebooks

This folder contains expository Jupyter notebooks that mirror the scripts in `examples/`,
with added narrative and labeled outputs.

## Recommended order

1. `00_quickstart_fit_forecast.ipynb` — minimal API walkthrough
2. `01_elb_shadow_rate_basics.ipynb` — ELB/shadow-rate basics
3. `02_factor_sv_end_to_end.ipynb` — FSV + ELB + robust shocks + `xarray`
4. `03_structural_irfs_and_fevd.ipynb` — IRFs/FEVD + `xarray`
5. `04_backtest_and_elb_censored_scoring.ipynb` — ELB-censored evaluation conventions

## Setup

From the repo root:

```bash
python -m pip install -e ".[dev,plot,xarray,arviz]"
```

Launch Jupyter:

```bash
jupyter lab
```

## Runtime knobs

The notebooks are written to run quickly on a laptop. If you want more stable posterior
summaries, increase:

- `sampler.draws`
- `sampler.burn_in`
- `pred_draws` (forecast draws; when applicable)

## Outputs

Notebooks do not write to `outputs/` by default. If you add plots or artifacts, prefer a
subdirectory like `outputs/notebooks/` so it is easy to clean up.
