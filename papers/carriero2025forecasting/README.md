# Carriero et al. (2025) replication harness

This folder contains a lightweight replication entrypoint for the baseline configs shipped in `config/`.

It is intentionally **config-driven**: the “model definitions” live in YAML, and the replication script just orchestrates runs and builds tables from `metrics.csv`.

## Prerequisites

- Install extras needed for YAML + (optionally) FRED fetching:
  - `python -m pip install -e ".[cli]"`
  - `python -m pip install -e ".[cli,fred]"` (to fetch from FRED)
- If fetching from FRED, set `FRED_API_KEY` in your environment (or a local `.env`).

## Run (recommended)

From the repo root:

```bash
python -m papers.carriero2025forecasting.run_replication --out-root outputs/carriero2025
```

This runs:
- `config/carriero2025_backtest_15var_linear_sv.yaml`
- `config/carriero2025_backtest_15var_shadow.yaml`

and writes outputs under:
- `outputs/carriero2025/15var_linear_sv/`
- `outputs/carriero2025/15var_shadow/`
- `outputs/carriero2025/tables/` (relative RMSE/MAE/CRPS vs baseline)

## Fetch the dataset from FRED

If you do not already have the cached dataset CSV (`data/cache/carriero2025_15var.csv`), run:

```bash
python -m papers.carriero2025forecasting.run_replication --fetch-data
```

This uses `config/carriero2025_fetch_fred_15var.yaml`.

