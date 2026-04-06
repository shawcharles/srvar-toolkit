import subprocess
import sys

import numpy as np
import pandas as pd

from srvar.artifacts import save_forecast_npz
from srvar.results import ForecastResult


def _write_config(path, csv_path) -> None:
    path.write_text(
        f"""\
data:
  csv_path: {csv_path}
  date_column: date
  variables: [y]
  dropna: true

model:
  p: 1
  include_intercept: true

prior:
  family: niw
  method: minnesota_legacy
  minnesota:
    lambda1: 0.2
    lambda2: 0.5
    lambda3: 1.0
    lambda4: 10.0

sampler:
  draws: 10
  burn_in: 2
  thin: 1
  seed: 42

backtest:
  mode: expanding
  min_obs: 3
  step: 1
  horizons: [1, 2]
  draws: 4
  quantile_levels: [0.1, 0.5, 0.9]

evaluation:
  metrics_table: true
  coverage:
    enabled: true
    intervals: [0.5]
  crps:
    enabled: true
  pit:
    enabled: false

output:
  save_plots: false
  save_forecasts: true
  store_forecasts_in_memory: false
""",
        encoding="utf-8",
    )


def _write_forecast(path, values) -> None:
    draws = np.asarray(values, dtype=float).reshape(4, 2, 1)
    save_forecast_npz(
        path,
        ForecastResult(
            variables=["y"],
            horizons=[1, 2],
            draws=draws,
            mean=np.mean(draws, axis=0),
            quantiles={0.1: np.quantile(draws, 0.1, axis=0), 0.9: np.quantile(draws, 0.9, axis=0)},
        ),
    )


def test_compare_forecast_means_to_realized_script_writes_outputs(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=6, freq="QS"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(csv_path, index=False)

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    (baseline_dir / "forecasts").mkdir(parents=True)
    (candidate_dir / "forecasts").mkdir(parents=True)
    _write_config(baseline_dir / "config.yml", csv_path)
    _write_config(candidate_dir / "config.yml", csv_path)

    _write_forecast(baseline_dir / "forecasts" / "origin_0002.npz", [3.5, 4.5, 3.5, 4.5, 3.5, 4.5, 3.5, 4.5])
    _write_forecast(candidate_dir / "forecasts" / "origin_0002.npz", [4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0])

    out_csv = tmp_path / "forecast_mean_summary.csv"
    out_md = tmp_path / "forecast_mean_summary.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_forecast_means_to_realized.py",
            str(baseline_dir),
            str(candidate_dir),
            "--variables",
            "y",
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"wrote_csv={out_csv}" in completed.stdout
    assert f"wrote_md={out_md}" in completed.stdout
    df = pd.read_csv(out_csv)
    assert list(df["variable"]) == ["y", "y"]
    assert list(df["horizon"]) == [1, 2]
    assert float(df.loc[df["horizon"] == 1, "abs_error_diff_mean"].iloc[0]) == -0.5
    assert float(df.loc[df["horizon"] == 2, "abs_error_diff_mean"].iloc[0]) == -0.5


def test_compare_forecast_means_to_realized_cases_filter_and_detail_outputs(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=6, freq="QS"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(csv_path, index=False)

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    (baseline_dir / "forecasts").mkdir(parents=True)
    (candidate_dir / "forecasts").mkdir(parents=True)
    _write_config(baseline_dir / "config.yml", csv_path)
    _write_config(candidate_dir / "config.yml", csv_path)

    _write_forecast(
        baseline_dir / "forecasts" / "origin_0002.npz",
        [3.5, 4.5, 3.5, 4.5, 3.5, 4.5, 3.5, 4.5],
    )
    _write_forecast(
        candidate_dir / "forecasts" / "origin_0002.npz",
        [4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0],
    )

    out_csv = tmp_path / "cases_summary.csv"
    out_detail_csv = tmp_path / "cases_detail.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_forecast_means_to_realized.py",
            str(baseline_dir),
            str(candidate_dir),
            "--cases",
            "y:2",
            "--out-csv",
            str(out_csv),
            "--out-detail-csv",
            str(out_detail_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"wrote_csv={out_csv}" in completed.stdout
    assert f"wrote_detail_csv={out_detail_csv}" in completed.stdout

    summary = pd.read_csv(out_csv)
    detail = pd.read_csv(out_detail_csv)
    assert list(summary["horizon"]) == [2]
    assert set(detail["horizon"]) == {2}
