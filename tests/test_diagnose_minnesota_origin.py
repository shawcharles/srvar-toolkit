import json

import numpy as np
import pandas as pd

from srvar.compare import run_minnesota_origin_diagnostic


def _write_origin_diagnostic_config(tmp_path) -> tuple[str, str]:
    rng = np.random.default_rng(123)
    t = 40
    beta = np.array(
        [
            [0.1, -0.05],
            [0.45, 0.10],
            [0.05, 0.35],
        ],
        dtype=float,
    )
    y = np.zeros((t, 2), dtype=float)
    sigma = np.diag([0.08, 0.1])
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        y[i] = x @ beta + rng.multivariate_normal(mean=np.zeros(2), cov=sigma)

    dates = pd.date_range("2000-01-01", periods=t, freq="QS")
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"date": dates, "y1": y[:, 0], "y2": y[:, 1]}).to_csv(csv_path, index=False)

    config_path = tmp_path / "origin_backtest.yml"
    config_path.write_text(
        f"""\
data:
  csv_path: {csv_path}
  date_column: date
  variables: [y1, y2]
  dropna: true

model:
  p: 1
  include_intercept: true
  volatility:
    enabled: true
    dynamics: rw
    covariance: diagonal

prior:
  family: niw
  method: minnesota_legacy
  minnesota:
    lambda1: 0.2
    lambda2: 0.5
    lambda3: 1.0
    lambda4: 10.0

sampler:
  draws: 16
  burn_in: 6
  thin: 2
  seed: 42

backtest:
  mode: expanding
  min_obs: 20
  step: 4
  horizons: [1, 2]
  draws: 12
  quantile_levels: [0.1, 0.5, 0.9]
  origin_start: "2004-01-01"
  origin_end: "2007-01-01"

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
  save_forecasts: false
  store_forecasts_in_memory: false
""",
        encoding="utf-8",
    )
    return str(config_path), "2004-10-01"


def test_run_minnesota_origin_diagnostic_writes_artifacts_and_tables(tmp_path) -> None:
    config_path, origin_date = _write_origin_diagnostic_config(tmp_path)
    out_root = tmp_path / "origin_diag"

    result = run_minnesota_origin_diagnostic(
        config_path,
        out_root=out_root,
        origin_date=origin_date,
        variables=["y1"],
        horizons=[1, 2],
    )

    assert result.metadata_json.exists()
    assert result.state_csv.exists()
    assert result.forecast_csv.exists()
    assert result.beta_csv.exists()
    assert result.baseline_out_dir.joinpath("fit_result.npz").exists()
    assert result.candidate_out_dir.joinpath("fit_result.npz").exists()
    assert result.baseline_out_dir.joinpath("forecast_result.npz").exists()
    assert result.candidate_out_dir.joinpath("forecast_result.npz").exists()

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["origin_date"].startswith(origin_date)
    assert metadata["variables"] == ["y1"]
    assert metadata["forecast_horizons"] == [1, 2]
    assert metadata["candidate_stability"] is not None

    state = pd.read_csv(result.state_csv)
    forecast = pd.read_csv(result.forecast_csv)
    beta = pd.read_csv(result.beta_csv)

    assert list(state["variable"]) == ["y1"]
    assert list(forecast["variable"]) == ["y1", "y1"]
    assert list(forecast["horizon"]) == [1, 2]
    assert {"baseline_forecast_mean", "candidate_forecast_mean", "abs_error_diff"} <= set(
        forecast.columns
    )
    assert set(beta["variable"]) == {"y1"}
    assert {"regressor", "baseline_beta_mean", "candidate_beta_mean", "beta_mean_diff"} <= set(
        beta.columns
    )

    candidate_cfg = result.candidate_config.read_text(encoding="utf-8")
    assert "method: minnesota_canonical" in candidate_cfg
