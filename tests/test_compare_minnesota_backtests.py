import json

import numpy as np
import pandas as pd

from srvar.compare import run_minnesota_backtest_comparison


def _write_base_backtest_config(tmp_path) -> str:
    rng = np.random.default_rng(123)
    t = 36
    beta = np.array(
        [
            [0.0, 0.0],
            [0.55, 0.0],
            [0.0, 0.45],
        ],
        dtype=float,
    )
    sigma = np.diag([0.08, 0.1])

    y = np.zeros((t, 2), dtype=float)
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        y[i] = x @ beta + rng.multivariate_normal(mean=np.zeros(2), cov=sigma)

    dates = pd.date_range("2000-01-01", periods=t, freq="MS")
    df = pd.DataFrame({"date": dates, "y1": y[:, 0], "y2": y[:, 1]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    config_path = tmp_path / "base_backtest.yml"
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

prior:
  family: niw
  method: minnesota_legacy
  minnesota:
    lambda1: 0.2
    lambda2: 0.5
    lambda3: 1.0
    lambda4: 10.0

sampler:
  draws: 30
  burn_in: 10
  thin: 2
  seed: 42

backtest:
  mode: expanding
  min_obs: 24
  step: 6
  horizons: [1, 2]
  draws: 12
  quantile_levels: [0.1, 0.5, 0.9]

evaluation:
  metrics_table: true
  coverage:
    enabled: true
    intervals: [0.5]
    use_latent: false
  crps:
    enabled: true
    use_latent: false
  pit:
    enabled: false

output:
  save_plots: false
  save_forecasts: false
  store_forecasts_in_memory: false
""",
        encoding="utf-8",
    )
    return str(config_path)


def test_run_minnesota_backtest_comparison_writes_bundle(tmp_path) -> None:
    config_path = _write_base_backtest_config(tmp_path)
    out_root = tmp_path / "comparison"

    result = run_minnesota_backtest_comparison(
        config_path,
        out_root=out_root,
        mode="both",
        save_forecasts=True,
    )

    assert result.baseline_config.exists()
    assert result.candidate_config.exists()
    assert result.baseline_out_dir.joinpath("metrics.csv").exists()
    assert result.candidate_out_dir.joinpath("metrics.csv").exists()
    assert result.comparison_csv.exists()
    assert result.summary_json.exists()

    candidate_cfg = result.candidate_config.read_text(encoding="utf-8")
    assert "method: minnesota_canonical" in candidate_cfg
    assert "save_forecasts: true" in candidate_cfg

    baseline_forecasts = sorted(result.baseline_out_dir.joinpath("forecasts").glob("origin_*.npz"))
    candidate_forecasts = sorted(result.candidate_out_dir.joinpath("forecasts").glob("origin_*.npz"))
    assert len(baseline_forecasts) == 2
    assert len(candidate_forecasts) == 2

    comparison = pd.read_csv(result.comparison_csv)
    assert len(comparison) == 4
    assert {"variable", "horizon", "rmse_rel", "rmse_diff", "crps_rel", "crps_diff"} <= set(
        comparison.columns
    )
    assert np.all(np.isfinite(comparison["rmse_rel"]))
    assert np.all(np.isfinite(comparison["rmse_diff"]))

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["baseline_method"] == "minnesota_legacy"
    assert summary["candidate_method"] == "minnesota_canonical"
    assert summary["rows"] == 4
    assert "rmse" in summary["metrics"]
