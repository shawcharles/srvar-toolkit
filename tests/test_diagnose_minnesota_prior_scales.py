import json
from pathlib import Path

import pandas as pd

from srvar.compare import run_minnesota_prior_scale_diagnostic


def _write_prior_scale_config(tmp_path: Path) -> tuple[str, str]:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=8, freq="QS"),
            "y1": [0.0, 0.4, 0.8, 0.1, 1.2, 0.3, 1.0, 0.2],
            "y2": [1.0, 1.3, 0.9, 1.4, 0.8, 1.1, 0.7, 1.2],
        }
    ).to_csv(csv_path, index=False)

    config_path = tmp_path / "prior_scale.yml"
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
  draws: 8
  burn_in: 0
  thin: 1
  seed: 42

backtest:
  mode: expanding
  min_obs: 4
  step: 1
  horizons: [1]
  draws: 6
  quantile_levels: [0.1, 0.5, 0.9]
  origin_start: "2001-01-01"
  origin_end: "2001-01-01"

evaluation:
  metrics_table: false
  coverage:
    enabled: false
  crps:
    enabled: false
  pit:
    enabled: false

output:
  save_plots: false
  save_forecasts: false
  store_forecasts_in_memory: false
""",
        encoding="utf-8",
    )
    return str(config_path), "2001-01-01"


def test_run_minnesota_prior_scale_diagnostic_matches_closed_form_ratios(tmp_path: Path) -> None:
    config_path, origin_date = _write_prior_scale_config(tmp_path)
    out_root = tmp_path / "prior_scale_diag"

    result = run_minnesota_prior_scale_diagnostic(
        config_path,
        out_root=out_root,
        origin_date=origin_date,
        cases=["y1:const", "y1:y1_lag1", "y1:y2_lag1"],
    )

    assert result.metadata_json.exists()
    assert result.summary_csv.exists()

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["origin_date"].startswith(origin_date)
    assert metadata["variables"] == ["y1", "y2"]
    assert metadata["cross_weight"] == 0.625

    summary = pd.read_csv(result.summary_csv)
    assert list(summary["regressor"]) == ["const", "y1_lag1", "y2_lag1"]

    const_row = summary.loc[summary["regressor"] == "const"].iloc[0]
    own_row = summary.loc[summary["regressor"] == "y1_lag1"].iloc[0]
    cross_row = summary.loc[summary["regressor"] == "y2_lag1"].iloc[0]

    assert abs(float(const_row["variance_ratio"]) - float(const_row["sigma2_equation"])) < 1e-12
    assert (
        abs(float(const_row["variance_ratio"]) - float(const_row["theoretical_variance_ratio"]))
        < 1e-12
    )
    assert bool(own_row["is_own_lag"])
    assert (
        abs(float(own_row["variance_ratio"]) - float(own_row["theoretical_variance_ratio"])) < 1e-12
    )
    assert not bool(cross_row["is_own_lag"])
    assert (
        abs(float(cross_row["variance_ratio"]) - float(cross_row["theoretical_variance_ratio"]))
        < 1e-12
    )
