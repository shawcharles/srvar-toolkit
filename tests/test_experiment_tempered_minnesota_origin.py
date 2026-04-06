import json
from pathlib import Path

import numpy as np
import pandas as pd

from srvar.compare import make_tempered_canonical_prior, run_tempered_minnesota_origin_experiment
from srvar.spec import ModelSpec, PriorSpec


def _write_tempered_origin_config(tmp_path: Path) -> tuple[str, str]:
    rng = np.random.default_rng(123)
    t = 36
    beta = np.array(
        [
            [0.1, -0.05],
            [0.5, 0.1],
            [0.0, 0.35],
        ],
        dtype=float,
    )
    sigma = np.diag([0.08, 0.1])
    y = np.zeros((t, 2), dtype=float)
    for i in range(1, t):
        x = np.concatenate([np.array([1.0]), y[i - 1]])
        y[i] = x @ beta + rng.multivariate_normal(mean=np.zeros(2), cov=sigma)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=t, freq="QS"),
            "y1": y[:, 0],
            "y2": y[:, 1],
        }
    ).to_csv(csv_path, index=False)

    config_path = tmp_path / "tempered_origin.yml"
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
  draws: 20
  burn_in: 5
  thin: 1
  seed: 42

backtest:
  mode: expanding
  min_obs: 20
  step: 4
  horizons: [1, 2]
  draws: 12
  quantile_levels: [0.1, 0.5, 0.9]
  origin_start: "2004-10-01"
  origin_end: "2004-10-01"

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
    return str(config_path), "2004-10-01"


def test_make_tempered_canonical_prior_geometric_blend() -> None:
    ds = pd.DataFrame({"y1": [0.0, 0.5, 1.0, 0.5, 0.2], "y2": [1.0, 1.2, 0.7, 1.1, 0.8]})
    model = ModelSpec(p=1, include_intercept=True)
    legacy = PriorSpec.niw_minnesota_legacy(
        p=model.p,
        y=ds.to_numpy(dtype=float),
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    canonical = PriorSpec.niw_minnesota_canonical(
        p=model.p,
        y=ds.to_numpy(dtype=float),
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=10.0,
    )
    tempered = make_tempered_canonical_prior(legacy=legacy, canonical=canonical, alpha=0.5)

    legacy_diag = np.diag(legacy.niw.v0)
    legacy_var = np.repeat(legacy_diag.reshape(-1, 1), repeats=2, axis=1)
    canonical_var = 1.0 / canonical.minnesota_canonical.inv_v0_vec.reshape((3, 2), order="F")
    tempered_var = 1.0 / tempered.minnesota_canonical.inv_v0_vec.reshape((3, 2), order="F")

    assert np.allclose(tempered_var, np.sqrt(legacy_var * canonical_var))


def test_run_tempered_minnesota_origin_experiment_writes_outputs(tmp_path: Path) -> None:
    config_path, origin_date = _write_tempered_origin_config(tmp_path)
    out_root = tmp_path / "tempered_origin"

    result = run_tempered_minnesota_origin_experiment(
        config_path,
        out_root=out_root,
        alpha=0.5,
        origin_date=origin_date,
        variables=["y1"],
        horizons=[1, 2],
    )

    assert result.metadata_json.exists()
    assert result.forecast_csv.exists()
    assert result.state_csv.exists()
    assert result.beta_csv.exists()
    assert result.baseline_dir.joinpath("fit_result.npz").exists()
    assert result.canonical_dir.joinpath("fit_result.npz").exists()
    assert result.tempered_dir.joinpath("fit_result.npz").exists()

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["alpha"] == 0.5
    assert metadata["origin_date"].startswith(origin_date)
    assert metadata["forecast_horizons"] == [1, 2]

    forecast = pd.read_csv(result.forecast_csv)
    state = pd.read_csv(result.state_csv)
    beta = pd.read_csv(result.beta_csv)

    assert {"baseline_forecast_mean", "canonical_forecast_mean", "tempered_forecast_mean"} <= set(
        forecast.columns
    )
    assert list(state["variable"]) == ["y1"]
    assert set(beta["variable"]) == {"y1"}
