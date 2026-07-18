import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from srvar.artifacts import save_fit_npz
from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _write_run_dir(
    root: Path,
    *,
    csv_path: Path,
    beta_draws: np.ndarray,
    legacy: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("config.yml").write_text(
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
  draws: 4
  burn_in: 0
  thin: 1
  seed: 42
""",
        encoding="utf-8",
    )

    ds = Dataset.from_arrays(
        values=pd.read_csv(csv_path)[["y1", "y2"]].to_numpy(dtype=float),
        variables=["y1", "y2"],
        time_index=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(csv_path)["date"])),
    )
    model = ModelSpec(p=1, include_intercept=True)
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=4, burn_in=0, thin=1)
    fit_res = FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=np.asarray(beta_draws, dtype=float),
    )
    if legacy:
        np.savez_compressed(
            root / "fit_result.npz",
            variables=np.asarray(ds.variables, dtype=object),
            time_index=np.asarray(ds.time_index.to_numpy(), dtype="datetime64[ns]"),
            values=ds.values,
            beta_draws=fit_res.beta_draws,
        )
    else:
        save_fit_npz(root / "fit_result.npz", fit_res)


def test_compare_fit_coefficients_script_writes_summary_and_detail(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=6, freq="QS"),
            "y1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y2": [0.0, 1.0, 0.5, 1.5, 1.0, 2.0],
        }
    ).to_csv(csv_path, index=False)

    baseline_beta = np.array(
        [
            [[0.0, 0.0], [0.2, 0.1], [0.0, 0.3]],
            [[0.1, 0.0], [0.3, 0.2], [0.1, 0.4]],
            [[0.0, 0.1], [0.2, 0.2], [0.0, 0.3]],
            [[0.1, 0.1], [0.3, 0.3], [0.1, 0.4]],
        ],
        dtype=float,
    )
    candidate_beta = np.array(
        [
            [[1.0, 0.0], [0.8, 0.1], [0.0, 0.3]],
            [[1.2, 0.0], [0.9, 0.2], [0.1, 0.4]],
            [[0.8, 0.1], [1.0, 0.2], [0.0, 0.3]],
            [[1.1, 0.1], [0.9, 0.3], [0.1, 0.4]],
        ],
        dtype=float,
    )

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_run_dir(baseline_dir, csv_path=csv_path, beta_draws=baseline_beta)
    _write_run_dir(candidate_dir, csv_path=csv_path, beta_draws=candidate_beta)

    out_csv = tmp_path / "fit_coefficient_summary.csv"
    out_detail_csv = tmp_path / "fit_coefficient_detail.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_fit_coefficients.py",
            str(baseline_dir),
            str(candidate_dir),
            "--cases",
            "y1:const",
            "y1:y1_lag1",
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

    assert list(summary["variable"]) == ["y1", "y1"]
    assert list(summary["regressor"]) == ["const", "y1_lag1"]
    const_row = summary.loc[summary["regressor"] == "const"].iloc[0]
    lag_row = summary.loc[summary["regressor"] == "y1_lag1"].iloc[0]

    assert float(const_row["baseline_mean"]) == 0.05
    assert float(const_row["candidate_mean"]) == 1.025
    assert float(const_row["prob_positive_diff"]) == 0.5
    assert bool(const_row["q80_disjoint"])

    assert float(lag_row["baseline_mean"]) == 0.25
    assert float(lag_row["candidate_mean"]) == 0.9
    assert set(detail["method"]) == {"baseline", "candidate"}
    assert set(detail["regressor"]) == {"const", "y1_lag1"}


def test_compare_fit_coefficients_legacy_flag_is_explicit(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=4, freq="QS"),
            "y1": [1.0, 2.0, 3.0, 4.0],
            "y2": [0.0, 1.0, 0.5, 1.5],
        }
    ).to_csv(csv_path, index=False)
    beta_draws = np.ones((2, 3, 2))
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_run_dir(baseline_dir, csv_path=csv_path, beta_draws=beta_draws, legacy=True)
    _write_run_dir(candidate_dir, csv_path=csv_path, beta_draws=beta_draws, legacy=True)

    command = [
        sys.executable,
        "scripts/compare_fit_coefficients.py",
        str(baseline_dir),
        str(candidate_dir),
    ]
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "legacy pickle-backed" in rejected.stderr

    completed = subprocess.run(
        [*command, "--allow-legacy-pickle"], check=True, capture_output=True, text=True
    )
    assert "wrote_csv=" in completed.stdout
