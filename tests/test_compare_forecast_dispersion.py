import subprocess
import sys

import numpy as np
import pandas as pd

from srvar.artifacts import save_forecast_npz
from srvar.results import ForecastResult


def _write_forecast(path, *, scale: float, legacy: bool = False) -> None:
    draws = np.array(
        [
            [[1.0, 2.0], [2.0, 4.0]],
            [[3.0, 4.0], [4.0, 8.0]],
            [[5.0, 6.0], [6.0, 12.0]],
            [[7.0, 8.0], [8.0, 16.0]],
        ],
        dtype=float,
    )
    draws = scale * draws
    mean = np.mean(draws, axis=0)
    quantiles = {0.1: np.quantile(draws, 0.1, axis=0), 0.9: np.quantile(draws, 0.9, axis=0)}
    if legacy:
        np.savez_compressed(
            path,
            variables=np.asarray(["a", "b"], dtype=object),
            horizons=np.asarray([1, 2], dtype=int),
            draws=draws,
            mean=mean,
            **{f"q_{q}": values for q, values in quantiles.items()},
        )
        return
    save_forecast_npz(
        path,
        ForecastResult(
            variables=["a", "b"], horizons=[1, 2], draws=draws, mean=mean, quantiles=quantiles
        ),
    )


def test_compare_forecast_dispersion_script_writes_outputs(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline" / "forecasts"
    candidate_dir = tmp_path / "candidate" / "forecasts"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    _write_forecast(baseline_dir / "origin_0001.npz", scale=1.0)
    _write_forecast(candidate_dir / "origin_0001.npz", scale=2.0)

    out_csv = tmp_path / "dispersion.csv"
    out_md = tmp_path / "dispersion.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_forecast_dispersion.py",
            str(baseline_dir),
            str(candidate_dir),
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
    assert out_csv.exists()
    assert out_md.exists()

    df = pd.read_csv(out_csv)
    row = df.loc[(df["variable"] == "a") & (df["horizon"] == 1)].iloc[0]
    assert row["predictive_std_diff"] > 0.0
    assert row["interval_50_width_diff"] > 0.0
    assert row["interval_80_width_rel"] == 2.0


def test_compare_forecast_dispersion_legacy_flag_is_explicit(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline" / "forecasts"
    candidate_dir = tmp_path / "candidate" / "forecasts"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)
    _write_forecast(baseline_dir / "origin_0001.npz", scale=1.0, legacy=True)
    _write_forecast(candidate_dir / "origin_0001.npz", scale=2.0, legacy=True)

    command = [
        sys.executable,
        "scripts/compare_forecast_dispersion.py",
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
