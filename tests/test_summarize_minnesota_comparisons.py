import json
import subprocess
import sys

import pandas as pd


def test_summarize_minnesota_comparisons_script_writes_outputs(tmp_path) -> None:
    root = tmp_path / "comparisons"
    (root / "bench_a").mkdir(parents=True)
    (root / "bench_b").mkdir(parents=True)

    payload_a = {
        "baseline_method": "minnesota_legacy",
        "candidate_method": "minnesota_canonical",
        "rows": 8,
        "mode": "both",
        "metrics": {
            "crps": {
                "baseline_mean": 1.0,
                "candidate_mean": 0.9,
                "diff_mean": -0.1,
                "relative_mean": 0.9,
            }
        },
    }
    payload_b = {
        "baseline_method": "minnesota_legacy",
        "candidate_method": "minnesota_canonical",
        "rows": 12,
        "mode": "both",
        "metrics": {
            "rmse": {
                "baseline_mean": 2.0,
                "candidate_mean": 2.2,
                "diff_mean": 0.2,
                "relative_mean": 1.1,
            }
        },
    }
    (root / "bench_a" / "comparison_summary.json").write_text(
        json.dumps(payload_a), encoding="utf-8"
    )
    (root / "bench_b" / "comparison_summary.json").write_text(
        json.dumps(payload_b), encoding="utf-8"
    )

    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_minnesota_comparisons.py",
            "--root",
            str(root),
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
    assert list(df["benchmark"]) == ["bench_a", "bench_b"]
    assert float(df.loc[df["benchmark"] == "bench_a", "crps_diff_mean"].iloc[0]) == -0.1
    assert float(df.loc[df["benchmark"] == "bench_b", "rmse_relative_mean"].iloc[0]) == 1.1
