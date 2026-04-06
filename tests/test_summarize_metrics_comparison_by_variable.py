import subprocess
import sys

import pandas as pd


def test_summarize_metrics_comparison_by_variable_script_writes_outputs(tmp_path) -> None:
    comparison_csv = tmp_path / "metrics_comparison.csv"
    pd.DataFrame(
        [
            {
                "variable": "A",
                "horizon": 1,
                "coverage_50_diff": 0.1,
                "coverage_80_diff": 0.0,
                "coverage_90_diff": -0.1,
                "crps_diff": -0.2,
                "rmse_diff": -0.3,
                "mae_diff": -0.1,
            },
            {
                "variable": "A",
                "horizon": 2,
                "coverage_50_diff": 0.1,
                "coverage_80_diff": 0.1,
                "coverage_90_diff": 0.0,
                "crps_diff": -0.4,
                "rmse_diff": 0.1,
                "mae_diff": -0.2,
            },
            {
                "variable": "B",
                "horizon": 1,
                "coverage_50_diff": -0.1,
                "coverage_80_diff": -0.1,
                "coverage_90_diff": -0.1,
                "crps_diff": 0.5,
                "rmse_diff": 0.4,
                "mae_diff": 0.3,
            },
        ]
    ).to_csv(comparison_csv, index=False)

    out_csv = tmp_path / "variable_summary.csv"
    out_md = tmp_path / "variable_summary.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_metrics_comparison_by_variable.py",
            str(comparison_csv),
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
    assert list(df["variable"]) == ["A", "B"]
    row_a = df.loc[df["variable"] == "A"].iloc[0]
    row_b = df.loc[df["variable"] == "B"].iloc[0]
    assert row_a["crps_diff"] == -0.3
    assert row_a["accuracy_improvement_count"] == 3
    assert row_a["coverage_improvement_count"] == 2
    assert row_b["accuracy_improvement_count"] == 0
    assert row_b["coverage_improvement_count"] == 0
