import pandas as pd


def test_build_relative_metrics_tables_writes_expected_outputs(tmp_path) -> None:
    from papers.carriero2025forecasting.tables import build_relative_metrics_tables

    baseline = pd.DataFrame(
        {
            "variable": ["y", "y", "y"],
            "horizon": [1, 3, 6],
            "rmse": [1.0, 2.0, 4.0],
            "mae": [1.0, 2.0, 4.0],
            "crps": [1.0, 2.0, 4.0],
        }
    )
    cand = pd.DataFrame(
        {
            "variable": ["y", "y", "y"],
            "horizon": [1, 3, 6],
            "rmse": [2.0, 4.0, 8.0],
            "mae": [2.0, 4.0, 8.0],
            "crps": [2.0, 4.0, 8.0],
        }
    )

    out_dir = tmp_path / "tables"
    build_relative_metrics_tables(
        baseline_name="baseline",
        baseline_metrics=baseline,
        candidate_metrics={"candidate": cand},
        horizons=[1, 3],
        out_dir=out_dir,
    )

    pivot = pd.read_csv(out_dir / "relative_rmse_candidate_vs_baseline_pivot.csv", index_col=0)
    assert list(pivot.columns) == ["1", "3"]
    assert float(pivot.loc["y", "1"]) == 2.0
    assert float(pivot.loc["y", "3"]) == 2.0
