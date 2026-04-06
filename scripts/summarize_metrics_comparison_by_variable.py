from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _render_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in df.columns]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body_rows: list[str] = []
        for row in df.itertuples(index=False, name=None):
            values = ["" if pd.isna(value) else str(value) for value in row]
            body_rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *body_rows])


def build_variable_summary_frame(metrics_comparison_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_comparison_csv)
    if "variable" not in df.columns:
        raise ValueError("metrics comparison must include a 'variable' column")

    grouped = df.groupby("variable").mean(numeric_only=True).reset_index()

    for col in ("crps_diff", "rmse_diff", "mae_diff"):
        if col not in grouped.columns:
            raise ValueError(f"metrics comparison is missing required column: {col}")

    accuracy_cols = ["crps_diff", "rmse_diff", "mae_diff"]
    coverage_cols = [c for c in ("coverage_50_diff", "coverage_80_diff", "coverage_90_diff") if c in grouped]
    grouped["accuracy_improvement_count"] = (grouped.loc[:, accuracy_cols] < 0.0).sum(axis=1)
    grouped["coverage_improvement_count"] = (grouped.loc[:, coverage_cols] > 0.0).sum(axis=1)

    return grouped.sort_values(["crps_diff", "rmse_diff", "mae_diff", "variable"]).reset_index(
        drop=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize a Minnesota metrics_comparison.csv file by variable across horizons."
        )
    )
    ap.add_argument(
        "comparison_csv",
        type=str,
        help="Path to metrics_comparison.csv produced by compare_minnesota_backtests.py",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Path for variable-level CSV output (default: alongside input)",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Path for variable-level Markdown output (default: alongside input)",
    )
    args = ap.parse_args()

    comparison_csv = Path(args.comparison_csv)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else comparison_csv.with_name("variable_summary.csv")
    )
    out_md = (
        Path(args.out_md) if args.out_md is not None else comparison_csv.with_name("variable_summary.md")
    )

    df = build_variable_summary_frame(comparison_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(df.to_csv(index=False), encoding="utf-8")
    out_md.write_text(_render_markdown_table(df), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_md={out_md}")


if __name__ == "__main__":
    main()
