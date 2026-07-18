from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from srvar.compare import build_fit_coefficient_detail, summarize_fit_coefficient_detail


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare posterior coefficient draws between paired baseline and candidate fit runs."
        )
    )
    ap.add_argument("baseline_run_dir", type=str, help="Baseline run directory")
    ap.add_argument("candidate_run_dir", type=str, help="Candidate run directory")
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional subset of equations to summarize",
    )
    ap.add_argument(
        "--regressors",
        nargs="+",
        default=None,
        help="Optional subset of regressors to summarize",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional VARIABLE:REGRESSOR cases, for example EXUSUK:HOUST_lag1",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Path for summary CSV output (default: next to baseline dir)",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Path for summary Markdown output (default: next to baseline dir)",
    )
    ap.add_argument(
        "--out-detail-csv",
        type=str,
        default=None,
        help="Optional path for long-format draw detail CSV output",
    )
    ap.add_argument(
        "--out-detail-md",
        type=str,
        default=None,
        help="Optional path for long-format draw detail Markdown output",
    )
    ap.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="Trusted artifacts only; this can execute pickle code.",
    )
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_run_dir)
    default_root = baseline_dir.parent
    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else default_root / "fit_coefficient_summary.csv"
    )
    out_md = (
        Path(args.out_md)
        if args.out_md is not None
        else default_root / "fit_coefficient_summary.md"
    )

    detail = build_fit_coefficient_detail(
        args.baseline_run_dir,
        args.candidate_run_dir,
        variables=args.variables,
        regressors=args.regressors,
        cases=args.cases,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    summary = summarize_fit_coefficient_detail(detail)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(summary.to_csv(index=False), encoding="utf-8")
    out_md.write_text(_render_markdown_table(summary), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_md={out_md}")

    if args.out_detail_csv is not None:
        out_detail_csv = Path(args.out_detail_csv)
        out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
        out_detail_csv.write_text(detail.to_csv(index=False), encoding="utf-8")
        print(f"wrote_detail_csv={out_detail_csv}")

    if args.out_detail_md is not None:
        out_detail_md = Path(args.out_detail_md)
        out_detail_md.parent.mkdir(parents=True, exist_ok=True)
        out_detail_md.write_text(_render_markdown_table(detail), encoding="utf-8")
        print(f"wrote_detail_md={out_detail_md}")


if __name__ == "__main__":
    main()
