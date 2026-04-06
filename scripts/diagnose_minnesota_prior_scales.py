from __future__ import annotations

import argparse

import pandas as pd

from srvar.compare import run_minnesota_prior_scale_diagnostic


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
            "Diagnose legacy-vs-canonical Minnesota prior variances for one scheduled backtest origin."
        )
    )
    ap.add_argument("config", type=str, help="Base backtest YAML config")
    ap.add_argument(
        "--out-root",
        type=str,
        default="outputs/minnesota_prior_scale_diagnostics",
        help="Output directory for configs and prior-scale comparison tables",
    )
    ap.add_argument(
        "--origin-date",
        type=str,
        default=None,
        help="Exact scheduled origin date to diagnose",
    )
    ap.add_argument(
        "--origin-index",
        type=int,
        default=None,
        help="Exact scheduled origin index to diagnose",
    )
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
    args = ap.parse_args()

    result = run_minnesota_prior_scale_diagnostic(
        args.config,
        out_root=args.out_root,
        origin_index=args.origin_index,
        origin_date=args.origin_date,
        variables=args.variables,
        regressors=args.regressors,
        cases=args.cases,
    )

    summary = pd.read_csv(result.summary_csv)
    summary_md = result.out_root / "prior_scale_comparison.md"
    top_csv = result.out_root / "prior_scale_top.csv"
    top_md = result.out_root / "prior_scale_top.md"

    top = (
        summary.assign(abs_log_variance_ratio=lambda df: df["log_variance_ratio"].abs())
        .sort_values(["variable", "abs_log_variance_ratio"], ascending=[True, False])
        .groupby("variable", as_index=False, group_keys=False)
        .head(10)
        .drop(columns=["abs_log_variance_ratio"])
        .reset_index(drop=True)
    )

    summary_md.write_text(_render_markdown_table(summary), encoding="utf-8")
    top_csv.write_text(top.to_csv(index=False), encoding="utf-8")
    top_md.write_text(_render_markdown_table(top), encoding="utf-8")

    print(f"metadata_json={result.metadata_json}")
    print(f"summary_csv={result.summary_csv}")
    print(f"summary_md={summary_md}")
    print(f"top_csv={top_csv}")
    print(f"top_md={top_md}")


if __name__ == "__main__":
    main()
