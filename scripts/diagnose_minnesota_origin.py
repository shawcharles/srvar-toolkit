from __future__ import annotations

import argparse

import pandas as pd

from srvar.compare import run_minnesota_origin_diagnostic


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
            "Run a paired legacy/canonical Minnesota fit diagnostic for one scheduled backtest origin."
        )
    )
    ap.add_argument("config", type=str, help="Base backtest YAML config")
    ap.add_argument(
        "--out-root",
        type=str,
        default="outputs/minnesota_origin_diagnostics",
        help="Output directory for configs, fit/forecast artifacts, and comparison tables",
    )
    ap.add_argument(
        "--origin-date",
        type=str,
        default=None,
        help="Exact scheduled origin date to diagnose (for example 2009-01-01)",
    )
    ap.add_argument(
        "--origin-index",
        type=int,
        default=None,
        help="Exact scheduled origin index to diagnose",
    )
    ap.add_argument(
        "--baseline-method",
        type=str,
        default="minnesota_legacy",
        choices=["minnesota", "minnesota_legacy"],
        help="Baseline NIW Minnesota method",
    )
    ap.add_argument(
        "--candidate-method",
        type=str,
        default="minnesota_canonical",
        choices=["minnesota_canonical"],
        help="Candidate NIW Minnesota method",
    )
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional subset of variables to summarize",
    )
    ap.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of backtest horizons to summarize",
    )
    args = ap.parse_args()

    result = run_minnesota_origin_diagnostic(
        args.config,
        out_root=args.out_root,
        origin_index=args.origin_index,
        origin_date=args.origin_date,
        baseline_method=args.baseline_method,
        candidate_method=args.candidate_method,
        variables=args.variables,
        horizons=args.horizons,
    )

    state_md = result.out_root / "state_comparison.md"
    forecast_md = result.out_root / "forecast_comparison.md"
    beta_top_csv = result.out_root / "beta_top_deltas.csv"
    beta_top_md = result.out_root / "beta_top_deltas.md"

    state_df = pd.read_csv(result.state_csv)
    forecast_df = pd.read_csv(result.forecast_csv)
    beta_df = pd.read_csv(result.beta_csv)
    beta_top = (
        beta_df.assign(abs_beta_mean_diff=lambda df: df["beta_mean_diff"].abs())
        .sort_values(["variable", "abs_beta_mean_diff"], ascending=[True, False])
        .groupby("variable", as_index=False, group_keys=False)
        .head(10)
        .drop(columns=["abs_beta_mean_diff"])
        .reset_index(drop=True)
    )

    state_md.write_text(_render_markdown_table(state_df), encoding="utf-8")
    forecast_md.write_text(_render_markdown_table(forecast_df), encoding="utf-8")
    beta_top_csv.write_text(beta_top.to_csv(index=False), encoding="utf-8")
    beta_top_md.write_text(_render_markdown_table(beta_top), encoding="utf-8")

    print(f"metadata_json={result.metadata_json}")
    print(f"state_csv={result.state_csv}")
    print(f"forecast_csv={result.forecast_csv}")
    print(f"beta_csv={result.beta_csv}")
    print(f"state_md={state_md}")
    print(f"forecast_md={forecast_md}")
    print(f"beta_top_csv={beta_top_csv}")
    print(f"beta_top_md={beta_top_md}")
    print(f"baseline_dir={result.baseline_out_dir}")
    print(f"candidate_dir={result.candidate_out_dir}")


if __name__ == "__main__":
    main()
