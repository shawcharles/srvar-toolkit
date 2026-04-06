from __future__ import annotations

import argparse
import json
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


def build_summary_frame(root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    root_path = Path(root)
    for path in sorted(root_path.glob("*/comparison_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = path.parent.name
        metrics = payload.get("metrics", {})
        row: dict[str, object] = {
            "benchmark": benchmark,
            "baseline_method": payload.get("baseline_method"),
            "candidate_method": payload.get("candidate_method"),
            "rows": payload.get("rows"),
            "mode": payload.get("mode"),
        }
        for metric_name, metric_payload in metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            for key in ["baseline_mean", "candidate_mean", "diff_mean", "relative_mean"]:
                if key in metric_payload:
                    row[f"{metric_name}_{key}"] = metric_payload[key]
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"no comparison_summary.json files found under {root_path}")

    return pd.DataFrame(rows).sort_values("benchmark").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize Minnesota comparison bundles into a single CSV/Markdown table."
    )
    ap.add_argument(
        "--root",
        type=str,
        default="outputs/minnesota_comparison",
        help="Directory containing benchmark subdirectories with comparison_summary.json",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="outputs/minnesota_comparison/summary.csv",
        help="Path for consolidated CSV output",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="outputs/minnesota_comparison/summary.md",
        help="Path for consolidated Markdown output",
    )
    args = ap.parse_args()

    df = build_summary_frame(args.root)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_csv.write_text(df.to_csv(index=False), encoding="utf-8")
    out_md.write_text(_render_markdown_table(df), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_md={out_md}")


if __name__ == "__main__":
    main()
