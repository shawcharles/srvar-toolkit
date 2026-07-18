from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from srvar.artifacts import load_forecast_npz


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


def build_dispersion_frame(
    forecasts_dir: str | Path, *, allow_legacy_pickle: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    root = Path(forecasts_dir)
    files = sorted(root.glob("origin_*.npz"))
    if not files:
        raise FileNotFoundError(f"no origin_*.npz forecast files found under {root}")

    for path in files:
        fc = load_forecast_npz(path, allow_legacy_pickle=allow_legacy_pickle)
        draws = np.asarray(fc.draws, dtype=float)
        q10 = np.quantile(draws, q=0.10, axis=0)
        q25 = np.quantile(draws, q=0.25, axis=0)
        q75 = np.quantile(draws, q=0.75, axis=0)
        q90 = np.quantile(draws, q=0.90, axis=0)
        std = np.std(draws, axis=0, ddof=0)

        for h_idx, horizon in enumerate(fc.horizons):
            for v_idx, variable in enumerate(fc.variables):
                rows.append(
                    {
                        "origin": path.stem,
                        "variable": variable,
                        "horizon": int(horizon),
                        "predictive_std": float(std[h_idx, v_idx]),
                        "interval_50_width": float(q75[h_idx, v_idx] - q25[h_idx, v_idx]),
                        "interval_80_width": float(q90[h_idx, v_idx] - q10[h_idx, v_idx]),
                    }
                )

    return pd.DataFrame(rows).sort_values(["origin", "variable", "horizon"]).reset_index(drop=True)


def compare_dispersion_dirs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    allow_legacy_pickle: bool = False,
) -> pd.DataFrame:
    base = build_dispersion_frame(baseline_dir, allow_legacy_pickle=allow_legacy_pickle)
    cand = build_dispersion_frame(candidate_dir, allow_legacy_pickle=allow_legacy_pickle)
    merged = cand.merge(
        base,
        on=["origin", "variable", "horizon"],
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    if merged.empty:
        raise ValueError("no overlapping origin/variable/horizon rows found")

    for metric in ("predictive_std", "interval_50_width", "interval_80_width"):
        merged[f"{metric}_diff"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
        merged[f"{metric}_rel"] = merged[f"{metric}_candidate"] / merged[f"{metric}_baseline"]
    return merged


def summarize_dispersion_comparison(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["variable", "horizon"])
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["variable", "horizon"])
        .reset_index(drop=True)
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare saved backtest forecast dispersion between baseline and candidate bundles."
        )
    )
    ap.add_argument("baseline_forecasts", type=str, help="Baseline forecasts directory")
    ap.add_argument("candidate_forecasts", type=str, help="Candidate forecasts directory")
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Path for variable/horizon-level CSV output (default: next to baseline dir)",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Path for variable/horizon-level Markdown output (default: next to baseline dir)",
    )
    ap.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="Trusted artifacts only; this can execute pickle code.",
    )
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_forecasts)
    candidate_dir = Path(args.candidate_forecasts)
    default_root = baseline_dir.parent.parent
    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else default_root / "forecast_dispersion_summary.csv"
    )
    out_md = (
        Path(args.out_md)
        if args.out_md is not None
        else default_root / "forecast_dispersion_summary.md"
    )

    comparison = compare_dispersion_dirs(
        baseline_dir, candidate_dir, allow_legacy_pickle=args.allow_legacy_pickle
    )
    summary = summarize_dispersion_comparison(comparison)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(summary.to_csv(index=False), encoding="utf-8")
    out_md.write_text(_render_markdown_table(summary), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_md={out_md}")


if __name__ == "__main__":
    main()
