from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from srvar.artifacts import load_forecast_npz
from srvar.config import _prepare_from_config, load_config


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


def _filter_cases(df: pd.DataFrame, cases: list[str] | None) -> pd.DataFrame:
    if not cases:
        return df

    allowed: set[tuple[str, int]] = set()
    for case in cases:
        raw = str(case).strip()
        variable, sep, horizon_s = raw.partition(":")
        if sep != ":" or not variable or not horizon_s:
            raise ValueError(
                f"invalid case {case!r}; expected format VARIABLE:HORIZON, e.g. EXUSUK:1"
            )
        allowed.add((variable, int(horizon_s)))

    mask = [(str(v), int(h)) in allowed for v, h in zip(df["variable"], df["horizon"], strict=True)]
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError("case filter produced zero rows")
    return out


def build_forecast_mean_comparison(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    *,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    baseline_dir = Path(baseline_run_dir)
    candidate_dir = Path(candidate_run_dir)

    cfg = load_config(baseline_dir / "config.yml")
    ds_full, _model, _prior0, _sampler, _rng, _fc = _prepare_from_config(cfg)

    baseline_files = {
        path.name: path for path in sorted((baseline_dir / "forecasts").glob("origin_*.npz"))
    }
    candidate_files = {
        path.name: path for path in sorted((candidate_dir / "forecasts").glob("origin_*.npz"))
    }
    shared_names = sorted(set(baseline_files).intersection(candidate_files))
    if not shared_names:
        raise FileNotFoundError("no overlapping forecast artifacts found")

    if variables is None:
        selected_variables = list(ds_full.variables)
    else:
        selected_variables = [str(v) for v in variables]
        missing = [v for v in selected_variables if v not in ds_full.variables]
        if missing:
            raise ValueError(f"variables not found in dataset: {missing}")

    rows: list[dict[str, object]] = []
    for name in shared_names:
        origin_end_i = int(Path(name).stem.split("_")[1])
        fc_b = load_forecast_npz(baseline_files[name])
        fc_c = load_forecast_npz(candidate_files[name])
        if fc_b.variables != fc_c.variables:
            raise ValueError(f"variable mismatch for forecast artifact {name}")

        for variable in selected_variables:
            v_idx = int(fc_b.variables.index(variable))
            for horizon in fc_b.horizons:
                target_i = origin_end_i + int(horizon)
                if target_i >= ds_full.T:
                    continue
                h_idx = int(horizon) - 1
                realized = float(ds_full.values[target_i, v_idx])
                baseline_mean = float(fc_b.mean[h_idx, v_idx])
                candidate_mean = float(fc_c.mean[h_idx, v_idx])
                rows.append(
                    {
                        "origin": Path(name).stem,
                        "origin_index": origin_end_i,
                        "origin_date": str(ds_full.time_index[origin_end_i]),
                        "target_date": str(ds_full.time_index[target_i]),
                        "variable": variable,
                        "horizon": int(horizon),
                        "realized": realized,
                        "baseline_mean": baseline_mean,
                        "candidate_mean": candidate_mean,
                        "baseline_error": baseline_mean - realized,
                        "candidate_error": candidate_mean - realized,
                        "baseline_abs_error": abs(baseline_mean - realized),
                        "candidate_abs_error": abs(candidate_mean - realized),
                        "mean_shift": candidate_mean - baseline_mean,
                        "abs_error_diff": abs(candidate_mean - realized)
                        - abs(baseline_mean - realized),
                    }
                )

    if not rows:
        raise ValueError("no comparison rows produced")

    return pd.DataFrame(rows).sort_values(
        ["variable", "horizon", "origin_index"]
    ).reset_index(drop=True)


def summarize_forecast_mean_comparison(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["variable", "horizon"]).agg(
        realized_mean=("realized", "mean"),
        baseline_mean_mean=("baseline_mean", "mean"),
        candidate_mean_mean=("candidate_mean", "mean"),
        baseline_error_mean=("baseline_error", "mean"),
        candidate_error_mean=("candidate_error", "mean"),
        baseline_abs_error_mean=("baseline_abs_error", "mean"),
        candidate_abs_error_mean=("candidate_abs_error", "mean"),
        mean_shift_mean=("mean_shift", "mean"),
        abs_error_diff_mean=("abs_error_diff", "mean"),
        candidate_better_count=("abs_error_diff", lambda s: int((s < 0.0).sum())),
        candidate_worse_count=("abs_error_diff", lambda s: int((s > 0.0).sum())),
        rows=("abs_error_diff", "size"),
    )
    return grouped.reset_index().sort_values(["variable", "horizon"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare saved forecast means against realized outcomes origin by origin."
    )
    ap.add_argument("baseline_run_dir", type=str, help="Baseline backtest output directory")
    ap.add_argument("candidate_run_dir", type=str, help="Candidate backtest output directory")
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional subset of variables to summarize",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional case filter in VARIABLE:HORIZON form, e.g. EXUSUK:1 CPIAUCSL:4",
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
        help="Optional path for origin-level detail CSV output",
    )
    ap.add_argument(
        "--out-detail-md",
        type=str,
        default=None,
        help="Optional path for origin-level detail Markdown output",
    )
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_run_dir)
    default_root = baseline_dir.parent
    out_csv = (
        Path(args.out_csv) if args.out_csv is not None else default_root / "forecast_mean_summary.csv"
    )
    out_md = (
        Path(args.out_md) if args.out_md is not None else default_root / "forecast_mean_summary.md"
    )

    comparison = build_forecast_mean_comparison(
        baseline_dir, args.candidate_run_dir, variables=args.variables
    )
    comparison = _filter_cases(comparison, args.cases)
    summary = summarize_forecast_mean_comparison(comparison)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(summary.to_csv(index=False), encoding="utf-8")
    out_md.write_text(_render_markdown_table(summary), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_md={out_md}")

    if args.out_detail_csv is not None:
        out_detail_csv = Path(args.out_detail_csv)
        out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
        out_detail_csv.write_text(comparison.to_csv(index=False), encoding="utf-8")
        print(f"wrote_detail_csv={out_detail_csv}")

    if args.out_detail_md is not None:
        out_detail_md = Path(args.out_detail_md)
        out_detail_md.parent.mkdir(parents=True, exist_ok=True)
        out_detail_md.write_text(_render_markdown_table(comparison), encoding="utf-8")
        print(f"wrote_detail_md={out_detail_md}")


if __name__ == "__main__":
    main()
