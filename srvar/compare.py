from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd


def compare_metrics_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("variable", "horizon"),
    mode: Literal["ratio", "diff", "both"] = "ratio",
) -> pd.DataFrame:
    keys = list(key_cols)
    for k in keys:
        if k not in baseline.columns:
            raise ValueError(f"baseline is missing required column: {k}")
        if k not in candidate.columns:
            raise ValueError(f"candidate is missing required column: {k}")

    metric_cols = sorted(set(baseline.columns).intersection(candidate.columns) - set(keys))
    if not metric_cols:
        raise ValueError("no common metric columns found to compare")

    base = baseline.loc[:, keys + metric_cols].copy()
    cand = candidate.loc[:, keys + metric_cols].copy()

    merged = cand.merge(base, on=keys, how="inner", suffixes=("_candidate", "_baseline"))
    if merged.empty:
        raise ValueError("no overlapping rows after merging on key columns")

    out_cols: list[str] = keys.copy()
    for m in metric_cols:
        b = f"{m}_baseline"
        c = f"{m}_candidate"
        out_cols.extend([b, c])

        if mode in {"ratio", "both"}:
            merged[f"{m}_rel"] = merged[c] / merged[b]
            out_cols.append(f"{m}_rel")
        if mode in {"diff", "both"}:
            merged[f"{m}_diff"] = merged[c] - merged[b]
            out_cols.append(f"{m}_diff")

    return merged.loc[:, out_cols]


def compare_metrics_csv(
    baseline_csv: str | Path,
    candidate_csv: str | Path,
    *,
    mode: Literal["ratio", "diff", "both"] = "ratio",
) -> pd.DataFrame:
    base = pd.read_csv(Path(baseline_csv))
    cand = pd.read_csv(Path(candidate_csv))

    for df in (base, cand):
        for col in df.columns:
            if col in {"variable"}:
                continue
            if col in {"horizon"}:
                df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                continue

    return compare_metrics_frames(base, cand, mode=mode)
