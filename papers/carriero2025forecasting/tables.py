from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom = np.asarray(denom, dtype=float)
    numer = np.asarray(numer, dtype=float)
    out = np.full_like(numer, np.nan, dtype=float)
    ok = np.isfinite(numer) & np.isfinite(denom) & (denom != 0.0)
    out[ok] = numer[ok] / denom[ok]
    return out


def build_relative_metrics_tables(
    *,
    baseline_name: str,
    baseline_metrics: pd.DataFrame,
    candidate_metrics: dict[str, pd.DataFrame],
    horizons: list[int],
    out_dir: str | Path,
) -> None:
    """Build simple relative-metric tables from backtest `metrics.csv` outputs.

    Outputs (per candidate model, per metric):
    - long format: variable,horizon,baseline,candidate,ratio
    - pivot format: variable × horizon table of ratios
    - horizon summary: mean ratio across variables
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    required_cols = {"variable", "horizon"}
    if not required_cols.issubset(set(baseline_metrics.columns)):
        raise ValueError("baseline_metrics must include columns: variable,horizon")

    base = baseline_metrics.copy()
    base["horizon"] = base["horizon"].astype(int)

    horizon_set = {int(h) for h in horizons}
    metrics = [m for m in ["rmse", "mae", "crps"] if m in base.columns]

    for cand_name, cand_df in candidate_metrics.items():
        cand = cand_df.copy()
        cand["horizon"] = cand["horizon"].astype(int)

        for metric in metrics:
            if metric not in cand.columns:
                continue

            merged = (
                base.loc[:, ["variable", "horizon", metric]]
                .rename(columns={metric: "baseline"})
                .merge(
                    cand.loc[:, ["variable", "horizon", metric]].rename(
                        columns={metric: "candidate"}
                    ),
                    on=["variable", "horizon"],
                    how="inner",
                )
            )
            merged = merged.loc[merged["horizon"].isin(horizon_set)].copy()
            merged["ratio"] = _ratio(merged["candidate"].to_numpy(), merged["baseline"].to_numpy())
            merged = merged.sort_values(["variable", "horizon"], kind="mergesort").reset_index(
                drop=True
            )

            stem = f"{metric}_{cand_name}_vs_{baseline_name}"
            merged.to_csv(out / f"relative_{stem}.csv", index=False)

            pivot = merged.pivot(index="variable", columns="horizon", values="ratio")
            pivot = pivot.reindex(columns=sorted(horizon_set))
            pivot.to_csv(out / f"relative_{stem}_pivot.csv")

            summary = (
                merged.groupby("horizon", sort=True)["ratio"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            summary.to_csv(out / f"relative_{stem}_by_horizon.csv", index=False)
