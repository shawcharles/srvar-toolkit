from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from srvar.backtest import backtest_from_config
from srvar.config import ConfigError, load_config

try:  # support both `python -m papers.carriero2025forecasting.run_replication` and direct execution
    from .tables import build_relative_metrics_tables
except Exception:  # pragma: no cover
    from tables import build_relative_metrics_tables  # type: ignore


ROOT = Path(__file__).resolve().parents[2]


MODEL_CONFIGS: dict[str, Path] = {
    "15var_linear_sv": ROOT / "config" / "carriero2025_backtest_15var_linear_sv.yaml",
    "15var_shadow": ROOT / "config" / "carriero2025_backtest_15var_shadow.yaml",
}

FETCH_CONFIGS: dict[str, Path] = {
    "15var": ROOT / "config" / "carriero2025_fetch_fred_15var.yaml",
    "20var": ROOT / "config" / "carriero2025_fetch_fred_20var.yaml",
}


def _ensure_carriero_dataset(*, which: str, overwrite: bool) -> Path | None:
    cfg_path = FETCH_CONFIGS.get(which)
    if cfg_path is None:
        raise ValueError(f"unknown dataset key: {which}")

    try:
        cfg = load_config(cfg_path)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return None

    output_cfg = cfg.get("output", {})
    if not isinstance(output_cfg, dict):
        print(f"Invalid fetch config (output must be a mapping): {cfg_path}", file=sys.stderr)
        return None

    csv_path = output_cfg.get("csv_path")
    if not isinstance(csv_path, str) or not csv_path:
        print(f"Invalid fetch config (output.csv_path missing): {cfg_path}", file=sys.stderr)
        return None

    out_csv = Path(csv_path)
    if out_csv.exists() and not overwrite:
        return out_csv

    try:
        from srvar.data.fetch_fred import fetch_fred_to_csv
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return None

    try:
        out_csv_p, meta_p, df = fetch_fred_to_csv(cfg, overwrite=overwrite)
    except Exception as exc:
        print(f"Failed to fetch dataset ({cfg_path}): {exc}", file=sys.stderr)
        return None

    print(f"Wrote dataset: {out_csv_p} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Wrote metadata: {meta_p}")
    return out_csv_p


def _read_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variable" not in df.columns or "horizon" not in df.columns:
        raise ValueError(f"unexpected metrics schema: {path}")
    df["horizon"] = df["horizon"].astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Carriero et al. (2025) replication harness (baseline configs)."
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "carriero2025",
        help="Root directory for replication outputs.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS.keys()),
        default=["15var_linear_sv", "15var_shadow"],
        help="Which model configs to run.",
    )
    ap.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch the Carriero dataset from FRED before running (requires FRED_API_KEY).",
    )
    ap.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Overwrite the cached dataset CSV when using --fetch-data.",
    )
    ap.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip running backtests (only build tables from existing outputs).",
    )
    ap.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 6, 12, 24],
        help="Horizons to include in comparison tables.",
    )
    args = ap.parse_args()

    out_root: Path = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.fetch_data:
        _ = _ensure_carriero_dataset(which="15var", overwrite=bool(args.overwrite_data))

    if not args.skip_backtest:
        for name in list(args.models):
            cfg_path = MODEL_CONFIGS[name]
            out_dir = out_root / name
            print(f"Running backtest: {name} ({cfg_path}) -> {out_dir}")
            try:
                backtest_from_config(cfg_path, out_dir=out_dir)
            except ImportError as exc:
                msg = str(exc)
                if "PyYAML" in msg:
                    msg = f"{msg}\n\nInstall CLI deps first:\n  python -m pip install -e '.[cli]'\n"
                print(msg, file=sys.stderr)
                raise
            except (ConfigError, ValueError) as exc:
                msg = str(exc)
                if "data.csv_path not found" in msg:
                    msg = (
                        f"{msg}\n\n"
                        "If you want this script to fetch the dataset from FRED, re-run with:\n"
                        "  --fetch-data\n"
                    )
                print(f"Backtest failed for {name}: {msg}", file=sys.stderr)
                raise

    # Build relative tables vs baseline.
    baseline = "15var_linear_sv"
    candidates = [m for m in list(args.models) if m != baseline]
    if candidates:
        baseline_metrics = _read_metrics_csv(out_root / baseline / "metrics.csv")
        candidate_metrics = {m: _read_metrics_csv(out_root / m / "metrics.csv") for m in candidates}
        tables_dir = out_root / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        build_relative_metrics_tables(
            baseline_name=baseline,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            horizons=list(args.horizons),
            out_dir=tables_dir,
        )
        print(f"Wrote tables: {tables_dir}")


if __name__ == "__main__":
    main()
