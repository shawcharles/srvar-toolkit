from __future__ import annotations

import argparse

from srvar.compare import run_minnesota_backtest_comparison


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run paired legacy/canonical Minnesota backtests and compare metrics."
    )
    ap.add_argument("config", type=str, help="Base backtest YAML config")
    ap.add_argument(
        "--out-root",
        type=str,
        default="outputs/minnesota_comparison",
        help="Output directory for configs, backtests, and comparison files",
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
        "--mode",
        type=str,
        default="both",
        choices=["ratio", "diff", "both"],
        help="Comparison mode for metrics.csv",
    )
    ap.add_argument(
        "--save-forecasts",
        action="store_true",
        help="Force output.save_forecasts=true in both generated variant configs",
    )
    args = ap.parse_args()

    result = run_minnesota_backtest_comparison(
        args.config,
        out_root=args.out_root,
        baseline_method=args.baseline_method,
        candidate_method=args.candidate_method,
        mode=args.mode,
        save_forecasts=True if args.save_forecasts else None,
    )
    print(f"baseline_dir={result.baseline_out_dir}")
    print(f"candidate_dir={result.candidate_out_dir}")
    print(f"comparison_csv={result.comparison_csv}")
    print(f"summary_json={result.summary_json}")


if __name__ == "__main__":
    main()
