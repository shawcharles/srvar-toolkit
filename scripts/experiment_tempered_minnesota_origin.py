from __future__ import annotations

import argparse

from srvar.compare import run_tempered_minnesota_origin_experiment


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run a three-way legacy/canonical/tempered Minnesota origin experiment "
            "for one scheduled backtest origin."
        )
    )
    ap.add_argument("config", type=str, help="Base backtest YAML config")
    ap.add_argument(
        "--out-root",
        type=str,
        default="outputs/minnesota_tempered_origin",
        help="Output directory for experiment artifacts and comparison tables",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Geometric blend weight between legacy (0.0) and canonical (1.0)",
    )
    ap.add_argument("--origin-date", type=str, default=None, help="Exact scheduled origin date")
    ap.add_argument("--origin-index", type=int, default=None, help="Exact scheduled origin index")
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
        help="Optional subset of horizons to summarize",
    )
    args = ap.parse_args()

    result = run_tempered_minnesota_origin_experiment(
        args.config,
        out_root=args.out_root,
        alpha=args.alpha,
        origin_index=args.origin_index,
        origin_date=args.origin_date,
        variables=args.variables,
        horizons=args.horizons,
    )

    print(f"metadata_json={result.metadata_json}")
    print(f"forecast_csv={result.forecast_csv}")
    print(f"state_csv={result.state_csv}")
    print(f"beta_csv={result.beta_csv}")
    print(f"baseline_dir={result.baseline_dir}")
    print(f"canonical_dir={result.canonical_dir}")
    print(f"tempered_dir={result.tempered_dir}")


if __name__ == "__main__":
    main()
