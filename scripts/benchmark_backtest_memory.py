from __future__ import annotations

import argparse
import resource
from typing import Any

import numpy as np

from srvar.evaluation import MetricsAccumulator
from srvar.results import ForecastResult


def _max_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KB, macOS reports bytes.
    if rss > 10_000_000:  # heuristically treat as bytes
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _make_eval_cfg(*, coverage: bool, crps: bool) -> dict[str, Any]:
    return {
        "coverage": {"enabled": bool(coverage), "intervals": [0.5, 0.8, 0.9], "use_latent": False},
        "crps": {"enabled": bool(crps), "use_latent": False},
        "pit": {"enabled": False, "bins": 10, "variables": [], "horizons": [], "use_latent": False},
        "elb_censor": {
            "enabled": False,
            "bound": None,
            "variables": [],
            "censor_realized": True,
            "censor_forecasts": False,
        },
        "metrics_table": True,
    }


def _run(*, mode: str, origins: int, draws: int, horizons: int, variables: int) -> None:
    rng = np.random.default_rng(0)
    var_names = [f"y{i + 1}" for i in range(int(variables))]
    horizon_list = list(range(1, int(horizons) + 1))

    eval_cfg = _make_eval_cfg(coverage=False, crps=False)

    if mode == "streaming":
        acc = MetricsAccumulator(variables=var_names, max_h=int(horizons), evaluation=eval_cfg)
        for _ in range(int(origins)):
            yt = rng.normal(size=(int(horizons), int(variables))).astype(float, copy=False)
            sims = rng.normal(size=(int(draws), int(horizons), int(variables))).astype(
                float, copy=False
            )
            fc = ForecastResult(
                variables=var_names,
                horizons=horizon_list,
                draws=sims,
                mean=sims.mean(axis=0),
                quantiles={},
            )
            acc.update(forecast=fc, y_true=yt)
        _ = acc.rows()
    elif mode == "in_memory":
        forecasts: list[ForecastResult] = []
        y_true = np.empty((int(origins), int(horizons), int(variables)), dtype=float)
        for i in range(int(origins)):
            y_true[i] = rng.normal(size=(int(horizons), int(variables)))
            sims = rng.normal(size=(int(draws), int(horizons), int(variables))).astype(
                float, copy=False
            )
            forecasts.append(
                ForecastResult(
                    variables=var_names,
                    horizons=horizon_list,
                    draws=sims,
                    mean=sims.mean(axis=0),
                    quantiles={},
                )
            )
        # Keep objects alive until after printing RSS.
        _ = (forecasts, y_true)
    else:
        raise ValueError("mode must be one of: streaming, in_memory")

    print(f"mode={mode} max_rss_mb={_max_rss_mb():.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest memory benchmark (synthetic).")
    ap.add_argument("--mode", choices=["streaming", "in_memory"], required=True)
    ap.add_argument("--origins", type=int, default=100)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--horizons", type=int, default=24)
    ap.add_argument("--variables", type=int, default=20)
    args = ap.parse_args()

    _run(
        mode=str(args.mode),
        origins=int(args.origins),
        draws=int(args.draws),
        horizons=int(args.horizons),
        variables=int(args.variables),
    )


if __name__ == "__main__":
    main()
