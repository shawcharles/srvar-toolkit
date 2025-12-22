import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit ELB+SV model and save plots")
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=750)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--t", type=int, default=240)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--elb", type=float, default=-0.05)
    parser.add_argument("--out", type=str, default="outputs/example_plots")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    t, n = int(args.t), 2
    elb_bound = float(args.elb)

    r_lat = np.zeros(t, dtype=float)
    y2 = np.zeros(t, dtype=float)
    for i in range(1, t):
        r_lat[i] = 0.97 * r_lat[i - 1] + 0.005 + rng.normal(0.0, 0.02)
        y2[i] = 0.90 * y2[i - 1] + 0.20 * r_lat[i - 1] + rng.normal(0.0, 0.30)

    r_obs = np.maximum(r_lat, elb_bound)
    y = np.column_stack([r_obs, y2])

    dates = pd.date_range(start="2000-01-01", periods=t, freq="MS")
    ds = Dataset.from_arrays(values=y, variables=["r", "y"], time_index=dates)

    model = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=elb_bound, applies_to=["r"]),
        volatility=VolatilitySpec(),
    )
    prior = PriorSpec.niw_minnesota(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
    )
    sampler = SamplerConfig(draws=int(args.draws), burn_in=int(args.burn_in), thin=int(args.thin))

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))
    fc = forecast(fit_res, horizons=[1, 4, 8, 12], draws=1500, rng=np.random.default_rng(2024))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from srvar.plotting import plot_forecast_fanchart, plot_shadow_rate, plot_volatility

    fig, _ax = plot_shadow_rate(fit_res, var="r", bands=(0.1, 0.9))
    fig.savefig(out_dir / "shadow_rate_r.png", dpi=200, bbox_inches="tight")

    fig, _ax = plot_volatility(fit_res, var="r", bands=(0.1, 0.9))
    fig.savefig(out_dir / "volatility_r.png", dpi=200, bbox_inches="tight")

    fig, _ax = plot_forecast_fanchart(fc, var="r", bands=(0.1, 0.9), use_latent=False)
    fig.savefig(out_dir / "forecast_fan_r_observed.png", dpi=200, bbox_inches="tight")

    fig, _ax = plot_forecast_fanchart(fc, var="r", bands=(0.1, 0.9), use_latent=True)
    fig.savefig(out_dir / "forecast_fan_r_shadow.png", dpi=200, bbox_inches="tight")

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
