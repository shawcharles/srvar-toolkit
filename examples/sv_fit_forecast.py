from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / "data" / "example.csv"

    df = pd.read_csv(csv_path)
    dt = pd.to_datetime(df["date"], errors="raise")
    y = df[["r", "y"]].to_numpy(dtype=float, copy=True)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"], time_index=dt)

    model = ModelSpec(p=2, include_intercept=True, volatility=VolatilitySpec())
    prior = PriorSpec.niw_minnesota(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
    )
    sampler = SamplerConfig(draws=800, burn_in=200, thin=1)

    rng = np.random.default_rng(999)
    fit_res = fit(ds, model, prior, sampler, rng=rng)
    fc = forecast(fit_res, horizons=[1, 4, 8], draws=500, rng=rng)

    print("fit complete (SV)")
    print("stored h draws:", None if fit_res.h_draws is None else fit_res.h_draws.shape)
    print("forecast mean shape:", fc.mean.shape)


if __name__ == "__main__":
    main()
