from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / "data" / "example.csv"

    df = pd.read_csv(csv_path)
    dt = pd.to_datetime(df["date"], errors="raise")
    y = df[["r", "y"]].to_numpy(dtype=float, copy=True)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"], time_index=dt)

    model = ModelSpec(p=2, include_intercept=True)
    prior = PriorSpec.niw_minnesota(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=100.0,
        own_lag_mean=0.0,
    )
    sampler = SamplerConfig(draws=1000, burn_in=200, thin=1)

    rng = np.random.default_rng(123)
    fit_res = fit(ds, model, prior, sampler, rng=rng)
    fc = forecast(fit_res, horizons=[1, 4, 8], draws=500, rng=rng)

    print("fit complete (Minnesota prior)")
    print("forecast mean shape:", fc.mean.shape)
    print("median forecast at h=1:", fc.quantiles[0.5][0].tolist())


if __name__ == "__main__":
    main()
