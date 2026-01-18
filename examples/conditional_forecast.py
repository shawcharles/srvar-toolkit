from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset
from srvar.api import fit
from srvar.scenario import conditional_forecast
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
    )
    sampler = SamplerConfig(draws=1000, burn_in=200, thin=1)

    rng = np.random.default_rng(123)
    fit_res = fit(ds, model, prior, sampler, rng=rng)

    # Scenario: pin the short rate at a fixed level for the first 3 months.
    fc = conditional_forecast(
        fit_res,
        horizons=[1, 4, 8, 12],
        constraints={"r": {1: -0.05, 2: -0.05, 3: -0.05}},
        draws=2000,
        rng=rng,
    )

    print("conditional forecast complete")
    print("mean path for r:", np.round(fc.mean[:, 0], 4))


if __name__ == "__main__":
    main()
