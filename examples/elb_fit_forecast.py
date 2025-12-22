from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / "data" / "example.csv"

    df = pd.read_csv(csv_path)
    dt = pd.to_datetime(df["date"], errors="raise")
    y = df[["r", "y"]].to_numpy(dtype=float, copy=True)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"], time_index=dt)

    elb_bound = -0.05
    model = ModelSpec(p=2, include_intercept=True, elb=ElbSpec(bound=elb_bound, applies_to=["r"]))
    k = (1 if model.include_intercept else 0) + ds.N * model.p
    prior = PriorSpec.niw_minnesota(
        p=model.p,
        y=ds.values,
        n=ds.N,
        include_intercept=model.include_intercept,
        lambda1=0.2,
    )
    sampler = SamplerConfig(draws=1500, burn_in=300, thin=1)

    rng = np.random.default_rng(321)
    fit_res = fit(ds, model, prior, sampler, rng=rng)
    fc = forecast(fit_res, horizons=[1, 4, 8], draws=500, rng=rng)

    print("fit complete (ELB)")
    print("elb bound:", elb_bound)
    print("min observed r:", float(np.min(ds.values[:, 0])))
    print("min forecast r (observed):", float(np.min(fc.draws[:, :, 0])))


if __name__ == "__main__":
    main()
