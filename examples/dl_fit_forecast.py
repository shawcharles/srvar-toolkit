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
    k = (1 if model.include_intercept else 0) + ds.N * model.p

    # Dirichlet–Laplace (DL) shrinkage prior
    # - abeta: Dirichlet concentration parameter
    # - dl_scaler: initialization scale for DL latent variables
    prior = PriorSpec.from_dl(k=k, n=ds.N, include_intercept=model.include_intercept, abeta=0.5, dl_scaler=0.1)

    sampler = SamplerConfig(draws=600, burn_in=200, thin=2)

    rng = np.random.default_rng(123)
    fit_res = fit(ds, model, prior, sampler, rng=rng)
    fc = forecast(fit_res, horizons=[1, 4, 8], draws=500, rng=rng)

    print("fit complete (DL prior)")
    print("dataset:", f"T={ds.T}", f"N={ds.N}", f"vars={ds.variables}")
    print("beta_draws shape:", None if fit_res.beta_draws is None else fit_res.beta_draws.shape)
    print("sigma_draws shape:", None if fit_res.sigma_draws is None else fit_res.sigma_draws.shape)
    print("forecast mean shape:", fc.mean.shape)
    print("forecast quantiles:", {q: a.shape for q, a in fc.quantiles.items()})


if __name__ == "__main__":
    main()
