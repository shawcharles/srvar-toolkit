from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.xarray import fit_to_xarray


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / "data" / "example.csv"

    df = pd.read_csv(csv_path)
    dt = pd.to_datetime(df["date"], errors="raise")
    y = df[["r", "y"]].to_numpy(dtype=float, copy=True)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"], time_index=dt)

    model = ModelSpec(
        p=2,
        include_intercept=True,
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=1,
            loading_prior_var=1.0,
            # Storing factor draws can be memory-intensive at scale.
            store_factor_draws=True,
        ),
    )
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

    print("fit complete (factor SV)")
    print("stored lambda draws:", None if fit_res.lambda_draws is None else fit_res.lambda_draws.shape)
    print("stored h_factor draws:", None if fit_res.h_factor_draws is None else fit_res.h_factor_draws.shape)
    print("stored factor draws:", None if fit_res.factor_draws is None else fit_res.factor_draws.shape)
    print("forecast mean shape:", fc.mean.shape)

    try:
        xr_ds = fit_to_xarray(fit_res)
    except ImportError:
        print("xarray not installed; skipping labeled output demo")
        return

    print("xarray variables:", sorted(xr_ds.data_vars))


if __name__ == "__main__":
    main()
