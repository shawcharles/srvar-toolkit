from pathlib import Path

import numpy as np
import pandas as pd

from srvar import Dataset, VolatilitySpec
from srvar.analysis import fevd_cholesky, historical_decomposition_cholesky, irf_cholesky
from srvar.api import fit
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.xarray import fevd_to_xarray, historical_decomposition_to_xarray, irf_to_xarray


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
            store_factor_draws=False,
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

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(123))

    irf = irf_cholesky(
        fit_res,
        horizons=12,
        draws=200,
        rng=np.random.default_rng(456),
    )
    fevd = fevd_cholesky(
        fit_res,
        horizons=12,
        draws=200,
        rng=np.random.default_rng(456),
    )
    hd = historical_decomposition_cholesky(
        fit_res,
        draws=200,
        rng=np.random.default_rng(456),
    )

    print("IRF draws:", irf.draws.shape)
    print("FEVD draws:", fevd.draws.shape)
    print("HD contributions:", hd.contributions_draws.shape)

    try:
        irf_ds = irf_to_xarray(irf)
        fevd_ds = fevd_to_xarray(fevd)
        hd_ds = historical_decomposition_to_xarray(hd)
    except ImportError:
        print("xarray not installed; skipping labeled output demo")
        return

    print("IRF xarray vars:", sorted(irf_ds.data_vars))
    print("FEVD xarray vars:", sorted(fevd_ds.data_vars))
    print("HD xarray vars:", sorted(hd_ds.data_vars))


if __name__ == "__main__":
    main()
