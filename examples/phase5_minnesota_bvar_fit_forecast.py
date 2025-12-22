import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    rng = np.random.default_rng(123)

    t, n = 80, 2
    y = rng.standard_normal((t, n))
    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=2, include_intercept=True)

    prior = PriorSpec.niw_minnesota(
        p=model.p,
        y=ds.values,
        include_intercept=model.include_intercept,
        lambda1=0.1,
        lambda2=0.5,
        lambda3=1.0,
        lambda4=100.0,
        own_lag_mean=0.0,
    )
    sampler = SamplerConfig(draws=500, burn_in=100, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))
    fc = forecast(fit_res, horizons=[1, 4], draws=200, rng=np.random.default_rng(2024))

    print(fc.mean.shape)
    print({k: v.shape for k, v in fc.quantiles.items()})


if __name__ == "__main__":
    main()
