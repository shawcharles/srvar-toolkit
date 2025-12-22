import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    rng = np.random.default_rng(123)

    # toy dataset
    t, n = 80, 2
    y = rng.standard_normal((t, n))
    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=2, include_intercept=True)

    # K = 1 + N*p
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=2000, burn_in=500, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=rng)

    fc = forecast(fit_res, horizons=[1, 4], draws=200, rng=rng)
    print(fc.mean.shape)
    print({k: v.shape for k, v in fc.quantiles.items()})


if __name__ == "__main__":
    main()
