import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    rng = np.random.default_rng(123)

    t, n = 120, 2
    y = rng.standard_normal((t, n))

    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)

    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    model = ModelSpec(p=1, include_intercept=True, elb=ElbSpec(bound=elb_bound, applies_to=["r"]))
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=500, burn_in=100, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))
    fc = forecast(fit_res, horizons=[1, 4], draws=200, rng=np.random.default_rng(2024))

    print("forecast mean shape:", fc.mean.shape)
    print("min forecast r:", float(np.min(fc.draws[:, :, 0])))


if __name__ == "__main__":
    main()
