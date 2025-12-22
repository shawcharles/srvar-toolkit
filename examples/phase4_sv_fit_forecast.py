import numpy as np

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    rng = np.random.default_rng(123)

    t, n = 120, 2
    y = rng.standard_normal((t, n))
    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=1, include_intercept=True, volatility=VolatilitySpec())
    prior = PriorSpec.niw_default(k=1 + n * model.p, n=n)
    sampler = SamplerConfig(draws=300, burn_in=50, thin=1)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))
    fc = forecast(fit_res, horizons=[1, 4], draws=200, rng=np.random.default_rng(2024))

    print("stored beta draws:", None if fit_res.beta_draws is None else fit_res.beta_draws.shape)
    print("stored h draws:", None if fit_res.h_draws is None else fit_res.h_draws.shape)
    print("forecast mean shape:", fc.mean.shape)


if __name__ == "__main__":
    main()
