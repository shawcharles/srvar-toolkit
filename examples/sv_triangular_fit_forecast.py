import numpy as np

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    rng = np.random.default_rng(123)

    ds = Dataset.from_arrays(
        values=rng.standard_normal((120, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=2,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="triangular", q_prior_var=1.0),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=400, burn_in=100, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=rng)
    print("beta_draws:", None if fit_res.beta_draws is None else fit_res.beta_draws.shape)
    print("q_draws:", None if fit_res.q_draws is None else fit_res.q_draws.shape)

    fc = forecast(fit_res, horizons=[1, 3, 6], draws=200, rng=rng)
    print("forecast mean shape:", fc.mean.shape)


if __name__ == "__main__":
    main()
