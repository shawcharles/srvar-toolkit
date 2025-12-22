import numpy as np

from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def main() -> None:
    rng = np.random.default_rng(0)

    t, n = 120, 2
    y = rng.standard_normal((t, n))
    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=1, include_intercept=True)
    k = 1 + n * model.p

    sampler = SamplerConfig(draws=500, burn_in=100, thin=2)

    prior_global = PriorSpec.from_blasso(k=k, n=n, include_intercept=True, mode="global")
    res_g = fit(ds, model, prior_global, sampler, rng=np.random.default_rng(123))
    print("Global BLASSO mean |beta|:", float(np.mean(np.abs(res_g.beta_draws))))

    prior_adaptive = PriorSpec.from_blasso(k=k, n=n, include_intercept=True, mode="adaptive")
    res_a = fit(ds, model, prior_adaptive, sampler, rng=np.random.default_rng(123))
    print("Adaptive BLASSO mean |beta|:", float(np.mean(np.abs(res_a.beta_draws))))


if __name__ == "__main__":
    main()
