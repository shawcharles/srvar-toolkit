import numpy as np

from srvar import Dataset
from srvar.api import fit
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_ssvs_fit_produces_gamma_draws() -> None:
    rng = np.random.default_rng(123)

    t, n = 80, 2
    y = rng.standard_normal((t, n))
    ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

    model = ModelSpec(p=1, include_intercept=True)
    k = 1 + n * model.p
    prior = PriorSpec.from_ssvs(k=k, n=n, include_intercept=True)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.gamma_draws is not None
    assert fit_res.gamma_draws.ndim == 2
    assert fit_res.gamma_draws.shape[1] == k

    # still produces coefficient/covariance draws
    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
