import numpy as np

from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def _toy_dataset(*, t: int = 60, n: int = 2, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((t, n))
    return Dataset.from_arrays(values=y, variables=[f"y{i+1}" for i in range(n)])


def test_blasso_global_runs_and_shrinks_vs_niw() -> None:
    ds = _toy_dataset(t=70, n=2, seed=1)
    model = ModelSpec(p=1, include_intercept=True)
    k = 1 + ds.N * model.p

    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    prior_niw = PriorSpec.niw_default(k=k, n=ds.N)
    fit_niw = fit(ds, model, prior_niw, sampler, rng=np.random.default_rng(999))
    assert fit_niw.beta_draws is not None

    prior_b = PriorSpec.from_blasso(k=k, n=ds.N, include_intercept=True, mode="global")
    fit_b = fit(ds, model, prior_b, sampler, rng=np.random.default_rng(999))
    assert fit_b.beta_draws is not None

    niw_mean = float(np.mean(np.abs(fit_niw.beta_draws)))
    b_mean = float(np.mean(np.abs(fit_b.beta_draws)))
    assert np.isfinite(niw_mean)
    assert np.isfinite(b_mean)
    assert b_mean <= niw_mean + 1e-6


def test_blasso_adaptive_runs() -> None:
    ds = _toy_dataset(t=70, n=2, seed=2)
    model = ModelSpec(p=1, include_intercept=True)
    k = 1 + ds.N * model.p

    prior_b = PriorSpec.from_blasso(k=k, n=ds.N, include_intercept=True, mode="adaptive")
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)
    fit_b = fit(ds, model, prior_b, sampler, rng=np.random.default_rng(111))

    assert fit_b.beta_draws is not None
    assert np.all(np.isfinite(fit_b.beta_draws))
