import numpy as np

from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import VolatilitySpec


def _toy_dataset(*, t: int = 70, n: int = 2, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((t, n))
    return Dataset.from_arrays(values=y, variables=[f"y{i+1}" for i in range(n)])


def test_dl_runs_no_elb() -> None:
    ds = _toy_dataset(t=80, n=2, seed=1)
    model = ModelSpec(p=1, include_intercept=True)
    k = 1 + ds.N * model.p

    prior = PriorSpec.from_dl(k=k, n=ds.N, include_intercept=True)
    sampler = SamplerConfig(draws=120, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(999))

    assert fit_res.beta_draws is not None
    assert fit_res.sigma_draws is not None
    assert fit_res.beta_draws.shape[1:] == (k, ds.N)
    assert fit_res.sigma_draws.shape[1:] == (ds.N, ds.N)
    assert np.all(np.isfinite(fit_res.beta_draws))
    assert np.all(np.isfinite(fit_res.sigma_draws))


def test_dl_sv_runs() -> None:
    ds = _toy_dataset(t=70, n=2, seed=2)
    model = ModelSpec(p=1, include_intercept=True, volatility=VolatilitySpec())
    k = 1 + ds.N * model.p

    prior = PriorSpec.from_dl(k=k, n=ds.N, include_intercept=True)
    sampler = SamplerConfig(draws=100, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(111))

    assert fit_res.posterior is None
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert np.all(np.isfinite(fit_res.beta_draws))
    assert np.all(np.isfinite(fit_res.h_draws))


def test_gig_mean_matches_theory() -> None:
    from scipy.special import kv

    from srvar.rng import gig_rvs

    rng = np.random.default_rng(42)
    p, a, b = 0.5, 2.0, 3.0

    omega = float(np.sqrt(a * b))
    expected_mean = float(np.sqrt(b / a) * kv(p + 1, omega) / kv(p, omega))

    draws = 4000
    samples = np.asarray([gig_rvs(p=p, a=a, b=b, rng=rng) for _ in range(draws)], dtype=float)
    empirical_mean = float(np.mean(samples))

    assert np.isfinite(empirical_mean)
    assert abs(empirical_mean - expected_mean) / abs(expected_mean) < 0.15


def test_dl_elb_runs() -> None:
    rng = np.random.default_rng(3)
    t, n = 80, 2
    y = rng.standard_normal((t, n))
    elb_bound = -0.05
    y[:, 0] = np.minimum(y[:, 0], elb_bound)
    ds = Dataset.from_arrays(values=y, variables=["r", "y"])

    model = ModelSpec(p=1, include_intercept=True, elb=ElbSpec(bound=elb_bound, applies_to=["r"]))
    k = 1 + ds.N * model.p
    prior = PriorSpec.from_dl(k=k, n=ds.N, include_intercept=True)
    sampler = SamplerConfig(draws=100, burn_in=20, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(777))

    assert fit_res.beta_draws is not None
    assert fit_res.latent_draws is not None
    assert np.all(np.isfinite(fit_res.beta_draws))
