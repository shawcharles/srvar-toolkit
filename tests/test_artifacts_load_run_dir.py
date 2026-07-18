import numpy as np
import pandas as pd

from srvar.artifacts import load_run_dir, save_fit_npz
from srvar.data.dataset import Dataset
from srvar.results import FitResult, PosteriorNIW
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import VolatilitySpec


def test_load_run_dir_reconstructs_specs_and_loads_draws(tmp_path) -> None:
    out = tmp_path / "out"
    out.mkdir(parents=True)

    config_text = """\
model:
  p: 2
  include_intercept: true
  volatility:
    enabled: true
    covariance: factor
    dynamics: rw
    k_factors: 1
    loading_prior_var: 1.0
    store_factor_draws: true
prior:
  family: niw
  method: default
sampler:
  draws: 10
  burn_in: 0
  thin: 1
  seed: 0
"""
    (out / "config.yml").write_text(config_text, encoding="utf-8")

    rng = np.random.default_rng(0)
    t = 12
    n = 2
    p = 2
    t_eff = t - p
    time = pd.date_range("2000-01-01", periods=t, freq="MS")
    ds = Dataset.from_arrays(
        values=rng.standard_normal((t, n)),
        variables=["y1", "y2"],
        time_index=time,
    )

    latent_values = ds.values + 0.01
    latent_ds = Dataset.from_arrays(values=latent_values, variables=ds.variables, time_index=time)

    k = 1 + n * p
    posterior = PosteriorNIW(
        mn=rng.standard_normal((k, n)),
        vn=np.eye(k, dtype=float),
        sn=np.eye(n, dtype=float),
        nun=float(n + 2),
    )

    fit = FitResult(
        dataset=ds,
        model=ModelSpec(
            p=p, include_intercept=True, volatility=VolatilitySpec(covariance="factor")
        ),
        prior=PriorSpec.niw_default(k=k, n=n),
        sampler=SamplerConfig(draws=1, burn_in=0, thin=1),
        posterior=posterior,
        latent_dataset=latent_ds,
        beta_draws=rng.standard_normal((3, k, n)),
        h_draws=rng.standard_normal((3, t_eff, n)),
        lambda_draws=rng.standard_normal((3, n, 1)),
        h_factor_draws=rng.standard_normal((3, t_eff, 1)),
    )

    save_fit_npz(out / "fit_result.npz", fit)

    loaded = load_run_dir(out)
    assert loaded.dataset.variables == ["y1", "y2"]
    np.testing.assert_allclose(loaded.dataset.values, ds.values)

    assert loaded.model.p == 2
    assert loaded.model.volatility is not None
    assert loaded.model.volatility.covariance == "factor"
    assert loaded.model.volatility.k_factors == 1

    assert loaded.prior.family.lower() == "niw"
    assert loaded.sampler.draws == 10
    assert loaded.sampler.burn_in == 0
    assert loaded.sampler.thin == 1

    assert loaded.posterior is not None
    np.testing.assert_allclose(loaded.posterior.mn, posterior.mn)
    np.testing.assert_allclose(loaded.posterior.vn, posterior.vn)
    np.testing.assert_allclose(loaded.posterior.sn, posterior.sn)
    assert float(loaded.posterior.nun) == float(posterior.nun)

    assert loaded.latent_dataset is not None
    np.testing.assert_allclose(loaded.latent_dataset.values, latent_values)

    assert loaded.lambda_draws is not None
    assert loaded.h_factor_draws is not None
