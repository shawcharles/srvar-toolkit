import numpy as np
import pandas as pd

from srvar import VolatilitySpec
from srvar.artifacts import load_fit_npz, save_fit_npz
from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_save_load_fit_npz_includes_factor_sv_draws(tmp_path) -> None:
    rng = np.random.default_rng(0)

    t = 8
    p = 2
    n = 3
    k = 2
    d = 4
    t_eff = t - p

    time = pd.date_range("2000-01-01", periods=t, freq="MS")
    ds = Dataset.from_arrays(
        values=rng.standard_normal((t, n)),
        variables=["y1", "y2", "y3"],
        time_index=time,
    )
    model = ModelSpec(
        p=p,
        include_intercept=True,
        volatility=VolatilitySpec(enabled=True, covariance="factor", dynamics="rw", k_factors=k),
    )
    prior = PriorSpec.niw_default(k=1 + n * p, n=n)
    sampler = SamplerConfig(draws=d, burn_in=0, thin=1)

    fit = FitResult(
        dataset=ds,
        model=model,
        prior=prior,
        sampler=sampler,
        posterior=None,
        beta_draws=rng.standard_normal((d, 1 + n * p, n)),
        h_draws=rng.standard_normal((d, t_eff, n)),
        lambda_draws=rng.standard_normal((d, n, k)),
        factor_draws=rng.standard_normal((d, t_eff, k)),
        h_factor_draws=rng.standard_normal((d, t_eff, k)),
        h0_factor_draws=rng.standard_normal((d, k)),
        sigma_eta2_factor_draws=np.abs(rng.standard_normal((d, k))),
    )

    path = tmp_path / "fit_result.npz"
    save_fit_npz(path, fit)
    loaded = load_fit_npz(path)

    assert loaded.lambda_draws is not None
    assert loaded.factor_draws is not None
    assert loaded.h_factor_draws is not None
    assert loaded.h0_factor_draws is not None
    assert loaded.sigma_eta2_factor_draws is not None

    np.testing.assert_allclose(loaded.dataset.values, ds.values)
    np.testing.assert_allclose(loaded.beta_draws, fit.beta_draws)
    np.testing.assert_allclose(loaded.h_draws, fit.h_draws)

    np.testing.assert_allclose(loaded.lambda_draws, fit.lambda_draws)
    np.testing.assert_allclose(loaded.factor_draws, fit.factor_draws)
    np.testing.assert_allclose(loaded.h_factor_draws, fit.h_factor_draws)
    np.testing.assert_allclose(loaded.h0_factor_draws, fit.h0_factor_draws)
    np.testing.assert_allclose(loaded.sigma_eta2_factor_draws, fit.sigma_eta2_factor_draws)
