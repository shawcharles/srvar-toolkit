import numpy as np

from srvar import Dataset, VolatilitySpec
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def test_factor_sv_fit_forecast_runs_and_identification_holds() -> None:
    ds = Dataset.from_arrays(
        values=np.random.default_rng(123).standard_normal((60, 3)),
        variables=["y1", "y2", "y3"],
    )

    model = ModelSpec(
        p=1,
        include_intercept=True,
        volatility=VolatilitySpec(
            enabled=True,
            covariance="factor",
            dynamics="rw",
            k_factors=2,
            loading_prior_var=1.0,
            store_factor_draws=False,
        ),
    )
    prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
    sampler = SamplerConfig(draws=60, burn_in=10, thin=2)

    fit_res = fit(ds, model, prior, sampler, rng=np.random.default_rng(456))
    assert fit_res.beta_draws is not None
    assert fit_res.h_draws is not None
    assert fit_res.sigma_eta2_draws is not None
    assert fit_res.lambda_draws is not None
    assert fit_res.h_factor_draws is not None
    assert fit_res.sigma_eta2_factor_draws is not None

    d, t_eff, n = fit_res.h_draws.shape
    assert n == ds.N
    assert fit_res.lambda_draws.shape == (d, ds.N, 2)
    assert fit_res.h_factor_draws.shape == (d, t_eff, 2)
    assert fit_res.sigma_eta2_factor_draws.shape == (d, 2)

    # Identification: lower-triangular in first k rows + positive diagonal
    assert np.allclose(fit_res.lambda_draws[:, 0, 1], 0.0)
    assert np.all(fit_res.lambda_draws[:, 0, 0] > 0.0)
    assert np.all(fit_res.lambda_draws[:, 1, 1] > 0.0)

    # PSD check for implied covariance at the final in-sample time index
    for draw_idx in range(min(3, d)):
        lam = fit_res.lambda_draws[draw_idx]
        h_eta_last = fit_res.h_draws[draw_idx, -1, :]
        h_f_last = fit_res.h_factor_draws[draw_idx, -1, :]

        sigma = lam @ np.diag(np.exp(h_f_last)) @ lam.T + np.diag(np.exp(h_eta_last))
        sigma = 0.5 * (sigma + sigma.T)
        eig = np.linalg.eigvalsh(sigma)
        assert float(eig.min()) > -1e-8

    fc = forecast(fit_res, horizons=[1, 2], draws=20, rng=np.random.default_rng(789))
    assert fc.draws.shape == (20, 2, ds.N)
    assert np.all(np.isfinite(fc.draws))
