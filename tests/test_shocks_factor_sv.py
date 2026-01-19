import numpy as np

from srvar.shocks import _quadratic_forms_factor_sv, update_precision_scales_factor_sv
from srvar.spec import ShockSpec


def test_quadratic_forms_factor_sv_matches_dense_inverse() -> None:
    rng = np.random.default_rng(0)
    t, n, k = 5, 4, 2

    errors = rng.standard_normal((t, n))
    loadings = rng.standard_normal((n, k))
    h_eta = rng.standard_normal((t, n))
    h_f = rng.standard_normal((t, k))

    q = _quadratic_forms_factor_sv(
        errors=errors, loadings=loadings, h_eta=h_eta, h_f=h_f, jitter=1e-12
    )

    q_dense = np.empty(t, dtype=float)
    for i in range(t):
        d_eta = np.diag(np.exp(h_eta[i, :]))
        d_f = np.diag(np.exp(h_f[i, :]))
        sigma = loadings @ d_f @ loadings.T + d_eta
        sol = np.linalg.solve(sigma, errors[i, :])
        q_dense[i] = float(errors[i, :] @ sol)

    assert np.allclose(q, q_dense, rtol=1e-8, atol=1e-10)


def test_update_precision_scales_factor_sv_returns_positive_values() -> None:
    rng = np.random.default_rng(1)
    t, n, k = 7, 3, 2

    errors = rng.standard_normal((t, n))
    loadings = rng.standard_normal((n, k))
    h_eta = rng.standard_normal((t, n))
    h_f = rng.standard_normal((t, k))

    lam = update_precision_scales_factor_sv(
        errors=errors,
        loadings=loadings,
        h_eta=h_eta,
        h_f=h_f,
        spec=ShockSpec(family="student_t", df=7.0),
        rng=rng,
    )
    assert lam.shape == (t,)
    assert np.all(np.isfinite(lam))
    assert np.all(lam > 0)

