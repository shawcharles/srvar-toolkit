import numpy as np
import pytest

from srvar.stats import giacomini_white_test, newey_west_covariance_matrix


def test_newey_west_covariance_matrix_matches_reference_formula() -> None:
    z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    out = newey_west_covariance_matrix(z, nlags=1)

    # Manual computation matching `functions/NeweyWest.m`:
    z0 = z - z.mean(axis=0, keepdims=True)
    n = z0.shape[0]
    omega = (z0.T @ z0) / float(n)

    zlag = np.zeros_like(z0)
    zlag[1:, :] = z0[:-1, :]
    gamma = (z0.T @ zlag + zlag.T @ z0) / float(n)
    omega = omega + 0.5 * gamma

    assert np.allclose(out, omega)


def test_gw_unconditional_tau1_identical_losses_p_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = giacomini_white_test(a, b, horizon=1, choice="unconditional")
    assert res.df == 1
    assert res.statistic == 0.0
    assert res.pvalue == 1.0
    assert res.significance_code == 0


def test_gw_conditional_tau1_has_two_instruments() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    res = giacomini_white_test(a, b, horizon=1, choice="conditional")
    assert res.df == 2
    assert 0.0 <= res.pvalue <= 1.0


def test_gw_tau_gt1_handles_zero_omega_and_nonzero_mean() -> None:
    # Constant loss differential => omega becomes 0 after demeaning; statistic should be inf.
    a = np.ones(10, dtype=float)
    b = np.zeros(10, dtype=float)
    res = giacomini_white_test(a, b, horizon=3, choice="unconditional")
    assert np.isposinf(res.statistic)
    assert res.pvalue == 0.0
    assert res.significance_code == 3


def test_gw_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="horizon"):
        giacomini_white_test(np.zeros(3), np.zeros(3), horizon=0)
    with pytest.raises(ValueError, match="alpha"):
        giacomini_white_test(np.zeros(3), np.zeros(3), horizon=1, alpha=1.0)
    with pytest.raises(ValueError, match="same shape"):
        giacomini_white_test(np.zeros(3), np.zeros(4), horizon=1)
