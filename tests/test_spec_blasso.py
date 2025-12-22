import numpy as np

from srvar.spec import PriorSpec


def test_from_blasso_shapes_and_defaults() -> None:
    k, n = 5, 2
    prior = PriorSpec.from_blasso(k=k, n=n)
    assert prior.family.lower() == "blasso"
    assert prior.blasso is not None
    assert prior.niw.m0.shape == (k, n)
    assert prior.niw.s0.shape == (n, n)


def test_blasso_spec_rejects_nonpositive_hyperparams() -> None:
    k, n = 5, 2
    try:
        _ = PriorSpec.from_blasso(k=k, n=n, a0_global=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_gamma_rate_helper_rejects_bad_inputs() -> None:
    from srvar.rng import gamma_rate

    rng = np.random.default_rng(0)
    try:
        _ = gamma_rate(shape=-1.0, rate=1.0, rng=rng)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
