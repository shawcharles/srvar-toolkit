import numpy as np

from srvar import Dataset
from srvar.runner import build_prior
from srvar.spec import ModelSpec


def test_runner_build_prior_parses_blasso() -> None:
    ds = Dataset.from_arrays(values=np.random.default_rng(0).standard_normal((40, 2)), variables=["y1", "y2"])
    model = ModelSpec(p=1, include_intercept=True)

    cfg = {
        "prior": {
            "family": "blasso",
            "blasso": {
                "mode": "adaptive",
                "tau_init": 1e3,
                "lambda_init": 1.5,
                "a0_c": 2.0,
                "b0_c": 3.0,
                "a0_L": 4.0,
                "b0_L": 5.0,
            },
        }
    }

    prior = build_prior(cfg, dataset=ds, model=model)
    assert prior.family.lower() == "blasso"
    assert prior.blasso is not None
    assert prior.blasso.mode == "adaptive"
    assert prior.blasso.tau_init == 1e3
    assert prior.blasso.lambda_init == 1.5
