import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from srvar.elb import apply_elb_floor
from srvar.var import design_matrix
from srvar.var import lag_matrix


@st.composite
def _y_p_intercept(draw: st.DrawFn) -> tuple[np.ndarray, int, bool]:
    n = draw(st.integers(min_value=1, max_value=4))
    p = draw(st.integers(min_value=1, max_value=4))
    t = draw(st.integers(min_value=p + 1, max_value=p + 15))
    include_intercept = draw(st.booleans())
    y = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(t, n),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    return y, p, include_intercept


@settings(max_examples=75, deadline=None)
@given(_y_p_intercept())
def test_design_matrix_shapes_and_targets(arg: tuple[np.ndarray, int, bool]) -> None:
    y, p, include_intercept = arg

    x, yt = design_matrix(y, p, include_intercept=include_intercept)

    t, n = y.shape
    assert yt.shape == (t - p, n)
    assert np.allclose(yt, y[p:, :])

    k_expected = (1 if include_intercept else 0) + n * p
    assert x.shape == (t - p, k_expected)

    if include_intercept:
        assert np.allclose(x[:, 0], 1.0)


@settings(max_examples=75, deadline=None)
@given(_y_p_intercept())
def test_lag_matrix_matches_design_matrix_no_intercept(arg: tuple[np.ndarray, int, bool]) -> None:
    y, p, _include_intercept = arg

    xl = lag_matrix(y, p)
    x, _yt = design_matrix(y, p, include_intercept=False)

    assert xl.shape == x.shape
    assert np.allclose(xl, x)


@settings(max_examples=50, deadline=None)
@given(
    hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=8),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    ),
    st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=3),
)
def test_apply_elb_floor_respects_floor(values: np.ndarray, bound: float, j: int) -> None:
    if values.ndim < 2:
        return

    n = values.shape[-1]
    if n == 0:
        return

    idx = [min(j, n - 1)]

    out = apply_elb_floor(values, bound=bound, indices=idx)

    assert out.shape == values.shape
    assert np.all(out[..., idx] >= bound - 1e-12)


def test_lag_matrix_raises_when_t_le_p() -> None:
    y = np.zeros((3, 2), dtype=float)
    try:
        lag_matrix(y, 3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
