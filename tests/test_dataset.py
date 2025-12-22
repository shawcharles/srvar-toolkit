import numpy as np
import pandas as pd
import pytest

from srvar.data import Dataset


def test_dataset_from_arrays_shapes() -> None:
    values = np.arange(12, dtype=float).reshape(6, 2)
    ds = Dataset.from_arrays(values=values, variables=["a", "b"])
    assert ds.T == 6
    assert ds.N == 2
    assert list(ds.variables) == ["a", "b"]


def test_dataset_time_index_length_mismatch_raises() -> None:
    values = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError):
        Dataset.from_arrays(values=values, variables=["a", "b"], time_index=pd.RangeIndex(0, 5))
