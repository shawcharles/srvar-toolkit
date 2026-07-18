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


def test_dataset_rejects_duplicate_timestamps_on_direct_construction() -> None:
    with pytest.raises(ValueError, match="time_index must be unique"):
        Dataset(
            time_index=["2000-01-01", "2000-01-01"],
            variables=["a"],
            values=np.array([[1.0], [2.0]]),
        )


def test_dataset_rejects_duplicate_datetime_timestamps_from_arrays() -> None:
    dates = pd.to_datetime(["2000-01-01", "2000-01-01"])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        Dataset.from_arrays(values=np.array([[1.0], [2.0]]), variables=["a"], time_index=dates)


def test_dataset_preserves_unique_unsorted_time_index_order() -> None:
    dates = pd.to_datetime(["2000-02-01", "2000-01-01"])
    ds = Dataset.from_arrays(values=np.array([[1.0], [2.0]]), variables=["a"], time_index=dates)
    assert ds.time_index.equals(dates)
