from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Dataset:
    """A lightweight container for multivariate time series data.

    The library consistently represents a dataset as a matrix ``values`` with shape
    ``(T, N)``, where:

    - ``T`` is the number of time points (observations)
    - ``N`` is the number of variables (series)

    Parameters
    ----------
    time_index:
        Time index for the observations. Can be a :class:`pandas.Index` (e.g. a
        :class:`pandas.DatetimeIndex`) or anything coercible to one.
    variables:
        Variable names of length ``N``.
    values:
        Numeric array of shape ``(T, N)``.

    Notes
    -----
    The class is immutable (``frozen=True``) and performs validation in
    :meth:`~Dataset.__post_init__`.
    """
    time_index: pd.Index
    variables: list[str]
    values: np.ndarray

    @staticmethod
    def from_arrays(
        *,
        values: np.ndarray,
        variables: Sequence[str],
        time_index: Iterable[object] | pd.Index | None = None,
    ) -> "Dataset":
        """Construct a :class:`~srvar.data.dataset.Dataset` from array-like inputs.

        Parameters
        ----------
        values:
            A numeric array of shape ``(T, N)``.
        variables:
            Sequence of variable names of length ``N``.
        time_index:
            Optional time index. If omitted, a :class:`pandas.RangeIndex` with
            ``start=0`` is used.

        Returns
        -------
        Dataset
            Validated dataset instance.

        Raises
        ------
        ValueError
            If shapes are inconsistent (e.g. ``len(variables) != values.shape[1]``)
            or if ``values`` is not two-dimensional.
        """
        x = np.asarray(values, dtype=float)
        if x.ndim != 2:
            raise ValueError("values must be a 2D array of shape (T, N)")

        vars_list = list(variables)
        if len(vars_list) != x.shape[1]:
            raise ValueError("len(variables) must equal values.shape[1]")

        if time_index is None:
            idx = pd.RangeIndex(start=0, stop=x.shape[0], step=1)
        else:
            idx = time_index if isinstance(time_index, pd.Index) else pd.Index(list(time_index))
            if len(idx) != x.shape[0]:
                raise ValueError("len(time_index) must equal values.shape[0]")

        return Dataset(time_index=idx, variables=vars_list, values=x)

    def __post_init__(self) -> None:
        x = np.asarray(self.values, dtype=float)
        if x.ndim != 2:
            raise ValueError("values must be a 2D array of shape (T, N)")

        if len(self.variables) != x.shape[1]:
            raise ValueError("len(variables) must equal values.shape[1]")

        if len(self.time_index) != x.shape[0]:
            raise ValueError("len(time_index) must equal values.shape[0]")

        object.__setattr__(self, "values", x)
        if not isinstance(self.time_index, pd.Index):
            object.__setattr__(self, "time_index", pd.Index(self.time_index))

    @property
    def T(self) -> int:
        """Number of time points (rows) in the dataset."""
        return int(self.values.shape[0])

    @property
    def N(self) -> int:
        """Number of variables (columns) in the dataset."""
        return int(self.values.shape[1])
