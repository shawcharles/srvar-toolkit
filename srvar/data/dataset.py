from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Dataset:
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
        return int(self.values.shape[0])

    @property
    def N(self) -> int:
        return int(self.values.shape[1])
