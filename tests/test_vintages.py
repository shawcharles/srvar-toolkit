import numpy as np
import pandas as pd
import pytest

from srvar.data import vintages


def test_vintage_observation_labels_and_inclusive_cutoff() -> None:
    index = vintages._parse_observation_index(pd.Series(["2020 Q1", "2020Q2", "2020-Q3"]))
    assert index.tolist() == [
        pd.Period("2020Q1", freq="Q-DEC"),
        pd.Period("2020Q2", freq="Q-DEC"),
        pd.Period("2020Q3", freq="Q-DEC"),
    ]

    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=index)
    dataset = vintages.dataset_from_vintage(
        vintage_df=frame,
        variables=["y"],
        vintage=pd.Period("2020Q2", freq="Q"),
    )
    assert dataset.T == 2
    assert np.allclose(dataset.values[:, 0], [1.0, 2.0])

    with pytest.raises(ValueError, match="invalid quarter label"):
        vintages._parse_observation_index(pd.Series(["not-a-quarter"]))


def test_load_vintages_accepts_string_sheet_names(monkeypatch, tmp_path) -> None:
    class FakeExcelFile:
        sheet_names = ["2020Q1", "2020Q2"]

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def fake_load_vintage_sheet(*, file_path, sheet_name: str):
        period = pd.Period(sheet_name, freq="Q")
        return period, pd.DataFrame({"y": [1.0]}, index=pd.PeriodIndex([period], freq="Q"))

    monkeypatch.setattr(vintages.pd, "ExcelFile", FakeExcelFile)
    monkeypatch.setattr(vintages, "load_vintage_sheet", fake_load_vintage_sheet)

    loaded = vintages.load_vintages_from_workbook(file_path=tmp_path / "vintages.xlsx")
    assert list(loaded) == [pd.Period("2020Q1", freq="Q"), pd.Period("2020Q2", freq="Q")]
