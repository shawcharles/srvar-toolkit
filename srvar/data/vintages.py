from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from .dataset import Dataset


_QUARTER_RE = re.compile(r"^\s*(\d{4})\s*[ -]?Q([1-4])\s*$", flags=re.IGNORECASE)


def _normalize_quarter_label(label: object) -> str:
    s = "" if label is None else str(label)
    s = s.replace("\u00a0", " ").strip()
    m = _QUARTER_RE.match(s)
    if m is None:
        raise ValueError(f"invalid quarter label: {label!r}")
    year = int(m.group(1))
    q = int(m.group(2))
    return f"{year} Q{q}"


def _quarter_label_to_period(label: object) -> pd.Period:
    s = _normalize_quarter_label(label)
    year_s, q_s = s.split()
    return pd.Period(f"{year_s}Q{q_s[1:]}", freq="Q")


def _parse_observation_index(observation: pd.Series) -> pd.PeriodIndex:
    obs = observation.astype(str).map(_normalize_quarter_label)
    years = obs.str.slice(0, 4)
    qs = obs.str.slice(-1)
    return pd.PeriodIndex.from_fields(year=years.astype(int), quarter=qs.astype(int), freq="Q")


def load_vintage_sheet(*, file_path: str | Path, sheet_name: str) -> tuple[pd.Period, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Workbook contains no default style.*",
            category=UserWarning,
        )
        xls = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    cols = list(xls.columns)
    obs_col = None
    for c in cols:
        if str(c).strip().lower() == "observation":
            obs_col = c
            break
    if obs_col is None:
        raise ValueError(f"sheet {sheet_name!r} has no 'observation' column")

    idx = _parse_observation_index(xls[obs_col])
    df = xls.drop(columns=[obs_col]).copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = idx

    last_vintage = _quarter_label_to_period(idx[-1])
    return last_vintage, df


def load_vintages_from_workbook(*, file_path: str | Path) -> dict[pd.Period, pd.DataFrame]:
    fp = Path(file_path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Workbook contains no default style.*",
            category=UserWarning,
        )
        sheets = pd.ExcelFile(fp, engine="openpyxl").sheet_names
    out: dict[pd.Period, pd.DataFrame] = {}
    for s in sheets:
        vintage, df = load_vintage_sheet(file_path=fp, sheet_name=s)
        if vintage in out:
            raise ValueError(f"duplicate vintage {vintage} in workbook {fp}")
        out[vintage] = df
    return out


def load_vintages_from_dir(*, data_dir: str | Path) -> dict[pd.Period, pd.DataFrame]:
    d = Path(data_dir)
    files = sorted(d.glob("vintages_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no vintages_*.xlsx found in {d}")

    out: dict[pd.Period, pd.DataFrame] = {}
    for fp in files:
        v = load_vintages_from_workbook(file_path=fp)
        overlap = set(out).intersection(v)
        if overlap:
            raise ValueError(f"duplicate vintage keys across files: {sorted(overlap)}")
        out.update(v)

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def dataset_from_vintage(
    *,
    vintage_df: pd.DataFrame,
    variables: list[str],
    vintage: pd.Period | None = None,
) -> Dataset:
    missing = [v for v in variables if v not in vintage_df.columns]
    if missing:
        raise ValueError(f"vintage is missing variables: {missing}")

    df = vintage_df.loc[:, variables]
    if vintage is not None:
        df = df.loc[:vintage]

    return Dataset.from_arrays(values=df.to_numpy(dtype=float), variables=variables, time_index=df.index)


def slice_to_common_history(vintages: dict[pd.Period, pd.DataFrame], *, variables: list[str]) -> dict[pd.Period, Dataset]:
    out: dict[pd.Period, Dataset] = {}
    for v, df in vintages.items():
        out[v] = dataset_from_vintage(vintage_df=df, variables=variables, vintage=v)
    return out
