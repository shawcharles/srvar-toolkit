from __future__ import annotations

import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .transformations import tcode_1d


def _require_fredapi() -> Any:
    try:
        from fredapi import Fred  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("fredapi is required; install with 'srvar-toolkit[fred]'") from e
    return Fred


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def plan_fetch_fred(
    cfg: dict[str, Any],
    *,
    out_csv: str | Path | None = None,
) -> dict[str, Any]:
    fred_cfg = _as_mapping(cfg.get("fred"), key="fred")
    series_map = _normalize_series_map(fred_cfg.get("series"))

    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    output_cfg = _as_mapping(output_cfg, key="output")

    date_column = output_cfg.get("date_column", "date")
    date_column = _as_str(date_column, key="output.date_column")

    if out_csv is None:
        out_csv = output_cfg.get("csv_path")
    out_csv_p = Path(_as_str(out_csv, key="output.csv_path"))

    processing_cfg = cfg.get("processing", {})
    if processing_cfg is None:
        processing_cfg = {}
    processing_cfg = _as_mapping(processing_cfg, key="processing")

    freq = processing_cfg.get("frequency")
    if freq is not None:
        freq = _as_str(freq, key="processing.frequency")
    aggregation = processing_cfg.get("aggregation", "last")
    aggregation = _as_str(aggregation, key="processing.aggregation")
    upsample = processing_cfg.get("upsample", "ffill")
    upsample = _as_str(upsample, key="processing.upsample")
    transform_order = processing_cfg.get("transform_order", "resample_first")
    transform_order = _as_str(transform_order, key="processing.transform_order")

    transforms_cfg = processing_cfg.get("transforms", {})
    if transforms_cfg is None:
        transforms_cfg = {}
    transforms_cfg = _as_mapping(transforms_cfg, key="processing.transforms")

    tcodes: dict[str, float] = {}
    for name, spec in series_map.items():
        if "tcode" in spec:
            tcodes[name] = _as_number(spec["tcode"], key=f"fred.series.{name}.tcode")
        elif name in transforms_cfg:
            tcodes[name] = _as_number(transforms_cfg[name], key=f"processing.transforms.{name}")

    start = fred_cfg.get("start")
    end = fred_cfg.get("end")
    if start is not None:
        start = _as_str(start, key="fred.start")
    if end is not None:
        end = _as_str(end, key="fred.end")

    api_key_env = fred_cfg.get("api_key_env", "FRED_API_KEY")
    api_key_env = _as_str(api_key_env, key="fred.api_key_env")

    return {
        "fred": {
            "start": start,
            "end": end,
            "api_key_env": api_key_env,
            "series": {name: {"id": str(spec.get("id")), "tcode": tcodes.get(name)} for name, spec in series_map.items()},
        },
        "processing": {
            "frequency": freq,
            "aggregation": aggregation,
            "upsample": upsample,
            "transform_order": transform_order,
            "dropna": bool(processing_cfg.get("dropna", True)),
        },
        "output": {
            "csv_path": str(out_csv_p),
            "date_column": date_column,
            "meta_path": str(out_csv_p.with_name(f"{out_csv_p.stem}_meta.json")),
        },
    }


def validate_fred_series_ids(cfg: dict[str, Any]) -> None:
    fred_cfg = _as_mapping(cfg.get("fred"), key="fred")
    series_map = _normalize_series_map(fred_cfg.get("series"))

    api_key_env = fred_cfg.get("api_key_env", "FRED_API_KEY")
    api_key_env = _as_str(api_key_env, key="fred.api_key_env")

    api_key = fred_cfg.get("api_key")
    if api_key is not None:
        api_key = _as_str(api_key, key="fred.api_key")

    key = _api_key(api_key=api_key, api_key_env=api_key_env)
    Fred = _require_fredapi()
    fred = Fred(api_key=key)

    bad: list[str] = []
    for name, spec in series_map.items():
        series_id = _as_str(spec.get("id"), key=f"fred.series.{name}.id")
        try:
            _ = fred.get_series_info(series_id)
        except Exception:
            bad.append(series_id)

    if bad:
        bad_s = ", ".join(bad)
        raise ValueError(f"FRED series validation failed (not found or not accessible): {bad_s}")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        return


def _as_mapping(x: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise ValueError(f"{key} must be a mapping")
    return x


def _as_str(x: Any, *, key: str) -> str:
    if not isinstance(x, str) or not x:
        raise ValueError(f"{key} must be a non-empty string")
    return x


def _as_bool(x: Any, *, key: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"{key} must be a boolean")
    return x


def _as_number(x: Any, *, key: str) -> float:
    if not isinstance(x, (int, float, np.integer, np.floating)) or isinstance(x, bool):
        raise ValueError(f"{key} must be a number")
    return float(x)


def _api_key(*, api_key: str | None, api_key_env: str) -> str:
    if api_key is not None:
        return str(api_key)
    _load_dotenv_if_available()
    k = os.environ.get(api_key_env)
    if not k:
        raise ValueError(f"{api_key_env} is not set (set it in env or .env, or pass api_key=...)")
    return str(k)


def _normalize_series_map(series_cfg: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(series_cfg, dict) or not series_cfg:
        raise ValueError("fred.series must be a non-empty mapping")

    out: dict[str, dict[str, Any]] = {}
    for name, v in series_cfg.items():
        if not isinstance(name, str) or not name:
            raise ValueError("fred.series keys must be non-empty strings")

        if isinstance(v, str):
            out[name] = {"id": v}
            continue

        if isinstance(v, dict):
            if "id" not in v:
                raise ValueError(f"fred.series.{name} must include 'id'")
            out[name] = dict(v)
            continue

        raise ValueError(f"fred.series.{name} must be a string or mapping")

    return out


def _resample_frame(
    df: pd.DataFrame,
    *,
    frequency: str,
    aggregation: str,
    upsample: str,
) -> pd.DataFrame:
    agg = aggregation.lower()
    up = upsample.lower()

    parts: list[pd.Series] = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            parts.append(df[col])
            continue

        inferred = pd.infer_freq(s.index)
        s2: pd.Series

        if inferred is None:
            s2 = s
        else:
            try:
                a = pd.tseries.frequencies.to_offset(inferred)
                b = pd.tseries.frequencies.to_offset(frequency)
                inferred_is_finer = a.nanos < b.nanos
            except Exception:
                inferred_is_finer = False

            if inferred_is_finer:
                if agg == "mean":
                    s2 = s.resample(frequency).mean()
                elif agg == "sum":
                    s2 = s.resample(frequency).sum()
                else:
                    s2 = s.resample(frequency).last()
            else:
                if up == "interpolate":
                    s2 = s.resample(frequency).interpolate("time")
                else:
                    s2 = s.resample(frequency).ffill()

        s2 = s2.rename(col)
        parts.append(s2)

    out = pd.concat(parts, axis=1, join="outer").sort_index()
    return out


def fetch_fred_dataframe(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    fred_cfg = _as_mapping(cfg.get("fred"), key="fred")
    series_map = _normalize_series_map(fred_cfg.get("series"))

    start = fred_cfg.get("start")
    end = fred_cfg.get("end")
    if start is not None:
        start = _as_str(start, key="fred.start")
    if end is not None:
        end = _as_str(end, key="fred.end")

    api_key_env = fred_cfg.get("api_key_env", "FRED_API_KEY")
    api_key_env = _as_str(api_key_env, key="fred.api_key_env")

    api_key = fred_cfg.get("api_key")
    if api_key is not None:
        api_key = _as_str(api_key, key="fred.api_key")

    key = _api_key(api_key=api_key, api_key_env=api_key_env)

    Fred = _require_fredapi()
    fred = Fred(api_key=key)

    processing_cfg = cfg.get("processing", {})
    if processing_cfg is None:
        processing_cfg = {}
    processing_cfg = _as_mapping(processing_cfg, key="processing")

    freq = processing_cfg.get("frequency")
    if freq is not None:
        freq = _as_str(freq, key="processing.frequency")
    aggregation = processing_cfg.get("aggregation", "last")
    aggregation = _as_str(aggregation, key="processing.aggregation")
    upsample = processing_cfg.get("upsample", "ffill")
    upsample = _as_str(upsample, key="processing.upsample")
    transform_order = processing_cfg.get("transform_order", "resample_first")
    transform_order = _as_str(transform_order, key="processing.transform_order")
    if transform_order not in ("resample_first", "transform_first"):
        raise ValueError("processing.transform_order must be 'resample_first' or 'transform_first'")

    transforms_cfg = processing_cfg.get("transforms", {})
    if transforms_cfg is None:
        transforms_cfg = {}
    transforms_cfg = _as_mapping(transforms_cfg, key="processing.transforms")

    tcodes: dict[str, float] = {}
    for name, spec in series_map.items():
        if "tcode" in spec:
            tcodes[name] = _as_number(spec["tcode"], key=f"fred.series.{name}.tcode")
        elif name in transforms_cfg:
            tcodes[name] = _as_number(transforms_cfg[name], key=f"processing.transforms.{name}")

    raw_parts: list[pd.Series] = []
    for name, spec in series_map.items():
        series_id = _as_str(spec.get("id"), key=f"fred.series.{name}.id")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
        except Exception as e:
            raise ValueError(f"Failed to fetch FRED series '{series_id}' (as '{name}'): {e}") from e
        if not isinstance(s, pd.Series):
            raise ValueError(f"fredapi returned unexpected type for {series_id}")
        s = s.rename(name)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()

        if transform_order == "transform_first" and name in tcodes:
            tc = tcodes[name]
            s = pd.Series(
                tcode_1d(s.to_numpy(dtype=float), tc, var_name=series_id),
                index=s.index,
                name=name,
                dtype=float,
            )

        raw_parts.append(s)

    df = pd.concat(raw_parts, axis=1, join="outer").sort_index()

    if freq is not None:
        df = _resample_frame(df, frequency=freq, aggregation=aggregation, upsample=upsample)

    if transform_order == "resample_first" and tcodes:
        out = df.copy()
        for col, tc in tcodes.items():
            if col not in out.columns:
                raise ValueError(f"transform column not found: {col}")

            series_id = series_map[col].get("id")
            var_name = str(series_id) if series_id is not None else col
            out[col] = tcode_1d(out[col].to_numpy(dtype=float), tc, var_name=var_name)
        df = out

    dropna = processing_cfg.get("dropna", True)
    dropna = _as_bool(dropna, key="processing.dropna")
    if dropna:
        df = df.dropna(axis=0, how="any")

    meta: dict[str, Any] = {
        "source": "fred",
        "fetched_at_unix": time.time(),
        "start": start,
        "end": end,
        "api_key_env": api_key_env,
        "series": {name: {"id": str(spec.get("id")), "tcode": tcodes.get(name)} for name, spec in series_map.items()},
        "processing": {
            "frequency": freq,
            "aggregation": aggregation,
            "upsample": upsample,
            "transform_order": transform_order,
            "dropna": dropna,
        },
        "transform_conventions": {
            "tcode_5": "100 * diff(log(x))",
            "tcode_6": "100 * diff(diff(log(x)))",
        },
        "versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "fredapi": _package_version("fredapi"),
            "python_dotenv": _package_version("python-dotenv"),
        },
    }

    return df, meta


def fetch_fred_to_csv(
    cfg: dict[str, Any],
    *,
    out_csv: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, pd.DataFrame]:
    output_cfg = cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    output_cfg = _as_mapping(output_cfg, key="output")

    date_column = output_cfg.get("date_column", "date")
    date_column = _as_str(date_column, key="output.date_column")

    if out_csv is None:
        out_csv = output_cfg.get("csv_path")
    out_csv = Path(_as_str(out_csv, key="output.csv_path"))

    if out_csv.exists() and not overwrite:
        raise ValueError(f"output file exists (use --overwrite): {out_csv}")

    df, meta = fetch_fred_dataframe(cfg)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    x = df.copy()
    x.index.name = date_column
    x = x.reset_index()
    x.to_csv(out_csv, index=False)

    meta_path = out_csv.with_name(f"{out_csv.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return out_csv, meta_path, df
