from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset


_FRED_BASE_URL = "https://api.stlouisfed.org/fred"


def _require_requests() -> Any:
    try:
        import requests  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("requests is required; install with 'srvar-toolkit[fred]'") from e
    return requests


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        return


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "srvar" / "fred"
    return Path.home() / ".cache" / "srvar" / "fred"


def _cache_key(*, endpoint: str, params: dict[str, Any]) -> str:
    payload = {"endpoint": endpoint, "params": params}
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _api_key(*, api_key: str | None) -> str:
    if api_key is not None:
        return str(api_key)
    _load_dotenv_if_available()
    k = os.environ.get("FRED_API_KEY")
    if not k:
        raise ValueError("FRED_API_KEY is not set (set it in env or .env, or pass api_key=...)")
    return str(k)


def _parse_observations(*, series_id: str, payload: dict[str, Any]) -> pd.Series:
    obs = payload.get("observations")
    if not isinstance(obs, list):
        raise ValueError("unexpected response: missing observations")

    dates: list[pd.Timestamp] = []
    vals: list[float] = []
    for row in obs:
        if not isinstance(row, dict):
            continue
        d = row.get("date")
        v = row.get("value")
        if not isinstance(d, str):
            continue
        dates.append(pd.to_datetime(d))
        if v in (None, "."):
            vals.append(np.nan)
        else:
            vals.append(float(v))

    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=series_id, dtype=float)
    s = s.sort_index()
    return s


def get_series(
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
    units: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> pd.Series:
    return get_vintage_series(
        series_id,
        realtime_start=None,
        realtime_end=None,
        start=start,
        end=end,
        api_key=api_key,
        frequency=frequency,
        aggregation_method=aggregation_method,
        units=units,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


def get_vintage_series(
    series_id: str,
    *,
    realtime_start: str | None,
    realtime_end: str | None,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
    units: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> pd.Series:
    key = _api_key(api_key=api_key)

    endpoint = f"{_FRED_BASE_URL}/series/observations"
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
    }
    if start is not None:
        params["observation_start"] = start
    if end is not None:
        params["observation_end"] = end
    if realtime_start is not None:
        params["realtime_start"] = realtime_start
    if realtime_end is not None:
        params["realtime_end"] = realtime_end
    if frequency is not None:
        params["frequency"] = frequency
    if aggregation_method is not None:
        params["aggregation_method"] = aggregation_method
    if units is not None:
        params["units"] = units

    cdir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cfile = cdir / f"{_cache_key(endpoint=endpoint, params=params)}.json"

    if use_cache:
        cached = _read_cache(cfile)
        if cached is not None:
            return _parse_observations(series_id=series_id, payload=cached)

    requests = _require_requests()
    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, dict):
        raise ValueError("unexpected response: not a JSON object")

    if use_cache:
        _write_cache(cfile, payload)

    return _parse_observations(series_id=series_id, payload=payload)


def get_many(
    series_map: dict[str, str],
    *,
    vintage: str | None = None,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
    units: str | None = None,
    join: str = "inner",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    parts: list[pd.Series] = []

    for name, sid in series_map.items():
        if vintage is None:
            s = get_series(
                sid,
                start=start,
                end=end,
                api_key=api_key,
                frequency=frequency,
                aggregation_method=aggregation_method,
                units=units,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        else:
            s = get_vintage_series(
                sid,
                realtime_start=vintage,
                realtime_end=vintage,
                start=start,
                end=end,
                api_key=api_key,
                frequency=frequency,
                aggregation_method=aggregation_method,
                units=units,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        s = s.rename(name)
        parts.append(s)

    df = pd.concat(parts, axis=1, join=join)
    df = df.sort_index()
    return df


def to_dataset(df: pd.DataFrame, *, dropna: bool = True) -> Dataset:
    x = df.copy()
    if dropna:
        x = x.dropna(axis=0, how="any")
    return Dataset.from_arrays(values=x.to_numpy(dtype=float), variables=list(x.columns), time_index=x.index)
