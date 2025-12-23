import types

import pandas as pd
import pytest


def test_plan_fetch_fred_no_network(tmp_path) -> None:
    from srvar.data.fetch_fred import plan_fetch_fred

    cfg = {
        "fred": {
            "series": {"CPI": {"id": "CPIAUCSL", "tcode": 5}},
            "api_key_env": "FRED_API_KEY",
            "start": "2000-01-01",
            "end": "2000-12-31",
        },
        "processing": {
            "frequency": "MS",
            "aggregation": "last",
            "upsample": "ffill",
            "transform_order": "resample_first",
            "dropna": True,
        },
        "output": {"csv_path": str(tmp_path / "fred.csv"), "date_column": "date"},
    }

    plan = plan_fetch_fred(cfg)
    assert plan["fred"]["series"]["CPI"]["id"] == "CPIAUCSL"
    assert plan["fred"]["series"]["CPI"]["tcode"] == 5
    assert plan["output"]["csv_path"].endswith("fred.csv")


def test_validate_fred_series_ids_uses_series_info(monkeypatch) -> None:
    from srvar.data import fetch_fred

    class FakeFred:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_series_info(self, series_id):
            if series_id == "BAD":
                raise Exception("not found")
            return {"id": series_id}

    monkeypatch.setattr(fetch_fred, "_require_fredapi", lambda: FakeFred)
    monkeypatch.setattr(fetch_fred, "_api_key", lambda api_key=None, api_key_env="FRED_API_KEY": "k")

    cfg_ok = {"fred": {"series": {"A": "GOOD"}}}
    fetch_fred.validate_fred_series_ids(cfg_ok)

    cfg_bad = {"fred": {"series": {"A": "BAD"}}}
    with pytest.raises(ValueError):
        fetch_fred.validate_fred_series_ids(cfg_bad)


def test_fetch_fred_dataframe_mocked(monkeypatch) -> None:
    from srvar.data import fetch_fred

    dates = pd.date_range("2020-01-01", periods=6, freq="MS")

    class FakeFred:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_series(self, series_id, observation_start=None, observation_end=None):
            return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates)

        def get_series_info(self, series_id):
            return {"id": series_id}

    monkeypatch.setattr(fetch_fred, "_require_fredapi", lambda: FakeFred)
    monkeypatch.setattr(fetch_fred, "_api_key", lambda api_key=None, api_key_env="FRED_API_KEY": "k")

    cfg = {
        "fred": {"series": {"X": {"id": "X", "tcode": 2}}},
        "processing": {"frequency": "MS", "aggregation": "last", "upsample": "ffill", "dropna": True},
    }

    df, meta = fetch_fred.fetch_fred_dataframe(cfg)
    assert list(df.columns) == ["X"]
    assert meta["source"] == "fred"
    assert meta["series"]["X"]["tcode"] == 2
    assert df.shape[0] == 5


def test_cli_dry_run_prints_plan(monkeypatch, capsys, tmp_path) -> None:
    from srvar import cli

    cfg_path = tmp_path / "fetch.yml"
    cfg_path.write_text(
        """
fred:
  series:
    CPI: CPIAUCSL
output:
  csv_path: "out.csv"
""".strip(),
        encoding="utf-8",
    )

    # Avoid requiring PyYAML extras in tests; load_config uses it.
    # In this repo, tests already import runner which loads PyYAML via extras,
    # so keep this minimal: monkeypatch load_config to return dict.
    monkeypatch.setattr(cli, "load_config", lambda p: {"fred": {"series": {"CPI": "CPIAUCSL"}}, "output": {"csv_path": "out.csv"}})

    rc = cli.main(["fetch-fred", str(cfg_path), "--dry-run", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CPIAUCSL" in out
