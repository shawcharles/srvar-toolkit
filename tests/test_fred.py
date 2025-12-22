import types

import numpy as np
import pandas as pd


def test_fred_get_series_uses_cache(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "observations": [
                    {"date": "2020-01-01", "value": "1.0"},
                    {"date": "2020-02-01", "value": "2.0"},
                    {"date": "2020-03-01", "value": "."},
                ]
            }

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return FakeResponse()

    fake_requests = types.SimpleNamespace(get=fake_get)
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    from srvar.data import fred

    s1 = fred.get_series("X", api_key="k", cache_dir=tmp_path, use_cache=True)
    assert isinstance(s1, pd.Series)
    assert list(s1.index.astype(str)) == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert np.allclose(s1.iloc[:2].to_numpy(), np.array([1.0, 2.0]))
    assert np.isnan(s1.iloc[2])
    assert calls["n"] == 1

    # Second call should read from cache and not hit HTTP.
    def fake_get_fail(*args, **kwargs):
        raise AssertionError("HTTP called despite cache")

    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(get=fake_get_fail))

    s2 = fred.get_series("X", api_key="k", cache_dir=tmp_path, use_cache=True)
    assert calls["n"] == 1
    assert s2.equals(s1)


def test_fred_vintage_params_are_sent(tmp_path, monkeypatch) -> None:
    captured = {"params": None}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"observations": [{"date": "2020-01-01", "value": "1.0"}]}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = dict(params) if params is not None else None
        return FakeResponse()

    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(get=fake_get))

    from srvar.data import fred

    _ = fred.get_vintage_series(
        "X",
        realtime_start="2019-09-30",
        realtime_end="2019-09-30",
        api_key="k",
        cache_dir=tmp_path,
        use_cache=False,
    )

    assert captured["params"] is not None
    assert captured["params"].get("realtime_start") == "2019-09-30"
    assert captured["params"].get("realtime_end") == "2019-09-30"
