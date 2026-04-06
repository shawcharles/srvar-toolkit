import csv
import math

import numpy as np
import pandas as pd


def _read_metrics(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assert_metrics_equal(rows_a, rows_b) -> None:
    a = sorted(rows_a, key=lambda r: (r["variable"], int(r["horizon"])))
    b = sorted(rows_b, key=lambda r: (r["variable"], int(r["horizon"])))
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert ra.keys() == rb.keys()
        assert ra["variable"] == rb["variable"]
        assert int(ra["horizon"]) == int(rb["horizon"])
        for k in ra.keys():
            if k in {"variable", "horizon"}:
                continue
            va = float(ra[k])
            vb = float(rb[k])
            if math.isnan(va) and math.isnan(vb):
                continue
            assert np.isclose(va, vb)


def test_backtest_streaming_matches_in_memory(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 3.0, np.nan, 5.0]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    def fake_fit(dataset, model, prior, sampler, *, rng=None):
        return FitResult(dataset=dataset, model=model, prior=prior, sampler=sampler, posterior=None)

    def fake_forecast(
        fit,
        *,
        horizons,
        draws,
        quantile_levels,
        stationarity="allow",
        stationarity_tol=1e-10,
        stationarity_max_draws=None,
        rng=None,
    ):
        hmax = int(max(horizons))
        last = float(fit.dataset.values[-1, 0])
        sims = np.full((int(draws), hmax, int(fit.dataset.N)), last, dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    cfg_base = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": False,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 3,
            "step": 1,
            "horizons": [1, 2],
            "draws": 10,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-03-01",
            "origin_end": "2000-04-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": True, "intervals": [0.5, 0.9], "use_latent": False},
            "crps": {"enabled": True, "use_latent": False},
            "wis": {"enabled": True, "intervals": [0.5, 0.9], "use_latent": False},
            "pinball": {"enabled": True, "quantiles": [0.1, 0.5, 0.9], "use_latent": False},
            "log_score": {"enabled": True, "variance_floor": 1e-12, "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"save_plots": False, "save_forecasts": False},
    }

    cfg_holder = {"cfg": cfg_base}
    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg_holder["cfg"])

    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_stream = tmp_path / "out_stream"
    cfg_stream = dict(cfg_base)
    cfg_stream["output"] = dict(cfg_base["output"])
    cfg_stream["output"]["store_forecasts_in_memory"] = False
    cfg_holder["cfg"] = cfg_stream
    runner.backtest_from_config(str(config_path), out_dir=str(out_stream))

    out_mem = tmp_path / "out_mem"
    cfg_mem = dict(cfg_base)
    cfg_mem["output"] = dict(cfg_base["output"])
    cfg_mem["output"]["store_forecasts_in_memory"] = True
    cfg_holder["cfg"] = cfg_mem
    runner.backtest_from_config(str(config_path), out_dir=str(out_mem))

    rows_stream = _read_metrics(out_stream / "metrics.csv")
    rows_mem = _read_metrics(out_mem / "metrics.csv")
    _assert_metrics_equal(rows_stream, rows_mem)


def test_backtest_missing_realizations_are_excluded_from_coverage(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=5, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 2.0, np.nan]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    def fake_fit(dataset, model, prior, sampler, *, rng=None):
        return FitResult(dataset=dataset, model=model, prior=prior, sampler=sampler, posterior=None)

    def fake_forecast(
        fit,
        *,
        horizons,
        draws,
        quantile_levels,
        stationarity="allow",
        stationarity_tol=1e-10,
        stationarity_max_draws=None,
        rng=None,
    ):
        hmax = int(max(horizons))
        last = float(fit.dataset.values[-1, 0])
        sims = np.full((int(draws), hmax, int(fit.dataset.N)), last, dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    cfg_base = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": False,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 3,
            "step": 1,
            "horizons": [1],
            "draws": 10,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-03-01",
            "origin_end": "2000-04-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": True, "intervals": [0.5], "use_latent": False},
            "crps": {"enabled": True, "use_latent": False},
            "wis": {"enabled": False, "intervals": [], "use_latent": False},
            "pinball": {"enabled": False, "quantiles": [], "use_latent": False},
            "log_score": {"enabled": False, "variance_floor": 1e-12, "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"save_plots": False, "save_forecasts": False},
    }

    cfg_holder = {"cfg": cfg_base}
    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg_holder["cfg"])

    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_stream = tmp_path / "out_stream"
    cfg_stream = dict(cfg_base)
    cfg_stream["output"] = dict(cfg_base["output"])
    cfg_stream["output"]["store_forecasts_in_memory"] = False
    cfg_holder["cfg"] = cfg_stream
    runner.backtest_from_config(str(config_path), out_dir=str(out_stream))

    out_mem = tmp_path / "out_mem"
    cfg_mem = dict(cfg_base)
    cfg_mem["output"] = dict(cfg_base["output"])
    cfg_mem["output"]["store_forecasts_in_memory"] = True
    cfg_holder["cfg"] = cfg_mem
    runner.backtest_from_config(str(config_path), out_dir=str(out_mem))

    rows_stream = _read_metrics(out_stream / "metrics.csv")
    rows_mem = _read_metrics(out_mem / "metrics.csv")
    _assert_metrics_equal(rows_stream, rows_mem)

    assert len(rows_stream) == 1
    assert float(rows_stream[0]["coverage_50"]) == 1.0
    assert float(rows_stream[0]["crps"]) == 0.0
    assert float(rows_stream[0]["rmse"]) == 0.0
    assert float(rows_stream[0]["mae"]) == 0.0
