import csv
import math

import numpy as np
import pandas as pd


def _read_metrics_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    return rows


def test_backtest_coverage_enabled_false_skips_coverage_quantiles(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 3.0]})
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
        sims = np.zeros((int(draws), hmax, int(fit.dataset.N)), dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    def _quantile_called(*args, **kwargs):
        raise AssertionError("np.quantile should not be called when coverage is disabled")

    monkeypatch.setattr(backtest.np, "quantile", _quantile_called)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": True,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 2,
            "step": 1,
            "horizons": [1],
            "draws": 5,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-02-01",
            "origin_end": "2000-02-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": False, "intervals": [0.5, 0.8], "use_latent": False},
            "crps": {"enabled": False},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    rows = _read_metrics_csv(out_dir / "metrics.csv")
    keys = set(rows[0].keys())
    assert not any(k.startswith("coverage_") for k in keys)


def test_backtest_crps_enabled_false_sets_nan(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    import srvar.metrics as metrics_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 3.0]})
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
        sims = np.zeros((int(draws), hmax, int(fit.dataset.N)), dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    def _crps_called(*args, **kwargs):
        raise AssertionError("crps_draws should not be called when CRPS is disabled")

    monkeypatch.setattr(metrics_mod, "crps_draws", _crps_called)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": True,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 2,
            "step": 1,
            "horizons": [1],
            "draws": 5,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-02-01",
            "origin_end": "2000-02-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": False, "intervals": [], "use_latent": False},
            "crps": {"enabled": False, "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    rows = _read_metrics_csv(out_dir / "metrics.csv")
    assert "crps" in rows[0]
    assert math.isnan(float(rows[0]["crps"]))


def test_backtest_wis_enabled_false_omits_column_and_skips_compute(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    import srvar.metrics as metrics_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 3.0]})
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
        sims = np.zeros((int(draws), hmax, int(fit.dataset.N)), dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    def _wis_called(*args, **kwargs):
        raise AssertionError("wis_draws should not be called when WIS is disabled")

    monkeypatch.setattr(metrics_mod, "wis_draws", _wis_called)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": True,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 2,
            "step": 1,
            "horizons": [1],
            "draws": 5,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-02-01",
            "origin_end": "2000-02-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": False, "intervals": [], "use_latent": False},
            "crps": {"enabled": False, "use_latent": False},
            "wis": {"enabled": False, "intervals": [0.5], "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    rows = _read_metrics_csv(out_dir / "metrics.csv")
    assert "wis" not in rows[0]


def test_backtest_pinball_enabled_false_omits_column_and_skips_compute(
    monkeypatch, tmp_path
) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    import srvar.metrics as metrics_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 3.0]})
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
        sims = np.zeros((int(draws), hmax, int(fit.dataset.N)), dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    def _pinball_called(*args, **kwargs):
        raise AssertionError("pinball_draws should not be called when pinball is disabled")

    monkeypatch.setattr(metrics_mod, "pinball_draws", _pinball_called)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": True,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 2,
            "step": 1,
            "horizons": [1],
            "draws": 5,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-02-01",
            "origin_end": "2000-02-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": False, "intervals": [], "use_latent": False},
            "crps": {"enabled": False, "use_latent": False},
            "wis": {"enabled": False, "intervals": [0.5], "use_latent": False},
            "pinball": {"enabled": False, "quantiles": [0.5], "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    rows = _read_metrics_csv(out_dir / "metrics.csv")
    assert "pinball" not in rows[0]


def test_backtest_log_score_enabled_false_omits_column_and_skips_compute(
    monkeypatch, tmp_path
) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    import srvar.metrics as metrics_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 1.0, 2.0, 3.0]})
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
        sims = np.zeros((int(draws), hmax, int(fit.dataset.N)), dtype=float)
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=sims.mean(axis=0),
            quantiles={},
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    def _log_score_called(*args, **kwargs):
        raise AssertionError("log_score_draws should not be called when log_score is disabled")

    monkeypatch.setattr(metrics_mod, "log_score_draws", _log_score_called)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": True,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 5, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 2,
            "step": 1,
            "horizons": [1],
            "draws": 5,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-02-01",
            "origin_end": "2000-02-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": False, "intervals": [], "use_latent": False},
            "crps": {"enabled": False, "use_latent": False},
            "wis": {"enabled": False, "intervals": [0.5], "use_latent": False},
            "pinball": {"enabled": False, "quantiles": [0.5], "use_latent": False},
            "log_score": {"enabled": False, "variance_floor": 1e-12, "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    rows = _read_metrics_csv(out_dir / "metrics.csv")
    assert "log_score" not in rows[0]
