import csv

import numpy as np
import pandas as pd


def _read_single_metric(path, *, key: str) -> float:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return float(rows[0][key])


def test_backtest_elb_censors_realized_values(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "R": [0.3, 0.3, 0.0, 0.0]})
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
        sims = np.full((int(draws), hmax, int(fit.dataset.N)), 0.25, dtype=float)
        mean = sims.mean(axis=0)
        quantiles = {float(q): np.quantile(sims, q=float(q), axis=0) for q in quantile_levels}
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=mean,
            quantiles=quantiles,
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    cfg_base = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["R"],
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
            "coverage": {"enabled": False, "intervals": []},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    cfg_holder = {"cfg": cfg_base}
    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg_holder["cfg"])

    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    runner.backtest_from_config(str(config_path), out_dir=str(tmp_path / "out_no_censor"))
    rmse_no_censor = _read_single_metric(tmp_path / "out_no_censor" / "metrics.csv", key="rmse")

    cfg_elb = dict(cfg_base)
    cfg_elb["evaluation"] = dict(cfg_base["evaluation"])
    cfg_elb["evaluation"]["elb_censor"] = {
        "enabled": True,
        "bound": 0.25,
        "variables": ["R"],
        "censor_realized": True,
        "censor_forecasts": False,
    }
    cfg_holder["cfg"] = cfg_elb

    runner.backtest_from_config(str(config_path), out_dir=str(tmp_path / "out_elb_censor"))
    rmse_elb = _read_single_metric(tmp_path / "out_elb_censor" / "metrics.csv", key="rmse")

    assert np.isclose(rmse_no_censor, 0.25)
    assert np.isclose(rmse_elb, 0.0)


def test_backtest_elb_can_censor_forecast_draws(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "R": [0.3, 0.3, 0.0, 0.0]})
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
        sims = np.full((int(draws), hmax, int(fit.dataset.N)), 0.0, dtype=float)
        mean = sims.mean(axis=0)
        quantiles = {float(q): np.quantile(sims, q=float(q), axis=0) for q in quantile_levels}
        return ForecastResult(
            variables=list(fit.dataset.variables),
            horizons=list(horizons),
            draws=sims,
            mean=mean,
            quantiles=quantiles,
        )

    monkeypatch.setattr(backtest, "fit", fake_fit)
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    cfg_base = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["R"],
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
            "elb_censor": {
                "enabled": True,
                "bound": 0.25,
                "variables": ["R"],
                "censor_realized": True,
                "censor_forecasts": False,
            },
            "coverage": {"enabled": False, "intervals": []},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    cfg_holder = {"cfg": cfg_base}
    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg_holder["cfg"])

    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    runner.backtest_from_config(str(config_path), out_dir=str(tmp_path / "out_uncensored_fc"))
    rmse_uncensored_fc = _read_single_metric(
        tmp_path / "out_uncensored_fc" / "metrics.csv", key="rmse"
    )

    cfg_censored_fc = dict(cfg_base)
    cfg_censored_fc["evaluation"] = dict(cfg_base["evaluation"])
    cfg_censored_fc["evaluation"]["elb_censor"] = dict(cfg_base["evaluation"]["elb_censor"])
    cfg_censored_fc["evaluation"]["elb_censor"]["censor_forecasts"] = True
    cfg_holder["cfg"] = cfg_censored_fc

    runner.backtest_from_config(str(config_path), out_dir=str(tmp_path / "out_censored_fc"))
    rmse_censored_fc = _read_single_metric(tmp_path / "out_censored_fc" / "metrics.csv", key="rmse")

    assert np.isclose(rmse_uncensored_fc, 0.25)
    assert np.isclose(rmse_censored_fc, 0.0)
