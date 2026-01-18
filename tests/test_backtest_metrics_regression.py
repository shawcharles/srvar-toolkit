import csv

import numpy as np
import pandas as pd


def test_backtest_metrics_csv_regression(monkeypatch, tmp_path) -> None:
    import srvar.backtest as backtest
    import srvar.config as config_mod
    from srvar import runner
    from srvar.results import FitResult, ForecastResult

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"date": dates, "y": [0.0, 0.0, 0.0, 1.0]})
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
            "horizons": [1, 2],
            "draws": 5,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-02-01",
            "origin_end": "2000-02-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": True, "intervals": [0.5], "use_latent": False},
            "crps": {"enabled": True, "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    metrics_path = out_dir / "metrics.csv"
    header = metrics_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "variable,horizon,crps,rmse,mae,coverage_50"

    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["variable"] == "y"
    assert rows[0]["horizon"] == "1"
    assert float(rows[0]["crps"]) == 0.0
    assert float(rows[0]["rmse"]) == 0.0
    assert float(rows[0]["mae"]) == 0.0
    assert float(rows[0]["coverage_50"]) == 1.0

    assert rows[1]["variable"] == "y"
    assert rows[1]["horizon"] == "2"
    assert float(rows[1]["crps"]) == 1.0
    assert float(rows[1]["rmse"]) == 1.0
    assert float(rows[1]["mae"]) == 1.0
    assert float(rows[1]["coverage_50"]) == 0.0
