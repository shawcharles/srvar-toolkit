import numpy as np
import pandas as pd


def test_backtest_factor_sv_smoke(tmp_path, monkeypatch) -> None:
    import srvar.config as config_mod
    from srvar import runner

    rng = np.random.default_rng(123)
    dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    df = pd.DataFrame(
        {
            "date": dates,
            "y1": rng.standard_normal(len(dates)),
            "y2": rng.standard_normal(len(dates)),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y1", "y2"],
            "dropna": True,
        },
        "model": {
            "p": 2,
            "include_intercept": True,
            "volatility": {
                "enabled": True,
                "dynamics": "rw",
                "covariance": "factor",
                "k_factors": 1,
                "loading_prior_var": 1.0,
                "store_factor_draws": False,
            },
        },
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 40, "burn_in": 10, "thin": 2, "seed": 0},
        "backtest": {
            "mode": "expanding",
            "min_obs": 12,
            "step": 1,
            "horizons": [1, 2],
            "draws": 30,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2001-03-01",
            "origin_end": "2001-03-01",
        },
        "evaluation": {
            "metrics_table": True,
            "coverage": {"enabled": True, "intervals": [0.5], "use_latent": False},
            "crps": {"enabled": True, "use_latent": False},
            "pit": {"enabled": False},
        },
        "output": {
            "out_dir": str(tmp_path / "out"),
            "save_plots": False,
            "save_forecasts": False,
            "store_forecasts_in_memory": False,
        },
    }

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)
    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    runner.backtest_from_config(str(config_path), out_dir=str(out_dir))

    metrics_path = out_dir / "metrics.csv"
    assert metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert len(metrics) == 4
    assert set(metrics["variable"]) == {"y1", "y2"}
    assert set(metrics["horizon"]) == {1, 2}
    assert np.all(np.isfinite(metrics["rmse"]))
    assert np.all(np.isfinite(metrics["mae"]))
