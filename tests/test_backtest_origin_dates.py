import numpy as np
import pandas as pd


def test_backtest_origin_start_end_dates(monkeypatch, tmp_path) -> None:
    import srvar.config as config_mod
    from srvar import runner

    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "date": dates,
            "y1": np.linspace(0.0, 1.0, len(dates)),
            "y2": np.linspace(1.0, 0.0, len(dates)),
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
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": "default"},
        "sampler": {"draws": 20, "burn_in": 0, "thin": 1, "seed": 123},
        "backtest": {
            "mode": "expanding",
            "min_obs": 5,
            "step": 1,
            "horizons": [1],
            "draws": 10,
            "quantile_levels": [0.1, 0.5, 0.9],
            "origin_start": "2000-06-01",
            "origin_end": "2000-08-01",
        },
        "output": {"out_dir": str(tmp_path / "out"), "save_plots": False, "save_forecasts": False},
        "evaluation": {
            "metrics_table": False,
            "coverage": {"enabled": False},
            "crps": {"enabled": False},
        },
    }

    config_path = tmp_path / "config.yml"
    config_path.write_text("# dummy; load_config is monkeypatched\n", encoding="utf-8")

    monkeypatch.setattr(config_mod, "load_config", lambda p: cfg)

    summaries: list[dict] = []

    def progress(event: str, payload: dict) -> None:
        if event == "summary" and payload.get("kind") == "backtest":
            summaries.append(payload)

    runner.backtest_from_config(str(config_path), out_dir=str(tmp_path / "out"), progress=progress)

    assert summaries, "expected backtest summary event"
    assert int(summaries[-1]["origins"]) == 3
