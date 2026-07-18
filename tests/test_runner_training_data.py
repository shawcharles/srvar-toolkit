from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from srvar.config import ConfigError
from srvar.runner import run_from_config


def _write_backtest_config(tmp_path: Path, *, prior_method: str = "default") -> Path:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=5, freq="MS"),
            "y": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    ).to_csv(csv_path, index=False)
    config = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "date",
            "variables": ["y"],
            "dropna": False,
        },
        "model": {"p": 1, "include_intercept": True},
        "prior": {"family": "niw", "method": prior_method},
        "sampler": {"draws": 2, "burn_in": 0, "thin": 1, "seed": 0},
        "backtest": {"mode": "expanding", "min_obs": 3, "step": 1, "horizons": [1]},
        "evaluation": {"metrics_table": False},
        "output": {"save_plots": False, "save_fit": False, "save_forecast": False},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_validate_only_allows_target_only_missing_backtest_rows(tmp_path: Path) -> None:
    config_path = _write_backtest_config(tmp_path)
    assert run_from_config(config_path, validate_only=True) is None


def test_normal_run_remains_strict_for_full_dataset_missing_values(tmp_path: Path) -> None:
    config_path = _write_backtest_config(tmp_path)
    with pytest.raises(ConfigError, match="training data must contain only finite values"):
        run_from_config(config_path)


def test_validate_only_checks_first_origin_prior_without_fitting(
    monkeypatch, tmp_path: Path
) -> None:
    import srvar.runner as runner

    config_path = _write_backtest_config(tmp_path, prior_method="not-a-prior")
    monkeypatch.setattr(runner, "fit", lambda *args, **kwargs: pytest.fail("fit must not run"))

    with pytest.raises(ConfigError, match="prior.method for family='niw'"):
        run_from_config(config_path, validate_only=True)
