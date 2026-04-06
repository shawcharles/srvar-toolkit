import subprocess
import sys

import pandas as pd

from srvar.config import load_config


def test_prepare_vintage_macro15_benchmark_script_writes_expected_dataset(tmp_path) -> None:
    out_csv = tmp_path / "vintage_macro15.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_vintage_macro15_benchmark.py",
            "--vintage",
            "2022Q3",
            "--out",
            str(out_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"wrote={out_csv}" in completed.stdout
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert list(df.columns) == [
        "date",
        "GDP",
        "UNRATE",
        "CPIAUCSL",
        "FEDFUNDS",
        "PAYEMS",
        "HOUST",
        "INDPRO",
        "MCUMFN",
        "EXUSUK",
        "M2SL",
        "PINCOME",
        "PCECC96",
        "PPIACO",
        "GS10",
        "BAA",
    ]
    assert len(df) == 202
    assert df.iloc[0]["date"] == "1972-04-01"
    assert df.iloc[-1]["date"] == "2022-07-01"
    assert not df.isna().any().any()


def test_vintage_macro15_backtest_config_parses() -> None:
    cfg = load_config("config/vintage_macro15_backtest_homoskedastic.yaml")
    assert cfg["prior"]["method"] == "minnesota_legacy"
    assert cfg["model"]["p"] == 4
    assert "volatility" not in cfg["model"]


def test_vintage_macro15_diagonal_sv_backtest_config_parses() -> None:
    cfg = load_config("config/vintage_macro15_backtest_diagonal_sv.yaml")
    assert cfg["prior"]["method"] == "minnesota_legacy"
    assert cfg["model"]["p"] == 4
    assert cfg["model"]["volatility"]["covariance"] == "diagonal"
