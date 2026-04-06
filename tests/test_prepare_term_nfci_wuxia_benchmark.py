import subprocess
import sys

import pandas as pd

from srvar.config import load_config


def test_prepare_term_nfci_wuxia_benchmark_script_writes_expected_dataset(tmp_path) -> None:
    out_csv = tmp_path / "term_nfci_wuxia.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_term_nfci_wuxia_benchmark.py",
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
    assert list(df.columns) == ["date", "T10Y2Y", "NFCI", "WuXiaShadow"]
    assert len(df) == 129
    assert df.iloc[0]["date"] == "1990-01-01"
    assert df.iloc[-1]["date"] == "2022-01-01"


def test_term_nfci_wuxia_backtest_config_parses() -> None:
    cfg = load_config("config/term_nfci_wuxia_backtest.yaml")
    assert cfg["prior"]["method"] == "minnesota_legacy"
    assert cfg["model"]["volatility"]["covariance"] == "diagonal"
