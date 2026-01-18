import csv
from io import StringIO

import pytest


def test_compare_metrics_cli_ratio(capsys, tmp_path) -> None:
    from srvar import cli

    base = tmp_path / "baseline.csv"
    cand = tmp_path / "candidate.csv"

    base.write_text(
        "variable,horizon,rmse,crps\ny1,1,2.0,0.4\ny1,2,4.0,0.8\n",
        encoding="utf-8",
    )
    cand.write_text(
        "variable,horizon,rmse,crps\ny1,1,1.0,0.2\ny1,2,6.0,1.2\n",
        encoding="utf-8",
    )

    rc = cli.main(["compare-metrics", str(base), str(cand), "--mode", "ratio"])
    assert rc == 0

    out = capsys.readouterr().out
    rows = list(csv.DictReader(StringIO(out)))
    assert len(rows) == 2
    assert float(rows[0]["rmse_rel"]) == pytest.approx(0.5)
    assert float(rows[1]["crps_rel"]) == pytest.approx(1.5)
