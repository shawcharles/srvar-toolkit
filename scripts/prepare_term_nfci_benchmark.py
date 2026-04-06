from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_dataset(
    *,
    term_csv: str | Path = "data/T10Y2Y.csv",
    nfci_csv: str | Path = "data/NFCI.csv",
    out_csv: str | Path = "data/cache/term_nfci_quarterly.csv",
) -> Path:
    term = pd.read_csv(term_csv)
    nfci = pd.read_csv(nfci_csv)

    term = term.rename(columns={"observation_date": "date"})
    nfci = nfci.rename(columns={"observation_date": "date"})
    term["date"] = pd.to_datetime(term["date"], errors="raise")
    nfci["date"] = pd.to_datetime(nfci["date"], errors="raise")

    merged = term.loc[:, ["date", "T10Y2Y"]].merge(
        nfci.loc[:, ["date", "NFCI"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare a local quarterly benchmark dataset from T10Y2Y and NFCI."
    )
    ap.add_argument("--term-csv", type=str, default="data/T10Y2Y.csv")
    ap.add_argument("--nfci-csv", type=str, default="data/NFCI.csv")
    ap.add_argument("--out", type=str, default="data/cache/term_nfci_quarterly.csv")
    args = ap.parse_args()

    out_path = build_dataset(
        term_csv=args.term_csv,
        nfci_csv=args.nfci_csv,
        out_csv=args.out,
    )
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
