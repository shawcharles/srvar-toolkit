from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_dataset(
    *,
    term_csv: str | Path = "data/T10Y2Y.csv",
    nfci_csv: str | Path = "data/NFCI.csv",
    wuxia_xlsx: str | Path = "data/WuXia/WuXiaShadowRate.xlsx",
    out_csv: str | Path = "data/cache/term_nfci_wuxia_quarterly.csv",
) -> Path:
    term = pd.read_csv(term_csv)
    nfci = pd.read_csv(nfci_csv)
    wuxia_raw = pd.read_excel(wuxia_xlsx, sheet_name="Data")

    term = term.rename(columns={"observation_date": "date"})
    nfci = nfci.rename(columns={"observation_date": "date"})
    term["date"] = pd.to_datetime(term["date"], errors="raise")
    nfci["date"] = pd.to_datetime(nfci["date"], errors="raise")

    col_date = wuxia_raw.columns[0]
    col_shadow = wuxia_raw.columns[2]
    wuxia = wuxia_raw.loc[:, [col_date, col_shadow]].copy()
    wuxia.columns = ["date", "WuXiaShadow"]
    wuxia["date"] = pd.to_datetime(wuxia["date"], errors="coerce")
    wuxia["WuXiaShadow"] = pd.to_numeric(wuxia["WuXiaShadow"], errors="coerce")
    wuxia = wuxia.dropna(subset=["date", "WuXiaShadow"]).copy()
    wuxia = wuxia[wuxia["date"].dt.month.isin([1, 4, 7, 10])]

    merged = term.loc[:, ["date", "T10Y2Y"]].merge(
        nfci.loc[:, ["date", "NFCI"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        wuxia.loc[:, ["date", "WuXiaShadow"]],
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
        description="Prepare a local quarterly benchmark dataset from T10Y2Y, NFCI, and Wu-Xia."
    )
    ap.add_argument("--term-csv", type=str, default="data/T10Y2Y.csv")
    ap.add_argument("--nfci-csv", type=str, default="data/NFCI.csv")
    ap.add_argument("--wuxia-xlsx", type=str, default="data/WuXia/WuXiaShadowRate.xlsx")
    ap.add_argument("--out", type=str, default="data/cache/term_nfci_wuxia_quarterly.csv")
    args = ap.parse_args()

    out_path = build_dataset(
        term_csv=args.term_csv,
        nfci_csv=args.nfci_csv,
        wuxia_xlsx=args.wuxia_xlsx,
        out_csv=args.out,
    )
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
