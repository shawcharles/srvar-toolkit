from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from srvar.data.transformations import tcode_1d
from srvar.data.vintages import load_vintages_from_dir

DEFAULT_VINTAGE = "2022Q3"
VARIABLE_SPECS: list[tuple[str, int, float]] = [
    ("GDP", 5, 4.0),
    ("UNRATE", 1, 1.0),
    ("CPIAUCSL", 5, 4.0),
    ("FEDFUNDS", 1, 1.0),
    ("PAYEMS", 5, 4.0),
    ("HOUST", 4, 1.0),
    ("INDPRO", 5, 4.0),
    ("MCUMFN", 1, 1.0),
    ("EXUSUK", 5, 4.0),
    ("M2SL", 5, 4.0),
    ("PINCOME", 5, 4.0),
    ("PCECC96", 5, 4.0),
    ("PPIACO", 5, 4.0),
    ("GS10", 1, 1.0),
    ("BAA", 1, 1.0),
]


def build_dataset(
    *,
    data_dir: str | Path = "data",
    vintage: str = DEFAULT_VINTAGE,
    out_csv: str | Path = "data/cache/vintage_macro15_quarterly.csv",
) -> Path:
    vintages = load_vintages_from_dir(data_dir=data_dir)
    vintage_period = pd.Period(vintage, freq="Q")
    if vintage_period not in vintages:
        available = ", ".join(str(v) for v in sorted(vintages))
        raise ValueError(f"vintage {vintage_period} not found; available vintages: {available}")

    raw = vintages[vintage_period]
    transformed: dict[str, object] = {}
    for name, tcode, scale in VARIABLE_SPECS:
        series = tcode_1d(raw[name].to_numpy(dtype=float), tcode, var_name=name)
        if scale != 1.0:
            series = scale * series
        transformed[name] = series

    out = pd.DataFrame(transformed, index=raw.index).dropna(how="any")
    out.index = out.index.to_timestamp(how="start")
    out.index.name = "date"

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a local transformed quarterly 15-variable macro benchmark from repo vintages."
        )
    )
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--vintage", type=str, default=DEFAULT_VINTAGE)
    ap.add_argument("--out", type=str, default="data/cache/vintage_macro15_quarterly.csv")
    args = ap.parse_args()

    out_path = build_dataset(data_dir=args.data_dir, vintage=args.vintage, out_csv=args.out)
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
