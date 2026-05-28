"""
TEMPLATE: Fetch MOEX data and save as Mamba-compatible wide parquet.
Source: kronos-artifact/scripts/fetch_sber.py
Purpose: Reference example for MOEX ISS API data fetching
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.moex import MOEXDataSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2026-05-26")
    parser.add_argument("--end", default="2026-05-28")
    parser.add_argument("--output", default="/tmp/alpha_data/SBER_fetched.parquet")
    args = parser.parse_args()

    ds = MOEXDataSource()
    candles = ds.fetch_candles(args.ticker, 5, args.start, args.end)
    if candles is None or candles.empty:
        print("No data fetched.")
        return

    if candles.index.tz is not None:
        candles.index = candles.index.tz_localize(None)

    out = pd.DataFrame({"timestamp": candles.index})
    for suffix in ["open", "high", "low", "close", "volume"]:
        out[f"{args.ticker}_{suffix}"] = candles[suffix].values.astype(np.float32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved {len(out)} rows \u2192 {args.output}")


if __name__ == "__main__":
    main()
