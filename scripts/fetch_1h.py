"""
Fetch 1-hour MOEX data for 9 tickers. MOEX ISS limits:
- 500 rows/page (auto-paginated by apimoex)
- ~0.5s rate limit (built into Fetcher class)
- 5 retries with exponential backoff
- Monthly chunking prevents API timeouts on large ranges

Sync logic:
  10-min lookback=500 bars (~10 trading days)
  1-hour  lookback=510 bars (~64 trading days)
  → 1-hour data starts 2022-10-01 (3 months before 10-min)
  → Both models predict at same calendar time in test period

Usage:
  python scripts/fetch_1h.py [--start 2022-10-01] [--end 2026-05-28]
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetcher import Fetcher, add_amount_column, save_parquets

OUR_TICKERS = ["SBER", "LKOH", "GAZP", "ALRS", "ROSN", "NVTK", "PLZL", "GMKN", "IMOEX"]


def month_chunks(start_date: str, end_date: str):
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    chunks = []
    cur = s
    while cur <= e:
        month_end = cur.replace(day=28) + timedelta(days=4)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        chunk_end = min(month_end, e)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch 1-hour MOEX data (9 tickers)")
    parser.add_argument("--start", default="2022-10-01", help="Start date")
    parser.add_argument("--end", default="2026-05-28", help="End date")
    parser.add_argument("--output", default="data/v3/1h/raw", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetcher = Fetcher()
    chunks = month_chunks(args.start, args.end)
    total_reqs = len(chunks) * len(OUR_TICKERS)
    print(f"Fetching {len(OUR_TICKERS)} tickers x {len(chunks)} months = {total_reqs} requests", flush=True)
    print(f"  Tickers: {', '.join(OUR_TICKERS)}", flush=True)
    print(f"  Period:  {args.start} -> {args.end}", flush=True)
    print(f"  Interval: 60 min", flush=True)
    print(flush=True)

    collected = {t: [] for t in OUR_TICKERS}
    done = 0
    errors = 0

    for cs, ce in chunks:
        for ticker in OUR_TICKERS:
            try:
                df = fetcher.fetch_candles(ticker, interval=60, start=cs, end=ce)
                if df is not None and len(df) > 0:
                    collected[ticker].append(df)
                done += 1
            except Exception as e:
                errors += 1
                print(f"  ERROR {ticker} {cs}->{ce}: {e}", flush=True)
                done += 1

            if done % 10 == 0:
                pct = done / total_reqs * 100
                print(f"  [{done}/{total_reqs} {pct:.0f}%] {errors} errors", flush=True)

    result = {}
    for ticker in OUR_TICKERS:
        if collected[ticker]:
            df = pd.concat(collected[ticker])
            df = df[~df.index.duplicated(keep="first")].sort_index()
            result[ticker] = df
            print(f"  {ticker}: {len(df)} bars ({df.index[0]} -> {df.index[-1]})", flush=True)
        else:
            result[ticker] = pd.DataFrame()
            print(f"  {ticker}: NO DATA", flush=True)

    with_amount = add_amount_column(result)
    save_parquets(with_amount, str(out_dir))
    print(f"Done. 1-hour data saved to {args.output}/", flush=True)


if __name__ == "__main__":
    main()
