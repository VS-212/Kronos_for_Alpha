"""
Preprocess 1-hour MOEX data into .npy format for inference.

Splits by date (same as 10-min pipeline), stacks tickers in wide format.
Per-window z-score normalization is handled by the predictor at inference time.

Usage:
  python scripts/preprocess_1h.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICKER_NAMES = ["SBER", "LKOH", "GAZP", "ALRS", "ROSN", "NVTK", "PLZL", "GMKN", "IMOEX"]
OHLCV = ["open", "high", "low", "close", "volume"]

# Same split dates as 10-min pipeline (from actual data, not config)
SPLITS = {
    "train": ("2023-01-03", "2025-05-22"),
    "val": ("2025-05-22", "2025-11-20"),
    "test": ("2025-11-20", "2026-05-28"),
}


def load_ticker_parquet(ticker: str, raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / f"{ticker}.parquet"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Save_parquets creates a 'timestamp' column (naive or tz-aware)
    if "begin" in df.columns:
        df = df.set_index("begin")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    # Normalize timezone: UTC → MSK → naive
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("Europe/Moscow")
    else:
        df.index = pd.to_datetime(df.index).tz_convert("Europe/Moscow")
    df.index = df.index.tz_localize(None)
    return df


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only main session 10:00-18:40 MSK."""
    if len(df) == 0:
        return df
    time_filter = (df.index.hour * 60 + df.index.minute >= 600) & (
        df.index.hour * 60 + df.index.minute <= 1120
    )
    return df[time_filter].copy()


def stacked_wide(ticker_dfs: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Stack tickers into wide format (N, 45) matching existing feats."""
    n = len(timestamps)
    cols = []
    for ticker in TICKER_NAMES:
        df = ticker_dfs.get(ticker)
        if df is not None and len(df) > 0:
            reindexed = df.reindex(timestamps, method="nearest", tolerance="1h")
            for feat in OHLCV:
                cols.append(reindexed[feat].values.astype(np.float32))
        else:
            for _ in OHLCV:
                cols.append(np.zeros(n, dtype=np.float32))
    return np.column_stack(cols)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess 1-hour MOEX data")
    parser.add_argument("--input", default="data/v3/1h/raw", help="Parquet input directory")
    parser.add_argument("--output", default="data/v3/1h/processed", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all tickers
    all_data = {}
    for ticker in TICKER_NAMES:
        df = load_ticker_parquet(ticker, args.input)
        df = filter_session(df)
        if len(df) > 0:
            all_data[ticker] = df
            print(f"  {ticker}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
        else:
            print(f"  {ticker}: NO DATA")

    # Split by date
    for split_name, (start_date, end_date) in SPLITS.items():
        split_start = pd.Timestamp(start_date)
        split_end = pd.Timestamp(end_date)

        # Find common timestamps across all tickers
        common_ts = None
        for ticker in TICKER_NAMES:
            df = all_data.get(ticker)
            if df is not None and len(df) > 0:
                mask = (df.index >= split_start) & (df.index < split_end)
                ts = df.index[mask]
                if common_ts is None:
                    common_ts = set(ts)
                else:
                    common_ts = common_ts & set(ts)

        if common_ts is None or len(common_ts) == 0:
            print(f"  {split_name}: no common timestamps")
            continue

        common_ts = pd.DatetimeIndex(sorted(common_ts))
        feats = stacked_wide(all_data, common_ts)

        np.save(out_dir / f"feats_{split_name}.npy", feats)
        np.save(
            out_dir / f"timestamps_{split_name}.npy",
            np.array([str(ts) for ts in common_ts]),
        )
        print(f"  {split_name}: {len(common_ts)} bars → {feats.shape}")

    print(f"Done. Data saved to {args.output}/")


if __name__ == "__main__":
    main()
