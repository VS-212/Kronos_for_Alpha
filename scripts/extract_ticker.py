"""
Extract single ticker data from wide-format feats for inference.

Usage:
  python scripts/extract_ticker.py --ticker SBER --mode test
  python scripts/extract_ticker.py --ticker SBER --mode all

Output: data/tickers/SBER/feats_test.npy, data/tickers/SBER/timestamps_test.npy
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICKER_NAMES = ["SBER", "LKOH", "GAZP", "ALRS", "ROSN", "NVTK", "PLZL", "GMKN", "IMOEX"]
OHLCV = ["open", "high", "low", "close", "volume"]
SPLITS = ["train", "val", "test"]
DATA_DIR = Path("data")


def ticker_column_indices(ticker: str) -> slice:
    idx = TICKER_NAMES.index(ticker)
    return slice(idx * 5, idx * 5 + 5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract single ticker from wide feats")
    parser.add_argument("--ticker", default="SBER", choices=TICKER_NAMES, help="Ticker symbol")
    parser.add_argument("--mode", default="test", choices=["train", "val", "test", "all"],
                        help="Split mode (or 'all' for all splits)")
    parser.add_argument("--output", default="data/tickers", help="Output directory")
    args = parser.parse_args()

    ticker = args.ticker
    out_dir = Path(args.output) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = SPLITS if args.mode == "all" else [args.mode]
    col_slice = ticker_column_indices(ticker)

    for split in splits:
        feats_path = DATA_DIR / f"feats_{split}.npy"
        ts_path = DATA_DIR / f"timestamps_{split}.npy"

        if not feats_path.exists():
            print(f"  SKIP {split}: {feats_path} not found")
            continue

        feats = np.load(feats_path)  # (N, 45)
        timestamps = np.load(ts_path, allow_pickle=True)  # (N,)

        ticker_feats = feats[:, col_slice]  # (N, 5)
        ticker_ts = timestamps

        np.save(out_dir / f"feats_{split}.npy", ticker_feats)
        np.save(out_dir / f"timestamps_{split}.npy", ticker_ts)
        print(f"  {split}: {ticker_feats.shape} → {out_dir}/feats_{split}.npy")
        print(f"         {ticker_ts.shape} → {out_dir}/timestamps_{split}.npy")

    print(f"Done. Ticker {ticker} data in {out_dir}/")


if __name__ == "__main__":
    main()
