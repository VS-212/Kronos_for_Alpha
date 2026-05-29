"""
Align 10-min and 1-hour predictions to common timestamps for head-to-head comparison.

Synchronization logic:
  10-min lookback=500 bars (~10 trading days), pred_len=12 (2 hours)
   1-hour lookback=510 bars (~64 trading days), pred_len=2  (2 hours)

  Both models predict the SAME 2-hour horizon. Alignment:
  1. Take 10-min predictions at :00 minute (hour boundaries)
  2. Take 1-hour predictions at the same calendar times
  3. Compare close[t+2h] from both

Usage:
  python scripts/align_predictions.py \
    --tf10-preds data/v3/predictions/10min_predlen12/predictions/ \
    --tf10-ts data/v3/predictions/10min_predlen12/timestamps.npy \
    --tf1h-preds data/v3/predictions/1hour_predlen2/predictions/ \
    --tf1h-ts data/v3/predictions/1hour_predlen2/timestamps.npy \
    --output data/v3/ensemble/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TICKER_NAMES = ["SBER", "LKOH", "GAZP", "ALRS", "ROSN", "NVTK", "PLZL", "GMKN", "IMOEX"]


def _load_preds(pred_dir: str) -> dict[str, np.ndarray]:
    """Load per-ticker prediction arrays. Expects {ticker}_preds_pl*.npy files."""
    pred_dir = Path(pred_dir)
    result = {}
    for ticker in TICKER_NAMES:
        files = list(pred_dir.glob(f"{ticker}_preds_pl*.npy"))
        if not files:
            print(f"  WARNING: no predictions for {ticker}")
            continue
        result[ticker] = np.load(files[0])  # (n_windows, sample_count, pred_len, 6) or (n_windows, pred_len, 6)
    return result


def _load_beliefs(belief_dir: str) -> dict[str, np.ndarray]:
    """Load per-ticker belief arrays."""
    belief_dir = Path(belief_dir)
    result = {}
    for ticker in TICKER_NAMES:
        files = list(belief_dir.glob(f"{ticker}_belief_pl*.npy"))
        if not files:
            print(f"  WARNING: no beliefs for {ticker}")
            continue
        result[ticker] = np.load(files[0])  # (n_windows, sample_count, pred_len, 4)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align 10-min and 1-hour predictions")
    parser.add_argument("--tf10-preds", required=True, help="10-min predictions directory")
    parser.add_argument("--tf10-ts", required=True, help="10-min timestamps .npy")
    parser.add_argument("--tf10-belief", default=None, help="10-min belief directory (optional)")
    parser.add_argument("--tf1h-preds", required=True, help="1-hour predictions directory")
    parser.add_argument("--tf1h-ts", required=True, help="1-hour timestamps .npy")
    parser.add_argument("--tf1h-belief", default=None, help="1-hour belief directory (optional)")
    parser.add_argument("--output", default="data/v3/ensemble/", help="Output directory")
    parser.add_argument("--horizon", type=int, default=2, help="Comparison horizon in hours")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load timestamps
    ts_10min = np.load(args.tf10_ts)
    ts_1hour = np.load(args.tf1h_ts)
    ts_10min_dt = pd.DatetimeIndex([pd.Timestamp(t) for t in ts_10min])
    ts_1hour_dt = pd.DatetimeIndex([pd.Timestamp(t) for t in ts_1hour])

    # Find common hour-boundary timestamps
    hour_mask_10 = ts_10min_dt.minute == 0
    common_ts = pd.DatetimeIndex(
        sorted(set(ts_10min_dt[hour_mask_10]) & set(ts_1hour_dt))
    )
    print(f"10-min timestamps:  {len(ts_10min_dt)}")
    print(f"1-hour timestamps:  {len(ts_1hour_dt)}")
    print(f"Aligned hour-boundary timestamps: {len(common_ts)}")
    print(f"  First: {common_ts[0] if len(common_ts) > 0 else 'N/A'}")
    print(f"  Last:  {common_ts[-1] if len(common_ts) > 0 else 'N/A'}")

    if len(common_ts) == 0:
        print("ERROR: No aligned timestamps found. Check that both TFs cover the same period.")
        return

    # Map timestamps to indices
    ts_10min_to_idx = {str(t): i for i, t in enumerate(ts_10min_dt)}
    ts_1hour_to_idx = {str(t): i for i, t in enumerate(ts_1hour_dt)}

    aligned_idx_10 = []
    aligned_idx_1h = []
    for ct in common_ts:
        key = str(ct)
        i10 = ts_10min_to_idx.get(key)
        i1h = ts_1hour_to_idx.get(key)
        if i10 is not None and i1h is not None:
            aligned_idx_10.append(i10)
            aligned_idx_1h.append(i1h)

    # Load predictions
    preds_10min = _load_preds(args.tf10_preds)
    preds_1hour = _load_preds(args.tf1h_preds)
    if args.tf10_belief:
        beliefs_10min = _load_beliefs(args.tf10_belief)
    if args.tf1h_belief:
        beliefs_1hour = _load_beliefs(args.tf1h_belief)

    # Compare per ticker
    horizon_10min_bars = args.horizon * 6  # 6 bars per hour at 10-min
    horizon_1h_bars = args.horizon * 1      # 1 bar per hour at 1-hour

    comparison = {"aligned_timestamps": [str(t) for t in common_ts]}

    for ticker in TICKER_NAMES:
        p10 = preds_10min.get(ticker)
        p1h = preds_1hour.get(ticker)
        if p10 is None or p1h is None:
            continue

        # Select aligned windows
        # preds shape: (n_windows, sample_count, pred_len, 6) or (n_windows, pred_len, 6)
        if p10.ndim == 4:
            p10_close = p10[aligned_idx_10, :, :, 3]  # (N, S, pred_len)
            p10_close = p10_close.mean(axis=1)  # (N, pred_len) — average MC
        else:
            p10_close = p10[aligned_idx_10, :, 3]  # (N, pred_len)

        if p1h.ndim == 4:
            p1h_close = p1h[aligned_idx_1h, :, :, 3]
            p1h_close = p1h_close.mean(axis=1)
        else:
            p1h_close = p1h[aligned_idx_1h, :, 3]

        # Compare at horizon
        comp = {
            "n_aligned": len(common_ts),
            "mean_10min": float(np.mean(p10_close[:, horizon_10min_bars - 1])),
            "mean_1hour": float(np.mean(p1h_close[:, horizon_1h_bars - 1])),
            "std_10min": float(np.std(p10_close[:, horizon_10min_bars - 1])),
            "std_1hour": float(np.std(p1h_close[:, horizon_1h_bars - 1])),
            "corr": float(np.corrcoef(
                p10_close[:, horizon_10min_bars - 1],
                p1h_close[:, horizon_1h_bars - 1],
            )[0, 1] if len(common_ts) > 2 else np.nan),
        }
        comparison[ticker] = comp
        print(f"  {ticker}: corr={comp['corr']:.3f}, "
              f"10m={comp['mean_10min']:.4f}±{comp['std_10min']:.4f}, "
              f"1h={comp['mean_1hour']:.4f}±{comp['std_1hour']:.4f}")

    # Save aligned predictions for backtest
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    np.save(aligned_dir / "timestamps.npy", np.array([str(t) for t in common_ts]))
    np.save(aligned_dir / "idx_10min.npy", np.array(aligned_idx_10))
    np.save(aligned_dir / "idx_1hour.npy", np.array(aligned_idx_1h))

    # Save comparison report
    report_path = out_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison report saved to {report_path}")


if __name__ == "__main__":
    main()
