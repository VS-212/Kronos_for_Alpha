"""
M-BACKTEST: Walk-forward validation pipeline
Contract: ticker OHLCV + Kronos model → per-window samples parquet + monthly metrics
Status: ✅ ready
"""

"""
Walk-forward calibration.

Strategy:
  - Load Mamba combined dataset (wide parquet).
  - Extract per-ticker OHLCV, main-session filter, 10-min freq.
  - Generate non-overlapping windows (step = pred_len).
  - Group windows by calendar month.
  - Batch-inference each month's windows (one GPU forward pass).
  - Save per-window samples + monthly aggregate metrics.

Usage:
    python -m src.evaluation.walk_forward --ticker SBER --month 2025-01
    python -m src.evaluation.walk_forward --ticker SBER --full
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.output import (
    compute_monthly_metrics,
    save_monthly_metrics,
    save_samples,
    save_summary,
)


def load_mamba_data(path: str) -> pd.DataFrame:
    """Load the wide-format Mamba dataset and set timestamp index."""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def extract_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract OHLCV columns for a single ticker from the wide Mamba df."""
    cols = [f"{ticker}_{col}" for col in ["open", "high", "low", "close", "volume"]]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")
    out = df[cols].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    return out


def filter_main_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only main session 10:00-18:40."""
    start_t = pd.Timestamp("10:00").time()
    end_t = pd.Timestamp("18:40").time()
    mask = (df.index.time >= start_t) & (df.index.time <= end_t)
    return df[mask]


def generate_windows(
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    step: int,
    start_date: str,
    end_date: str,
) -> list:
    """Generate non-overlapping prediction windows.

    Each window = x_df (lookback candles) + y_actual (pred_len candles).
    Only windows where the prediction period falls within [start_date, end_date].
    """
    windows = []

    max_start = len(df) - lookback - pred_len
    if max_start <= 0:
        return windows

    for i in range(0, max_start + 1, step):
        pred_ts = df.index[i + lookback]
        if pred_ts < pd.Timestamp(start_date) or pred_ts > pd.Timestamp(end_date):
            continue

        x_df = df.iloc[i : i + lookback]
        y_df = df.iloc[i + lookback : i + lookback + pred_len]

        if len(y_df) < pred_len:
            continue

        windows.append(
            {
                "start_idx": i,
                "pred_ts": pred_ts,
                "month": pred_ts.strftime("%Y-%m"),
                "x_df": x_df,
                "y_df": y_df,
            }
        )

    return windows


def group_windows_by_month(windows: list) -> dict:
    """Group windows list by month string."""
    groups = {}
    for w in windows:
        m = w["month"]
        if m not in groups:
            groups[m] = []
        groups[m].append(w)
    return dict(sorted(groups.items()))


def run_batch_inference(
    model,
    batch_windows: list,
    pred_len: int,
    T: float,
    top_p: float,
    sample_count: int,
    batch_size: int = 50,
) -> list:
    """Run batch inference on a list of windows, split into sub-batches.

    T4 (16 GB) OOMs at ~130 windows × 5 samples (885 sequences).
    Split into sub-batches of `batch_size` to stay within memory.

    Returns list of sample arrays, one per window (sample_count, pred_len, 6).
    """
    import gc

    import torch

    all_samples = [None] * len(batch_windows)
    for i in range(0, len(batch_windows), batch_size):
        chunk = batch_windows[i : i + batch_size]
        df_list = [w["x_df"] for w in chunk]
        chunk_results = model.predict_samples_batch(
            df_list,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )
        for j, arr in enumerate(chunk_results):
            all_samples[i + j] = arr

        gc.collect()
        torch.cuda.empty_cache()

    return all_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description="Walk-forward calibration")
    parser.add_argument("--ticker", default="SBER", help="Ticker symbol")
    parser.add_argument("--month", help="Single month to process (e.g. 2025-01)")
    parser.add_argument("--full", action="store_true", help="Process all months 2025-2026")
    parser.add_argument("--gpu", default="cuda", help="Device")
    args = parser.parse_args(argv)

    # ── Config ──
    global_path = Path(__file__).resolve().parent.parent.parent / "config" / "global.yaml"
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(global_path) as f:
        global_cfg = yaml.safe_load(f)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mamba_path = os.environ.get("MAMBA_PATH") or global_cfg["data"]["mamba_path"]
    lookback = global_cfg["data"]["lookback_candles"]
    pred_len = cfg["walk_forward"]["pred_len"]
    step = cfg["walk_forward"].get("step", pred_len)
    T = cfg["walk_forward"]["temperature"]
    top_p = cfg["walk_forward"]["top_p"]
    sample_count = cfg["walk_forward"]["sample_count"]
    output_dir = cfg["walk_forward"]["output_dir"]

    ticker = args.ticker

    print("=" * 60)
    print(f"WALK-FORWARD CALIBRATION")
    print(f"Ticker: {ticker}")
    print(f"pred_len={pred_len}, step={step}, sc={sample_count}")
    print(f"T={T}, top_p={top_p}")
    print("=" * 60)

    # ── Load data ──
    print(f"\nLoading Mamba data from {mamba_path}...")
    mamba_df = load_mamba_data(mamba_path)
    print(f"  Total rows: {len(mamba_df)}, range: {mamba_df.index.min()} → {mamba_df.index.max()}")

    ticker_df = extract_ticker(mamba_df, ticker)
    ticker_df = filter_main_session(ticker_df)
    print(f"  {ticker} main-session rows: {len(ticker_df)}")

    # ── Generate windows ──
    start_date = global_cfg["data"]["walk_forward_start"]
    end_date = global_cfg["data"]["walk_forward_end"]

    windows = generate_windows(ticker_df, lookback, pred_len, step, start_date, end_date)
    print(f"  Total windows: {len(windows)}")

    if not windows:
        print("ERROR: No valid windows. Check dates and lookback.")
        return

    by_month = group_windows_by_month(windows)

    if args.month:
        months_to_process = [args.month]
    elif args.full:
        months_to_process = list(by_month.keys())
    else:
        months_to_process = [list(by_month.keys())[0]]

    print(f"  Months to process: {months_to_process}")

    # ── Model ──
    print("\nLoading Kronos model...")
    from src.core.kronos import KronosModel

    model = KronosModel(
        model_name=global_cfg["model"]["name"],
        tokenizer_name=global_cfg["model"]["tokenizer"],
        device=args.gpu,
        max_context=global_cfg["model"]["max_context"],
        session_filter=False,  # already filtered
    )
    model.load()
    print("  Model loaded.")

    config = {"T": T, "top_p": top_p, "pred_len": pred_len, "sample_count": sample_count}
    all_rows = []
    all_results = []

    # ── Process each month ──
    for month in months_to_process:
        month_windows = by_month.get(month, [])
        if not month_windows:
            print(f"\n  Month {month}: no windows, skipping.")
            continue

        print(f"\n── {month}: {len(month_windows)} windows ──")
        sample_arrays = run_batch_inference(
            model,
            month_windows,
            pred_len,
            T,
            top_p,
            sample_count,
        )

        for w, samples_6d in zip(month_windows, sample_arrays):
            close_samples = samples_6d[:, :, 3]
            actuals = w["y_df"][["open", "high", "low", "close"]].values.astype(np.float32)
            all_rows.append(
                {
                    "window_id": w["start_idx"],
                    "pred_ts": w["pred_ts"],
                    "month": w["month"],
                    "prev_close": float(w["x_df"].iloc[-1]["close"]),
                    "samples": close_samples,
                    "actuals": actuals,
                }
            )

        print(f"  → {len(month_windows)} windows done, total accumulated: {len(all_rows)}")

    # ── Save combined output ──
    if not all_rows:
        print("ERROR: No rows produced.")
        return

    out_path = save_samples(ticker, all_rows, pred_len, sample_count, config, output_dir)
    print(f"\nSaved all samples: {out_path} ({len(all_rows)} windows)")

    # ── Compute monthly metrics from the full parquet ──
    df = pd.read_parquet(out_path)
    monthly_df = compute_monthly_metrics(df, pred_len, sample_count)
    if not monthly_df.empty:
        metrics_path = save_monthly_metrics(ticker, monthly_df, config, output_dir)
        print(f"Saved monthly metrics: {metrics_path}")
        for _, row in monthly_df.iterrows():
            print(
                f"  {row['month']}: n={int(row['n_windows'])}  "
                f"cons_dir_acc={row.get('consensus_dir_acc', 0):.3f}  "
                f"dir_acc={row.get('direction_accuracy', 0):.3f}  "
                f"exp={row.get('expectancy', 0):.6f}"
            )
            all_results.append(row.to_dict())

    # ── Summary ──
    if all_results:
        best = max(
            all_results, key=lambda r: (r.get("consensus_dir_acc", 0), r.get("expectancy", 0))
        )
        print(
            f"\nBEST MONTH: {best['month']}  "
            f"cons_dir_acc={best.get('consensus_dir_acc', 0):.3f}  "
            f"exp={best.get('expectancy', 0):.6f}"
        )

        avg_metrics = {}
        keys = [k for k in all_results[0].keys() if k not in ("month", "n_windows")]
        for k in keys:
            vals = [r[k] for r in all_results if not np.isnan(r.get(k, np.nan))]
            avg_metrics[k] = float(np.mean(vals)) if vals else 0.0
        avg_metrics["n_months"] = len(all_results)
        avg_metrics["n_windows_total"] = sum(r.get("n_windows", 0) for r in all_results)
        print(f"\nAVERAGE ACROSS ALL MONTHS:")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        all_configs = [
            {
                "ticker": ticker,
                "config": config,
                "months": len(all_results),
                "avg_metrics": avg_metrics,
            }
        ]
        save_summary([ticker], all_configs, output_dir)
        print(f"\nSummary: {Path(output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
