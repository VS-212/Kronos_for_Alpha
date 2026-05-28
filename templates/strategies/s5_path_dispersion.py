"""
TEMPLATE: S5 Path-Dispersion Trend Strength strategy (standalone).
Source: kronos-artifact/alpha/experiments/s5.py
Purpose: Reference example for signal-atom-based strategy with inline backtest/report
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples, reconstruct
from src.signals.atoms import direction, trend_strength, linearity, dispersion


def compute_signal(close_samples: np.ndarray, prev_close: float,
                   min_r2: float = 0.0) -> float:
    """S5 composite signal for one window."""
    ts = trend_strength(close_samples, prev_close)
    lin = linearity(close_samples)

    if lin["mean_r2"] < min_r2:
        return 0.0

    mu = ts["mean_return"]
    signal = ts["trend_strength"] * np.sign(mu) * (0.5 + 0.5 * lin["mean_r2"])
    return float(signal)


def backtest(df: pd.DataFrame, pred_len: int, sample_count: int,
             entry_threshold: float = 0.55,
             sl_pct: float = 0.015,
             tp_pct: float = 0.03) -> pd.DataFrame:
    """Run S5 backtest on samples. Returns trade log.

    Entry: at open of T if signal strength >= entry_threshold.
    Exit: first SL/TP hit using candle high/low, or close at T+pred_len.
    SL checked first (conservative).
    """
    trades = []

    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        signal = compute_signal(close_only, prev_close)
        if signal > entry_threshold:
            pred_dir = 1
        elif signal < -entry_threshold:
            pred_dir = -1
        else:
            continue

        entry_price = float(actuals[0, 0])
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        if pred_dir == -1:
            tp_price, sl_price = entry_price * (1 - tp_pct), entry_price * (1 + sl_pct)

        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])

        for i in range(pred_len):
            hi = float(actuals[i, 1])
            lo = float(actuals[i, 2])

            if pred_dir == 1:  # long
                sl_hit = lo <= sl_price
                tp_hit = hi >= tp_price
            else:  # short
                sl_hit = hi >= sl_price
                tp_hit = lo <= tp_price

            if sl_hit:
                exit_step = i
                exit_reason = "sl"
                exit_price = sl_price
                break
            if tp_hit:
                exit_step = i
                exit_reason = "tp"
                exit_price = tp_price
                break

        trade_return = pred_dir * (exit_price - entry_price) / entry_price

        trades.append({
            "window_id": int(row["window_id"]),
            "pred_ts": row["pred_ts"],
            "month": row["month"],
            "direction": pred_dir,
            "signal": signal,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_step": exit_step,
            "exit_reason": exit_reason,
            "return": trade_return,
        })

    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> dict:
    """Compute backtest metrics from trade log."""
    if len(trades) == 0:
        return {"error": "no trades"}

    n = len(trades)
    wins = int((trades["return"] > 0).sum())
    losses = int((trades["return"] <= 0).sum())
    total_ret = trades["return"].sum()
    avg_ret = trades["return"].mean()
    std_ret = trades["return"].std()
    avg_signal = trades["signal"].mean()

    equity = (1 + trades["return"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = float(drawdown.min())

    sharpe = avg_ret / max(std_ret, 1e-10) * np.sqrt(252 * 6 / 6)
    win_rate = wins / max(n, 1)
    avg_win = float(trades.loc[trades["return"] > 0, "return"].mean()) if wins else 0.0
    avg_loss = float(trades.loc[trades["return"] <= 0, "return"].mean()) if losses else 0.0
    sum_wins = avg_win * wins if wins else 0.0
    sum_losses = abs(avg_loss * losses) if losses else 0.0
    profit_factor = sum_wins / max(sum_losses, 1e-10)

    tp_hits = int((trades["exit_reason"] == "tp").sum())
    sl_hits = int((trades["exit_reason"] == "sl").sum())
    closes = int((trades["exit_reason"] == "close").sum())

    return {
        "n": n,
        "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "total_return_pct": total_ret * 100,
        "avg_return_pct": avg_ret * 100,
        "std_return_pct": std_ret * 100,
        "sharpe": sharpe,
        "mdd_pct": mdd * 100,
        "profit_factor": profit_factor,
        "avg_signal": avg_signal,
        "tp_hits": tp_hits, "sl_hits": sl_hits, "closes": closes,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S5 backtest")
    parser.add_argument("--samples", default="/tmp/opencode/SBER_samples_pl6_sc5.parquet")
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--entry-threshold", type=float, default=0.55)
    parser.add_argument("--sl-pct", type=float, default=0.015)
    parser.add_argument("--tp-pct", type=float, default=0.03)
    args = parser.parse_args()

    df = load_samples(args.samples)
    print(f"Loaded {len(df)} windows from {args.samples}")

    trades = backtest(df, args.pred_len, args.sample_count,
                      entry_threshold=args.entry_threshold,
                      sl_pct=args.sl_pct, tp_pct=args.tp_pct)
    print(f"Trades: {len(trades)}")

    r = report(trades)
    print("\n=== S5 Backtest Report ===")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out = Path(args.samples).parent / "trades_s5.parquet"
    trades.to_parquet(out, index=False)
    print(f"\nTrade log saved: {out}")
