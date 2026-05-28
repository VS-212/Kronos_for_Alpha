"""
TEMPLATE: S13 Consensus-Bounded Directional Entry strategy (standalone).
Source: kronos-artifact/alpha/experiments/s13.py
Purpose: Reference example for consensus-boundary-based strategy with quarterly breakdown
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples, reconstruct
from src.signals.atoms import consensus, boundaries


def backtest(df: pd.DataFrame, pred_len: int, sample_count: int,
             tp_q: float = 0.80, sl_q: float = 0.20,
             entry_threshold: float = 0.8) -> pd.DataFrame:
    trades = []
    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        cons = consensus(close_only, prev_close, threshold=entry_threshold)
        if not cons["has_consensus"][0]:
            continue
        pred_dir = int(cons["consensus_dir"][0])

        entry_price = float(actuals[0, 0])
        bnd = boundaries(close_only, prev_close, tp_q=tp_q, sl_q=sl_q)

        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])

        for i in range(pred_len):
            hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
            if pred_dir == 1:
                if lo <= bnd["sl_long"][i]:
                    exit_step = i; exit_reason = "sl"; exit_price = float(bnd["sl_long"][i]); break
                if hi >= bnd["tp_long"][i]:
                    exit_step = i; exit_reason = "tp"; exit_price = float(bnd["tp_long"][i]); break
            else:
                if hi >= bnd["sl_short"][i]:
                    exit_step = i; exit_reason = "sl"; exit_price = float(bnd["sl_short"][i]); break
                if lo <= bnd["tp_short"][i]:
                    exit_step = i; exit_reason = "tp"; exit_price = float(bnd["tp_short"][i]); break

        trade_return = pred_dir * (exit_price - entry_price) / entry_price
        trades.append({
            "window_id": int(row["window_id"]),
            "pred_ts": row["pred_ts"],
            "month": row["month"],
            "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:])-1)//3+1}",
            "direction": pred_dir,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_step": exit_step,
            "exit_reason": exit_reason,
            "return": trade_return,
        })
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"error": "no trades", "n": 0}
    n = len(trades)
    tp_hits = int((trades["exit_reason"] == "tp").sum())
    sl_hits = int((trades["exit_reason"] == "sl").sum())
    closes = int((trades["exit_reason"] == "close").sum())
    wins = int((trades["return"] > 0).sum())
    losses = int((trades["return"] <= 0).sum())
    total_ret = trades["return"].sum()
    avg_ret = trades["return"].mean()
    std_ret = trades["return"].std()
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
    pf = sum_wins / max(sum_losses, 1e-10)
    return {
        "n": n, "tp_hits": tp_hits, "sl_hits": sl_hits, "closes": closes,
        "tp_rate": tp_hits / max(n, 1), "sl_rate": sl_hits / max(n, 1),
        "close_rate": closes / max(n, 1),
        "wins": wins, "losses": losses, "win_rate": win_rate,
        "total_return_pct": total_ret * 100, "avg_return_pct": avg_ret * 100,
        "std_return_pct": std_ret * 100, "sharpe": sharpe,
        "mdd_pct": mdd * 100, "profit_factor": pf,
        "avg_win_pct": avg_win * 100, "avg_loss_pct": avg_loss * 100,
    }


def quarterly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, grp in trades.groupby("quarter", sort=True):
        r = report(grp)
        r["quarter"] = q
        rows.append(r)
    return pd.DataFrame(rows).set_index("quarter")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S13 backtest")
    parser.add_argument("--samples", default="/tmp/opencode/SBER_samples_pl6_sc5.parquet")
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--tp-q", type=float, default=0.80)
    parser.add_argument("--sl-q", type=float, default=0.20)
    parser.add_argument("--entry-threshold", type=float, default=0.8)
    args = parser.parse_args()

    df = load_samples(args.samples)
    df["year"] = df["month"].str[:4]

    # Split: 2025 = train, 2026 = test
    train_df = df[df["year"] == "2025"]
    test_df = df[df["year"] == "2026"]

    print(f"Train (2025): {len(train_df)} windows | Test (2026): {len(test_df)} windows")

    trades_train = backtest(train_df, args.pred_len, args.sample_count,
                           tp_q=args.tp_q, sl_q=args.sl_q,
                           entry_threshold=args.entry_threshold)
    trades_test = backtest(test_df, args.pred_len, args.sample_count,
                          tp_q=args.tp_q, sl_q=args.sl_q,
                          entry_threshold=args.entry_threshold)

    r_train = report(trades_train)
    r_test = report(trades_test)

    print(f"\n=== S13 | tp_q={args.tp_q} sl_q={args.sl_q} ===")
    print(f"{'Metric':<20} {'Train 2025':>12} {'Test 2026':>12}")
    print("-" * 46)
    for k in ["n", "win_rate", "total_return_pct", "sharpe", "mdd_pct", "profit_factor",
              "tp_rate", "sl_rate", "close_rate", "avg_win_pct", "avg_loss_pct"]:
        v1 = r_train.get(k, 0)
        v2 = r_test.get(k, 0)
        if k == "n":
            print(f"{k:<20} {v1:>12} {v2:>12}")
        else:
            print(f"{k:<20} {v1:>11.4f}% {v2:>11.4f}%")

    print("\n=== Quarterly Breakdown (Train 2025) ===")
    qb_train = quarterly_breakdown(trades_train)
    cols = ["n", "win_rate", "total_return_pct", "sharpe", "mdd_pct", "profit_factor"]
    print(qb_train[cols].to_string(float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    if len(trades_test) > 0:
        print("\n=== Quarterly Breakdown (Test 2026) ===")
        qb_test = quarterly_breakdown(trades_test)
        print(qb_test[cols].to_string(float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    # Save
    out_dir = Path(args.samples).parent
    tag = f"tp{str(args.tp_q).replace('.','')}_sl{str(args.sl_q).replace('.','')}"
    trades_train.to_parquet(out_dir / f"trades_s13_train_{tag}.parquet", index=False)
    trades_test.to_parquet(out_dir / f"trades_s13_test_{tag}.parquet", index=False)
    print(f"\nSaved: trades_s13_train_{tag}.parquet + test")
