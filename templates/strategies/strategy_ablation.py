"""
TEMPLATE: Pure BB ablation — NO Kronos, entry based only on Bollinger Band position.
Source: kronos-artifact/alpha/experiments/bb_pure.py
Purpose: Reference example for ablation study: BB alone vs Kronos+BB combo
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples, reconstruct


def load_config() -> dict:
    with open(Path(__file__).resolve().parent.parent / "config" / "global.yaml") as f:
        return yaml.safe_load(f)


def load_mamba_sber(cfg: dict) -> pd.DataFrame:
    mamba_path = cfg["data"]["mamba_path"]
    raw = pd.read_parquet(mamba_path)
    ts = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index(ts)
    cols = [f"SBER_{c}" for c in ["open", "high", "low", "close", "volume"]]
    df = raw[cols].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    df = df.between_time("10:00", "18:40")
    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std(ddof=1)
    df["bb_upper"] = df["sma20"] + 2 * df["std20"]
    df["bb_lower"] = df["sma20"] - 2 * df["std20"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma20"]
    return df


def lookup_bb(mamba: pd.DataFrame, pred_ts: pd.Timestamp) -> dict:
    mask = mamba.index < pred_ts
    if mask.sum() == 0:
        return {"sma20": np.nan, "bb_upper": np.nan, "bb_lower": np.nan, "bb_width": np.nan, "last_close": np.nan}
    row = mamba[mask].iloc[-1]
    return {k: float(row[k]) for k in ["sma20", "bb_upper", "bb_lower", "bb_width", "close"]}


def run_strategy_1_pure(df, mamba, pred_len, tp_pct=0.003, sl_pct=0.002):
    """Pure BB Filter: price outside BB -> enter in band direction. Fixed TP/SL."""
    trades = []
    for _, row in df.iterrows():
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue
        bb = lookup_bb(mamba, row["pred_ts"])
        if np.isnan(bb["bb_upper"]):
            continue
        entry_price = float(actuals[0, 0])
        if entry_price >= bb["bb_upper"]:
            pred_dir = -1
        elif entry_price <= bb["bb_lower"]:
            pred_dir = 1
        else:
            continue
        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])
        for i in range(pred_len):
            hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
            if pred_dir == 1:
                sl_p = entry_price * (1 - sl_pct)
                tp_p = entry_price * (1 + tp_pct)
                if lo <= sl_p:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_p; break
                if hi >= tp_p:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_p; break
            else:
                sl_p = entry_price * (1 + sl_pct)
                tp_p = entry_price * (1 - tp_pct)
                if hi >= sl_p:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_p; break
                if lo <= tp_p:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_p; break
        trade_return = pred_dir * (exit_price - entry_price) / entry_price
        trades.append({"window_id": int(row["window_id"]), "month": row["month"],
            "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:])-1)//3+1}",
            "direction": pred_dir, "entry_price": entry_price, "exit_price": exit_price,
            "exit_step": exit_step, "exit_reason": exit_reason, "return": trade_return})
    return pd.DataFrame(trades)


def run_strategy_2_pure(df, mamba, pred_len, bb_zone=0.5, tp_pct=0.003, sl_pct=0.002):
    """Pure BB Mean Rev: near band extreme -> enter in REVERSAL direction."""
    trades = []
    for _, row in df.iterrows():
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue
        bb = lookup_bb(mamba, row["pred_ts"])
        if np.isnan(bb["bb_upper"]) or bb["bb_width"] == 0:
            continue
        entry_price = float(actuals[0, 0])
        zone_dist = bb_zone * (bb["bb_upper"] - bb["bb_lower"])
        near_lower = entry_price <= bb["bb_lower"] + zone_dist
        near_upper = entry_price >= bb["bb_upper"] - zone_dist
        if near_lower:
            pred_dir = 1  # mean reversion up from lower band
        elif near_upper:
            pred_dir = -1  # mean reversion down from upper band
        else:
            continue
        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])
        for i in range(pred_len):
            hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
            if pred_dir == 1:
                sl_p = entry_price * (1 - sl_pct)
                tp_p = entry_price * (1 + tp_pct)
                if lo <= sl_p:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_p; break
                if hi >= tp_p:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_p; break
            else:
                sl_p = entry_price * (1 + sl_pct)
                tp_p = entry_price * (1 - tp_pct)
                if hi >= sl_p:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_p; break
                if lo <= tp_p:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_p; break
        trade_return = pred_dir * (exit_price - entry_price) / entry_price
        trades.append({"window_id": int(row["window_id"]), "month": row["month"],
            "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:])-1)//3+1}",
            "direction": pred_dir, "entry_price": entry_price, "exit_price": exit_price,
            "exit_step": exit_step, "exit_reason": exit_reason, "return": trade_return})
    return pd.DataFrame(trades)


def run_strategy_3_pure(df, mamba, pred_len, min_bb_width=0.005, tp_pct=0.003, sl_pct=0.002):
    """Pure BB Width + SMA20 direction: close > SMA20 -> long, < SMA20 -> short."""
    trades = []
    for _, row in df.iterrows():
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue
        bb = lookup_bb(mamba, row["pred_ts"])
        if np.isnan(bb["bb_width"]) or bb["bb_width"] < min_bb_width:
            continue
        entry_price = float(actuals[0, 0])
        if entry_price >= bb["sma20"]:
            pred_dir = 1
        else:
            pred_dir = -1
        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])
        for i in range(pred_len):
            hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
            if pred_dir == 1:
                sl_p = entry_price * (1 - sl_pct)
                tp_p = entry_price * (1 + tp_pct)
                if lo <= sl_p:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_p; break
                if hi >= tp_p:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_p; break
            else:
                sl_p = entry_price * (1 + sl_pct)
                tp_p = entry_price * (1 - tp_pct)
                if hi >= sl_p:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_p; break
                if lo <= tp_p:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_p; break
        trade_return = pred_dir * (exit_price - entry_price) / entry_price
        trades.append({"window_id": int(row["window_id"]), "month": row["month"],
            "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:])-1)//3+1}",
            "direction": pred_dir, "entry_price": entry_price, "exit_price": exit_price,
            "exit_step": exit_step, "exit_reason": exit_reason, "return": trade_return})
    return pd.DataFrame(trades)


def run_strategy_4_pure(df, mamba, pred_len):
    """Pure BB TP/SL: enter when price breaks outside BB, exit at opposite band."""
    trades = []
    for _, row in df.iterrows():
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue
        bb = lookup_bb(mamba, row["pred_ts"])
        if np.isnan(bb["bb_upper"]) or np.isnan(bb["bb_lower"]):
            continue
        entry_price = float(actuals[0, 0])
        if entry_price >= bb["bb_upper"]:
            pred_dir = -1
            tp_price = bb["bb_lower"]
            sl_price = bb["bb_upper"] + (bb["bb_upper"] - bb["bb_lower"])
        elif entry_price <= bb["bb_lower"]:
            pred_dir = 1
            tp_price = bb["bb_upper"]
            sl_price = bb["bb_lower"] - (bb["bb_upper"] - bb["bb_lower"])
        else:
            continue
        exit_step = pred_len - 1
        exit_reason = "close"
        exit_price = float(actuals[-1, 3])
        for i in range(pred_len):
            hi, lo = float(actuals[i, 1]), float(actuals[i, 2])
            if pred_dir == 1:
                if lo <= sl_price:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_price; break
                if hi >= tp_price:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_price; break
            else:
                if hi >= sl_price:
                    exit_step = i; exit_reason = "sl"; exit_price = sl_price; break
                if lo <= tp_price:
                    exit_step = i; exit_reason = "tp"; exit_price = tp_price; break
        trade_return = pred_dir * (exit_price - entry_price) / entry_price
        trades.append({"window_id": int(row["window_id"]), "month": row["month"],
            "quarter": f"{row['month'][:4]}-Q{(int(row['month'][5:])-1)//3+1}",
            "direction": pred_dir, "entry_price": entry_price, "exit_price": exit_price,
            "exit_step": exit_step, "exit_reason": exit_reason, "return": trade_return})
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
    sharpe = avg_ret / max(std_ret, 1e-10) * np.sqrt(252)
    win_rate = wins / max(n, 1)
    avg_win = float(trades.loc[trades["return"] > 0, "return"].mean()) if wins else 0.0
    avg_loss = float(trades.loc[trades["return"] <= 0, "return"].mean()) if losses else 0.0
    sum_wins = avg_win * wins if wins else 0.0
    sum_losses = abs(avg_loss * losses) if losses else 0.0
    pf = sum_wins / max(sum_losses, 1e-10)
    return {"n": n, "wins": wins, "losses": losses, "win_rate": win_rate,
        "total_return_pct": total_ret * 100, "sharpe": sharpe, "mdd_pct": mdd * 100,
        "profit_factor": pf, "tp_hits": tp_hits, "sl_hits": sl_hits, "closes": closes,
        "avg_win_pct": avg_win * 100, "avg_loss_pct": avg_loss * 100,
        "avg_return_pct": avg_ret * 100}


def quarterly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, grp in trades.groupby("quarter", sort=True):
        r = report(grp); r["quarter"] = q; rows.append(r)
    return pd.DataFrame(rows).set_index("quarter")


if __name__ == "__main__":
    cfg = load_config()
    print("Loading Mamba + BB ...")
    mamba = load_mamba_sber(cfg)
    print(f"Mamba: {len(mamba)} rows, {mamba.index.min()} -> {mamba.index.max()}")

    df = load_samples("/tmp/opencode/SBER_samples_pl6_sc5.parquet")
    df["year"] = df["month"].str[:4]
    train_df = df[df["year"] == "2025"]
    test_df = df[df["year"] == "2026"]
    print(f"Samples: train {len(train_df)} | test {len(test_df)}\n")

    strategies = [
        ("1. Pure BB Filter", run_strategy_1_pure),
        ("2. Pure BB Mean Rev", run_strategy_2_pure),
        ("3. Pure BB Width+SMA", run_strategy_3_pure),
        ("4. Pure BB TP/SL bands", run_strategy_4_pure),
    ]

    for name, func in strategies:
        tr = func(train_df, mamba, 6)
        te = func(test_df, mamba, 6)
        r_tr = report(tr)
        r_te = report(te)

        print(f"=== {name} ===")
        if r_tr.get("error"):
            print("  Train: 0 trades")
        else:
            print(f"  Train: {r_tr['n']} trades, WR {r_tr['win_rate']:.3f}, "
                  f"Ret {r_tr['total_return_pct']:+.2f}%, Sharpe {r_tr['sharpe']:.3f}, "
                  f"MDD {r_tr['mdd_pct']:.2f}%, PF {r_tr['profit_factor']:.3f}")
            qb = quarterly_breakdown(tr)
            qb_str = " | ".join(f"{q}: R={r['total_return_pct']:+.1f}%" for q, r in qb.iterrows())
            print(f"  Quarters: {qb_str}")
        if r_te.get("error"):
            print("  Test: 0 trades")
        elif len(te) > 0:
            print(f"  Test: {r_te['n']} trades, WR {r_te['win_rate']:.3f}, "
                  f"Ret {r_te['total_return_pct']:+.2f}%, Sharpe {r_te['sharpe']:.3f}, "
                  f"MDD {r_te['mdd_pct']:.2f}%")
        print()
