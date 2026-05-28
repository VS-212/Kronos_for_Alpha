"""
TEMPLATE: Hour distribution analysis for top 7 strategies.
Source: kronos-artifact/alpha/experiments/test_hour_dist.py
Purpose: Reference example for intraday hour distribution analysis across strategies
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.evaluation.output import load_samples
from src.strategies import (load_config, load_mamba_sber, report,
    run_strategy_20, run_strategy_28, run_strategy_34, run_strategy_38,
    run_strategy_2, run_strategy_5, run_strategy_8)

PRED_LEN = 6
SAMPLE_COUNT = 5
TP_Q = 0.90
SL_Q = 0.10

print("Loading data...")
cfg = load_config()
mamba = load_mamba_sber(cfg)

df = load_samples("/tmp/opencode/SBER_samples_pl6_sc5.parquet")
df["year"] = df["month"].str[:4]
train_df = df[df["year"] == "2025"]
test_df = df[df["year"] == "2026"]
print(f"Samples: train {len(train_df)} | test {len(test_df)}\n")

# Top 7 strategies
strategies = [
    ("28. VolOB vm=2", lambda d: run_strategy_28(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=48, move_threshold=0.002, volume_mult=2.0)),
    ("38. Low-vol OB ap=0.3", lambda d: run_strategy_38(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, adr_lookback=800, ob_lookback=48, adr_period=20, adr_pct=0.3)),
    ("20. OB mt=2 ma=12", lambda d: run_strategy_20(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=48, move_threshold=0.002, max_age=12)),
    ("34. VWAP+OB", lambda d: run_strategy_34(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, k=1.0, lookback=96)),
    ("20. OB mt=2 ma=24", lambda d: run_strategy_20(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=48, move_threshold=0.002, max_age=24)),
    ("5. BB Narrow", lambda d: run_strategy_5(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q)),
    ("8. %B Reversion", lambda d: run_strategy_8(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q)),
]

def hour_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-hour metrics from a trades DataFrame."""
    if len(trades) == 0:
        return pd.DataFrame()
    trades = trades.copy()
    trades["hour"] = trades["pred_ts"].dt.hour
    rows = []
    for h, grp in trades.groupby("hour", sort=True):
        n = len(grp)
        wins = int((grp["return"] > 0).sum())
        losses = n - wins
        win_rate = wins / max(n, 1)
        total_ret = grp["return"].sum()
        avg_ret = grp["return"].mean()
        std_ret = grp["return"].std()
        sharpe = avg_ret / max(std_ret, 1e-10) * np.sqrt(252 * 6 / 6)
        tp_hits = int((grp["exit_reason"] == "tp").sum())
        sl_hits = int((grp["exit_reason"] == "sl").sum())
        closes = n - tp_hits - sl_hits
        sum_wins = float(grp.loc[grp["return"] > 0, "return"].sum()) if wins else 0.0
        sum_losses = abs(float(grp.loc[grp["return"] <= 0, "return"].sum())) if losses else 0.0
        pf = sum_wins / max(sum_losses, 1e-10)
        trade_pct = n / max(len(trades), 1) * 100
        rows.append({
            "hour": h, "n": n, "trade%": trade_pct,
            "win_rate": win_rate, "avg_ret_pct": avg_ret * 100,
            "sharpe": sharpe, "pf": pf,
            "tp_rate": tp_hits / max(n, 1), "sl_rate": sl_hits / max(n, 1),
            "close_rate": closes / max(n, 1),
        })
    return pd.DataFrame(rows).set_index("hour")

print("=" * 130)
print(f"{'Strategy':<25} {'Set':>4} {'Hour':>5} {'n':>5} {'Trade%':>7} {'WinRate':>8} {'AvgRet%':>9} {'Sharpe':>7} {'PF':>6} {'TP%':>6} {'SL%':>6} {'Close%':>7}")
print("=" * 130)

for name, func in strategies:
    for label, data in [("Tr", train_df), ("Te", test_df)]:
        tr = func(data)
        if len(tr) == 0:
            continue
        hb = hour_breakdown(tr)
        for h, r in hb.iterrows():
            print(f"{name:<25} {label:>4} {h:>5} {r['n']:>5} {r['trade%']:>6.1f}% "
                  f"{r['win_rate']:>7.2f} {r['avg_ret_pct']:>+8.4f} {r['sharpe']:>6.2f} "
                  f"{r['pf']:>5.2f} {r['tp_rate']*100:>5.1f}% {r['sl_rate']*100:>5.1f}% {r['close_rate']*100:>6.1f}%")
    print()
