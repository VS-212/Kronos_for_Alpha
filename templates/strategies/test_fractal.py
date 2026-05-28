"""
TEMPLATE: Test fractal strategies 15 and 16 independently.
Source: kronos-artifact/alpha/experiments/test_fractal.py
Purpose: Reference example for fractal-based signal strategy testing
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
import yaml

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples
from src.strategies import (load_config, load_mamba_sber, report,
                           quarterly_breakdown, run_strategy_15, run_strategy_16)

PRED_LEN = 6
SAMPLE_COUNT = 5
TP_Q = 0.90
SL_Q = 0.10

print("Loading data...")
cfg = load_config()
mamba = load_mamba_sber(cfg)
print(f"Mamba: {len(mamba)} rows, {mamba.index.min()} -> {mamba.index.max()}")

df = load_samples("/tmp/opencode/SBER_samples_pl6_sc5.parquet")
df["year"] = df["month"].str[:4]
train_df = df[df["year"] == "2025"]
test_df = df[df["year"] == "2026"]
print(f"Samples: train {len(train_df)} | test {len(test_df)}\n")

strategies = [
    ("15. Fractal Signal (age=8)", lambda d: run_strategy_15(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, fractal_max_age=8)),
    ("15. Fractal Signal (age=12)", lambda d: run_strategy_15(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, fractal_max_age=12)),
    ("15. Fractal Signal (age=16)", lambda d: run_strategy_15(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, fractal_max_age=16)),
    ("16. Fractal + BB (age=12)", lambda d: run_strategy_16(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, fractal_max_age=12)),
]

cols_q = ["n", "win_rate", "total_return_pct", "avg_return_pct",
          "avg_win_pct", "avg_loss_pct", "sharpe", "mdd_pct", "profit_factor",
          "tp_rate", "sl_rate", "close_rate"]

for name, func in strategies:
    tr = func(train_df)
    te = func(test_df) if len(test_df) > 0 else pd.DataFrame()
    r_tr = report(tr) if len(tr) > 0 else {}
    r_te = report(te) if len(te) > 0 else {}

    print(f"=== {name} ===")
    if not r_tr.get("error"):
        print(f"  Train: {r_tr['n']} trades")
        print(f"    WinRate={r_tr['win_rate']:.4f}  TotalRet={r_tr['total_return_pct']:+.4f}%  "
              f"AvgRet={r_tr['avg_return_pct']:+.6f}%")
        print(f"    Sharpe={r_tr['sharpe']:.4f}  MDD={r_tr['mdd_pct']:.4f}%  "
              f"PF={r_tr['profit_factor']:.4f}")
        print(f"    TP={r_tr['tp_hits']}({r_tr['tp_rate']*100:.1f}%)  "
              f"SL={r_tr['sl_hits']}({r_tr['sl_rate']*100:.1f}%)  "
              f"Close={r_tr['closes']}({r_tr['close_rate']*100:.1f}%)")
        qb = quarterly_breakdown(tr)
        print(qb[cols_q].to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    if not r_te.get("error") and len(te) > 0:
        print(f"\n  Test: {r_te['n']} trades")
        print(f"    WinRate={r_te['win_rate']:.4f}  TotalRet={r_te['total_return_pct']:+.4f}%  "
              f"AvgRet={r_te['avg_return_pct']:+.6f}%  Sharpe={r_te['sharpe']:.4f}  "
              f"MDD={r_te['mdd_pct']:.4f}%")
        qb_te = quarterly_breakdown(te)
        print(qb_te[cols_q].to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print()
