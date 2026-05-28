"""
TEMPLATE: Sweep consensus_threshold [0.0, 0.6, 0.8, 1.0] on top-7 strategies.
Source: kronos-artifact/alpha/experiments/sweep_consensus.py
Purpose: Reference example for consensus threshold hyperparameter sweep
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
                        run_strategy_1, run_strategy_2, run_strategy_5,
                        run_strategy_20, run_strategy_28,
                        run_strategy_34, run_strategy_38)

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
print(f"Mamba: {len(mamba)} | Train: {len(train_df)} | Test: {len(test_df)}\n")

THRESHOLDS = [0.0, 0.6, 0.8, 1.0]

strategies = {
    1:  ("S1 BB+Consensus", lambda d, ct: run_strategy_1(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
    2:  ("S2 BB Mean Rev", lambda d, ct: run_strategy_2(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
    5:  ("S5 BB Narrow", lambda d, ct: run_strategy_5(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
    20: ("S20 OB", lambda d, ct: run_strategy_20(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
    28: ("S28 VolOB", lambda d, ct: run_strategy_28(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
    34: ("S34 VWAP+OB", lambda d, ct: run_strategy_34(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
    38: ("S38 Low-vol OB", lambda d, ct: run_strategy_38(
        d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, consensus_threshold=ct)),
}

results = []

for sid, (sname, func) in strategies.items():
    print(f"Sweeping {sname}...")
    for ct in THRESHOLDS:
        tr = func(train_df, ct)
        r = report(tr) if len(tr) > 0 else {"error": "no trades", "n": 0}
        if r.get("error"):
            print(f"  ct={ct}: 0 trades")
            results.append({
                "strategy": sid, "name": sname, "consensus_threshold": ct,
                "n_tr": 0, "wr_tr": 0, "ret_tr": 0, "sharpe_tr": 0,
                "mdd_tr": 0, "pf_tr": 0,
                "n_te": 0, "wr_te": 0, "ret_te": 0, "sharpe_te": 0,
                "mdd_te": 0, "pf_te": 0,
            })
            continue

        te = func(test_df, ct)
        r_te = report(te) if len(te) > 0 else {"error": "no trades", "n": 0}

        results.append({
            "strategy": sid, "name": sname, "consensus_threshold": ct,
            "n_tr": r["n"], "wr_tr": r["win_rate"],
            "ret_tr": r["total_return_pct"], "sharpe_tr": r["sharpe"],
            "mdd_tr": r["mdd_pct"], "pf_tr": r["profit_factor"],
            "n_te": r_te.get("n", 0), "wr_te": r_te.get("win_rate", 0),
            "ret_te": r_te.get("total_return_pct", 0),
            "sharpe_te": r_te.get("sharpe", 0),
            "mdd_te": r_te.get("mdd_pct", 0), "pf_te": r_te.get("profit_factor", 0),
        })
        print(f"  ct={ct}: Tr={r['n']}t Sh={r['sharpe']:.2f} Ret={r['total_return_pct']:+.2f}%  "
              f"Te={r_te.get('n',0)}t Sh={r_te.get('sharpe',0):.2f} Ret={r_te.get('total_return_pct',0):+.2f}%")

print("\n" + "=" * 140)
print(f"{'#':>2} {'Strategy':<20} {'ct':>4} {'Trn':>5} {'TrWR':>6} {'TrRet%':>8} {'TrSh':>7} {'TrMDD%':>7} {'TrPF':>6} | {'TeN':>5} {'TeWR':>6} {'TeRet%':>8} {'TeSh':>7} {'TeMDD%':>7} {'TePF':>6}")
print("=" * 140)

for r in sorted(results, key=lambda x: (x["strategy"], x["consensus_threshold"])):
    print(f"{r['strategy']:>2} {r['name']:<20} {r['consensus_threshold']:>3.1f} "
          f"{r['n_tr']:>5} {r['wr_tr']:>5.2f} {r['ret_tr']:>+7.2f} {r['sharpe_tr']:>6.2f} {r['mdd_tr']:>+6.2f} {r['pf_tr']:>5.2f} | "
          f"{r['n_te']:>5} {r['wr_te']:>5.2f} {r['ret_te']:>+7.2f} {r['sharpe_te']:>6.2f} {r['mdd_te']:>+6.2f} {r['pf_te']:>5.2f}")

print("\n" + "=" * 140)
print("BEST BY TRAIN SHARPE PER STRATEGY")
print("=" * 140)
for sid in [1, 2, 5, 20, 28, 34, 38]:
    sr = [r for r in results if r["strategy"] == sid and r["n_tr"] > 0]
    if not sr:
        continue
    best = max(sr, key=lambda x: x["sharpe_tr"])
    print(f"  {sid}. {best['name']:<20} ct={best['consensus_threshold']:>3.1f}  "
          f"Tr S={best['sharpe_tr']:.2f} R={best['ret_tr']:+.2f}% n={best['n_tr']}  "
          f"Te S={best['sharpe_te']:.2f} R={best['ret_te']:+.2f}% n={best['n_te']}")
