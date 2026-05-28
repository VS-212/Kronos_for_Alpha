"""
TEMPLATE: Hyperparameter grid: tp_q x sl_q sweep for top 3 strategies.
Source: kronos-artifact/alpha/experiments/sweep_tpsl.py
Purpose: Reference example for TP/SL quantile grid search
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
import itertools
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples
from src.strategies import (load_config, load_mamba_sber, report,
                           run_strategy_2, run_strategy_5, run_strategy_8)

PRED_LEN = 6
SAMPLE_COUNT = 5

print("Loading data...")
cfg = load_config()
mamba = load_mamba_sber(cfg)
df = load_samples("/tmp/opencode/SBER_samples_pl6_sc5.parquet")
df["year"] = df["month"].str[:4]
train_df = df[df["year"] == "2025"]
test_df = df[df["year"] == "2026"]
print(f"Mamba: {len(mamba)} | Train: {len(train_df)} | Test: {len(test_df)}\n")

# Grid
tp_qs = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
sl_qs = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

strategies = {
    2: ("BB Mean Reversion", lambda d, tp, sl: run_strategy_2(d, mamba, PRED_LEN, SAMPLE_COUNT, tp, sl)),
    5: ("BB Narrow Mean Rev", lambda d, tp, sl: run_strategy_5(d, mamba, PRED_LEN, SAMPLE_COUNT, tp, sl)),
    8: ("%B Reversion", lambda d, tp, sl: run_strategy_8(d, mamba, PRED_LEN, SAMPLE_COUNT, tp, sl)),
}

results = []

for sid, (sname, func) in strategies.items():
    print(f"Sweeping {sname}...")
    for tp_q, sl_q in itertools.product(tp_qs, sl_qs):
        if tp_q <= sl_q:
            continue
        tr = func(train_df, tp_q, sl_q)
        r = report(tr) if len(tr) > 0 else {"error": "no trades", "n": 0}
        if r.get("error"):
            continue
        results.append({
            "strategy": sid, "name": sname,
            "tp_q": tp_q, "sl_q": sl_q,
            "n": r["n"], "win_rate": r["win_rate"],
            "total_ret": r["total_return_pct"],
            "avg_ret": r["avg_return_pct"],
            "sharpe": r["sharpe"], "mdd": r["mdd_pct"],
            "pf": r["profit_factor"],
        })

print("\n" + "=" * 120)
print(f"{'#':>2} {'Strategy':<22} {'tp_q':>5} {'sl_q':>5} {'n':>5} {'WR':>6} {'Ret%':>8} {'AvgRet%':>9} {'Sharpe':>8} {'MDD%':>8} {'PF':>6}")
print("=" * 120)

# Best by Sharpe for each strategy
for sid in [2, 5, 8]:
    strat_results = [r for r in results if r["strategy"] == sid]
    best = max(strat_results, key=lambda x: x["sharpe"])
    sname = best["name"]

    print(f"\n  Best for {sid}. {sname}: tp_q={best['tp_q']}, sl_q={best['sl_q']} "
          f"Sharpe={best['sharpe']:.2f}, Ret={best['total_ret']:+.2f}%, n={best['n']}")
    print(f"  {'tp_q':>5} {'sl_q':>5} {'n':>5} {'WR':>6} {'Ret%':>8} {'AvgRet%':>9} {'Sharpe':>8} {'MDD%':>8} {'PF':>6}")
    print(f"  {'-'*55}")
    for r in sorted(strat_results, key=lambda x: -x["sharpe"])[:10]:
        print(f"  {r['tp_q']:>4.2f} {r['sl_q']:>4.2f} {r['n']:>5} {r['win_rate']:>5.2f} "
              f"{r['total_ret']:>+7.2f} {r['avg_ret']:>+8.4f} {r['sharpe']:>7.2f} "
              f"{r['mdd']:>+7.2f} {r['pf']:>5.2f}")

# Test the best config for each strategy
print("\n" + "=" * 70)
print("TEST SET — best config per strategy")
print("=" * 70)

for sid in [2, 5, 8]:
    strat_results = [r for r in results if r["strategy"] == sid]
    best = max(strat_results, key=lambda x: x["sharpe"])
    func = strategies[sid][1]
    te = func(test_df, best["tp_q"], best["sl_q"])
    r_te = report(te) if len(te) > 0 else {"error": "no trades", "n": 0}
    if not r_te.get("error"):
        print(f"  {sid}. {strategies[sid][0]:<22} "
              f"tp={best['tp_q']:.2f} sl={best['sl_q']:.2f}  "
              f"n={r_te['n']:>4}  Sharpe={r_te['sharpe']:.2f}  "
              f"Ret={r_te['total_return_pct']:+.2f}%  "
              f"MDD={r_te['mdd_pct']:.2f}%")
