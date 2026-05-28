"""
TEMPLATE: Test fractal strategies 17-19 with hyperparameter sweeps.
Source: kronos-artifact/alpha/experiments/test_fractals.py
Purpose: Reference example for fractal breakout/cluster/AO strategy sweep
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples
from src.strategies import (load_config, load_mamba_sber, report,
                           quarterly_breakdown, run_strategy_17, run_strategy_18,
                           run_strategy_19)

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

RESULTS = []

def run_and_record(label, func):
    tr = func(train_df)
    te = func(test_df)
    r_tr = report(tr) if len(tr) > 0 else {}
    r_te = report(te) if len(te) > 0 else {}
    RESULTS.append({
        "label": label,
        "n_tr": r_tr.get("n", 0), "n_te": r_te.get("n", 0),
        "sharpe_tr": r_tr.get("sharpe", 0), "sharpe_te": r_te.get("sharpe", 0),
        "ret_tr": r_tr.get("total_return_pct", 0),
        "ret_te": r_te.get("total_return_pct", 0),
        "mdd_tr": r_tr.get("mdd_pct", 0), "mdd_te": r_te.get("mdd_pct", 0),
        "wr_tr": r_tr.get("win_rate", 0), "wr_te": r_te.get("win_rate", 0),
        "pf_tr": r_tr.get("profit_factor", 0), "pf_te": r_te.get("profit_factor", 0),
        "tp_rate_tr": r_tr.get("tp_rate", 0), "tp_rate_te": r_te.get("tp_rate", 0),
    })
    print(f"  {label}: Tr={r_tr.get('n',0)}t Sh={r_tr.get('sharpe',0):.2f} Ret={r_tr.get('total_return_pct',0):+.1f}%"
          f"  Te={r_te.get('n',0)}t Sh={r_te.get('sharpe',0):.2f} Ret={r_te.get('total_return_pct',0):+.1f}%")


# --- Strategy 17: Fractal Breakout ---
print("=== 17. Fractal Breakout ===")
for age in [6, 12, 24]:
    run_and_record(f"Breakout age={age}",
        lambda d, a=age: run_strategy_17(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, fractal_max_age=a))
print()

# --- Strategy 18: Fractal Cluster ---
print("=== 18. Fractal Cluster ===")
for age in [24, 48, 96]:
    for tol in [0.001, 0.002, 0.005]:
        run_and_record(f"Cluster age={age} tol={tol}",
            lambda d, a=age, t=tol: run_strategy_18(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, cluster_max_age=a, cluster_tolerance=t))
print()

# --- Strategy 19: Fractal + AO ---
print("=== 19. Fractal + AO ===")
for age in [6, 12, 24]:
    run_and_record(f"AO+Fractal age={age}",
        lambda d, a=age: run_strategy_19(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, fractal_max_age=a))
print()

# --- Summary Table ---
print("\n" + "=" * 130)
print(f"{'Strategy':<30} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'Ret(T)':>7} {'Ret(Te)':>7} {'MDD(T)':>7} {'MDD(Te)':>7} {'WR(T)':>6} {'WR(Te)':>6}")
print("=" * 130)
for r in RESULTS:
    print(f"{r['label']:<30} {r['n_tr']:>5} {r['n_te']:>5} {r['sharpe_tr']:>6.2f} {r['sharpe_te']:>6.2f} {r['ret_tr']:>+6.1f} {r['ret_te']:>+6.1f} {r['mdd_tr']:>6.1f} {r['mdd_te']:>6.1f} {r['wr_tr']:>5.1%} {r['wr_te']:>5.1%}")

# Best of each
print("\n" + "=" * 130)
print("BEST OF EACH FAMILY:")
print("=" * 130)
for family, label_base in [(17, "Breakout"), (18, "Cluster"), (19, "AO+Fractal")]:
    subset = [r for r in RESULTS if label_base in r["label"]]
    best = max(subset, key=lambda r: r["sharpe_te"] * r["n_te"]**0.3)  # sharpe x sqrt(n)
    print(f"  Best {family}: {best['label']} — Tr={best['n_tr']}t Sh={best['sharpe_tr']:.2f}  Te={best['n_te']}t Sh={best['sharpe_te']:.2f} Ret={best['ret_te']:+.1f}%")

print("\n=== REFERENCE (from earlier runs) ===")
print("  #2  BB Mean Rev:     919/355t  Sh T1.75 Te2.21  Ret +24.3%/+5.8%")
print("  #5  BB Narrow Mean:  718/154t  Sh T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  #8  %B Reversion:    435/157t  Sh T2.20 Te3.66  Ret +15.7%/+4.5%")
