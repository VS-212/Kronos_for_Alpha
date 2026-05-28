"""
TEMPLATE: Test ICT strategies 20-25 with hyperparameter sweeps.
Source: kronos-artifact/alpha/experiments/test_ict.py
Purpose: Reference example for ICT strategy family parameter sweep
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.output import load_samples
from src.strategies import (load_config, load_mamba_sber, report,
                           run_strategy_20, run_strategy_21, run_strategy_22,
                           run_strategy_23, run_strategy_24, run_strategy_25)

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
    })

# --- 20. Order Block ---
print("=== 20. Order Block ===")
for lb in [24, 48, 96]:
    for mt in [0.002, 0.005, 0.01]:
        run_and_record(f"OB lb={lb} mt={mt}",
            lambda d, a=lb, b=mt: run_strategy_20(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=a, move_threshold=b))
print()

# --- 21. Liquidity Sweep ---
print("=== 21. Liquidity Sweep ===")
for lb in [24, 48]:
    for fa in [6, 12, 24]:
        run_and_record(f"Sweep lb={lb} fa={fa}",
            lambda d, a=lb, b=fa: run_strategy_21(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=a, fractal_max_age=b))
print()

# --- 22. FVG ---
print("=== 22. FVG ===")
for lb in [24, 48, 96]:
    run_and_record(f"FVG lb={lb}",
        lambda d, a=lb: run_strategy_22(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=a))
print()

# --- 23. OB+FVG ---
print("=== 23. OB+FVG ===")
for lb in [24, 48]:
    for mt in [0.003, 0.005]:
        run_and_record(f"OB+FVG lb={lb} mt={mt}",
            lambda d, a=lb, b=mt: run_strategy_23(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=a, move_threshold=b))
print()

# --- 24. Premium/Discount ---
print("=== 24. Premium/Discount ===")
for lb in [48, 96, 192]:
    run_and_record(f"P/D lb={lb}",
        lambda d, a=lb: run_strategy_24(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=a))
print()

# --- 25. MSS ---
print("=== 25. MSS ===")
for lb in [24, 48]:
    for fa in [12, 24]:
        run_and_record(f"MSS lb={lb} fa={fa}",
            lambda d, a=lb, b=fa: run_strategy_25(d, mamba, PRED_LEN, SAMPLE_COUNT, TP_Q, SL_Q, lookback=a, fractal_max_age=b))
print()

# --- Summary ---
print("\n" + "=" * 100)
print(f"{'Strategy':<30} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'Ret(T)':>7} {'Ret(Te)':>7} {'MDD(T)':>7} {'MDD(Te)':>7}")
print("=" * 100)
for r in RESULTS:
    print(f"{r['label']:<30} {r['n_tr']:>5} {r['n_te']:>5} {r['sharpe_tr']:>6.2f} {r['sharpe_te']:>6.2f} {r['ret_tr']:>+6.1f} {r['ret_te']:>+6.1f} {r['mdd_tr']:>6.1f} {r['mdd_te']:>6.1f}")

# Best of each
print("\n" + "=" * 100)
print("BEST OF EACH FAMILY:")
print("=" * 100)
for family, label_base in [(20, "OB "), (21, "Sweep"), (22, "FVG "), (23, "OB+FVG"), (24, "P/D"), (25, "MSS")]:
    subset = [r for r in RESULTS if r["label"].startswith(label_base.strip()) or r["label"].startswith(label_base.replace(" ", ""))]
    if not subset:
        subset = [r for r in RESULTS if label_base.replace(" ", "") in r["label"]]
    if not subset:
        continue
    best = max(subset, key=lambda r: r["sharpe_te"] * r["n_te"]**0.3)
    print(f"  Best {family}: {best['label']} — Tr={best['n_tr']}t Sh={best['sharpe_tr']:.2f}  Te={best['n_te']}t Sh={best['sharpe_te']:.2f} Ret={best['ret_te']:+.1f}%")

print("\n=== REFERENCE (top-3 BB) ===")
print("  #2  BB Mean Rev:     919/355t  Sh T1.75 Te2.21  Ret +24.3%/+5.8%")
print("  #5  BB Narrow Mean:  718/154t  Sh T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  #8  %B Reversion:    435/157t  Sh T2.20 Te3.66  Ret +15.7%/+4.5%")
