"""
TEMPLATE: Sweep magnitude_threshold for strategy 43 (Authors' magnitude) + compare vs TOP baselines.
Source: kronos-artifact/alpha/experiments/sweep_authors.py
Purpose: Reference example for parallel strategy sweep with baselines
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from utils.parallel import run_parallel

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SAMPLES_PATH = "/tmp/opencode/SBER_samples_pl6_sc5.parquet"
P = {"pred_len": 6, "sample_count": 5, "tp_q": 0.90, "sl_q": 0.10}

configs = []

for mt in [0.001, 0.002, 0.005, 0.01, 0.02]:
    configs.append({
        "name": f"S43_mag{mt}",
        "fn": "experiments.bb_strategies.run_strategy_43",
        "kwargs": {**P, "magnitude_threshold": mt}
    })

baselines = [
    ("S1_BB", "experiments.bb_strategies.run_strategy_1"),
    ("S5_BB_Narrow", "experiments.bb_strategies.run_strategy_5"),
    ("S20_OB", "experiments.bb_strategies.run_strategy_20"),
    ("S28_VolOB", "experiments.bb_strategies.run_strategy_28"),
    ("S34_VWAP_OB", "experiments.bb_strategies.run_strategy_34"),
    ("S38_LowVol_OB", "experiments.bb_strategies.run_strategy_38"),
]
for name, fn in baselines:
    configs.append({"name": f"BL_{name}", "fn": fn, "kwargs": {**P}})

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

print("\n" + "=" * 130)
print(f"{'Config':<40} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'Ret(T)':>7} {'Ret(Te)':>7} {'MDD(T)':>7} {'MDD(Te)':>7}")
print("=" * 130)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        print(f"{name:<40} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f} "
              f"{r_tr.get('mdd_pct',0):>6.1f} {r_te.get('mdd_pct',0):>6.1f}")

print("\n" + "=" * 130)
print("BEST PER STRATEGY (by Test Sharpe x sqrt(n_te)^0.3)")
print("=" * 130)
groups = {"S43": "S43_", "BL_S1_BB": "BL_S1_BB", "BL_S5_BB_Narrow": "BL_S5_BB_Narrow",
          "BL_S20_OB": "BL_S20_OB", "BL_S28_VolOB": "BL_S28_VolOB",
          "BL_S34_VWAP_OB": "BL_S34_VWAP_OB", "BL_S38_LowVol_OB": "BL_S38_LowVol_OB"}
for label, prefix in groups.items():
    subset = [r for n, r in results.items() if n.startswith(prefix)]
    if not subset:
        continue
    best = max(subset, key=lambda r: (r.get("test", {}).get("sharpe", 0) or 0) * (r.get("test", {}).get("n", 0) or 0) ** 0.3)
    r_tr, r_te = best.get("train", {}), best.get("test", {})
    print(f"  {label:<20} Tr={r_tr.get('n',0):>4}t/{r_tr.get('sharpe',0):.2f}/{r_tr.get('total_return_pct',0):+.1f}%  "
          f"Te={r_te.get('n',0):>4}t/{r_te.get('sharpe',0):.2f}/{r_te.get('total_return_pct',0):+.1f}%")

print("\n=== TOP-5 BASELINE REFERENCE (from AGENTS.md) ===")
print("  #2  BB Mean Rev:     919/355t  T1.75 Te2.21  Ret +24.3%/+5.8%")
print("  #5  BB Narrow Mean:  718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  #8  %B Reversion:    435/157t  T2.20 Te3.66  Ret +15.7%/+4.5%")
print("  #9  RSI Divergence:   54/21t   T1.67 Te6.50  Ret  +1.3%/+0.9%")
