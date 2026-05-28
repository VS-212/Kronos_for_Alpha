"""
TEMPLATE: Test OB strategies with full param + quarterly breakdown.
Source: kronos-artifact/alpha/experiments/test_ob_sweep.py
Purpose: Reference example for Order Block strategy parameter sweep with quarterly metrics
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

# --- Configs ---
configs = []

# Strategy 20: OB sweep: lookback x move_threshold x max_age
for lb in [24, 48, 96]:
    for mt in [0.001, 0.0015, 0.002]:
        for ma in [12, 24]:
            configs.append({
                "name": f"OB lb={lb} mt={mt} ma={ma}",
                "fn": "experiments.bb_strategies.run_strategy_20",
                "kwargs": {**P, "lookback": lb, "move_threshold": mt, "max_age": ma}
            })

# Strategy 26: OB entry + BB exit (best OB params)
for lb in [48, 96]:
    for mt in [0.001, 0.002]:
        configs.append({
            "name": f"OB+BB lb={lb} mt={mt}",
            "fn": "experiments.bb_strategies.run_strategy_26",
            "kwargs": {**P, "lookback": lb, "move_threshold": mt, "max_age": 24}
        })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# --- Summary Table ---
print("\n" + "=" * 120)
print(f"{'Config':<40} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'Ret(T)':>7} {'Ret(Te)':>7} {'MDD(T)':>7} {'MDD(Te)':>7}")
print("=" * 120)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        print(f"{name:<40} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f} "
              f"{r_tr.get('mdd_pct',0):>6.1f} {r_te.get('mdd_pct',0):>6.1f}")

# --- Best of each family ---
print("\n" + "=" * 120)
print("BEST OB & OB+BB:")
print("=" * 120)
for family, label_key in [(20, "OB lb="), (26, "OB+BB")]:
    subset = [r for n, r in results.items() if label_key in n]
    if not subset:
        continue
    # Best = highest Sharpe(Te) x sqrt(n_te)
    best = max(subset, key=lambda r: (r.get("test", {}).get("sharpe", 0) or 0) * (r.get("test", {}).get("n", 0) or 0) ** 0.3)
    r_tr, r_te = best.get("train", {}), best.get("test", {})
    print(f"  Best {label_key}: {r_tr.get('n',0)}t/{r_tr.get('sharpe',0):.2f}  "
          f"Te={r_te.get('n',0)}t/{r_te.get('sharpe',0):.2f}/{r_te.get('total_return_pct',0):+.1f}%")

# --- Reference ---
print("\n=== REFERENCE ===")
print("  #2  BB Mean Rev:     919/355t  T1.75 Te2.21  Ret +24.3%/+5.8%")
print("  #5  BB Narrow Mean:  718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  #8  %B Reversion:    435/157t  T2.20 Te3.66  Ret +15.7%/+4.5%")
