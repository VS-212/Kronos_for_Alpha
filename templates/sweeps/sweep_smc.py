"""
TEMPLATE: Test Breaker Block, Volume OB, EQH/EQL strategies with param sweeps.
Source: kronos-artifact/alpha/experiments/test_smc_sweep.py
Purpose: Reference example for SMC strategy family parameter sweep
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.parallel import run_parallel

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SAMPLES_PATH = "/tmp/opencode/SBER_samples_pl6_sc5.parquet"
P = {"pred_len": 6, "sample_count": 5, "tp_q": 0.90, "sl_q": 0.10}

configs = []

# 27. Breaker Block
for lb in [48, 96]:
    for mt in [0.001, 0.002]:
        configs.append({
            "name": f"Breaker lb={lb} mt={mt}",
            "fn": "experiments.bb_strategies.run_strategy_27",
            "kwargs": {**P, "lookback": lb, "move_threshold": mt, "max_age": 24}
        })

# 28. Volume OB
for lb in [48, 96]:
    for vm in [1.5, 2.0, 3.0]:
        configs.append({
            "name": f"VolOB lb={lb} vm={vm}",
            "fn": "experiments.bb_strategies.run_strategy_28",
            "kwargs": {**P, "lookback": lb, "move_threshold": 0.002, "max_age": 24,
                       "volume_mult": vm}
        })

# 29. EQH/EQL
for tol in [0.001, 0.002]:
    for nbars in [3, 5]:
        configs.append({
            "name": f"EQH tol={tol} n={nbars}",
            "fn": "experiments.bb_strategies.run_strategy_29",
            "kwargs": {**P, "lookback": 96, "tolerance": tol, "n_bars": nbars}
        })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# Summary
print("\n" + "=" * 110)
print(f"{'Config':<30} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'Ret(T)':>7} {'Ret(Te)':>7}")
print("=" * 110)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        print(f"{name:<30} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f}")

# Best of each
print("\n" + "=" * 110)
print("BEST OF EACH:")
print("=" * 110)
for family, key in [(27, "Breaker"), (28, "VolOB"), (29, "EQH")]:
    subset = [(n, r) for n, r in results.items() if key in n]
    if not subset:
        continue
    best = max(subset, key=lambda x: (x[1].get("test", {}).get("sharpe", 0) or 0) *
               (x[1].get("test", {}).get("n", 0) or 0) ** 0.3)
    r_tr, r_te = best[1].get("train", {}), best[1].get("test", {})
    print(f"  Best {key}: {best[0]}")
    print(f"    Train: {r_tr.get('n',0)}t Sh={r_tr.get('sharpe',0):.2f} Ret={r_tr.get('total_return_pct',0):+.1f}%")
    print(f"    Test:  {r_te.get('n',0)}t Sh={r_te.get('sharpe',0):.2f} Ret={r_te.get('total_return_pct',0):+.1f}%")

print("\n=== REFERENCE ===")
print("  OB (mt=2,ma=12):  750/163t  T1.57 Te5.88  Ret +18.5%/+6.4%")
print("  OB (mt=2,ma=24):  806/234t  T1.42 Te5.55  Ret +17.6%/+8.7%")
print("  #5 BB Narrow:     718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  #8 %%B Rev:        435/157t  T2.20 Te3.66  Ret +15.7%/+4.5%")
