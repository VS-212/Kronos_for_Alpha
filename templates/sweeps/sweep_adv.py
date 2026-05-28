"""
TEMPLATE: Asymmetry + dispersion + exit_on_flip sweep for S44/S45.
Source: kronos-artifact/alpha/experiments/sweep_adv.py
Purpose: Reference example for advanced parameter sweep (asymmetry, dispersion)
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

# S44: asymmetry sweep (asymmetry_min)
for asym in [1.5, 2.0, 3.0, 5.0, 10.0]:
    configs.append({
        "name": f"S44_asym{asym}",
        "fn": "experiments.bb_strategies.run_strategy_44",
        "kwargs": {**P, "asymmetry_min": asym}
    })

# S45: combined (asymmetry + dispersion + flip)
for asym in [1.5, 2.0, 3.0]:
    for dc in [0.01, 0.02, 0.03]:
        configs.append({
            "name": f"S45_asym{asym}_dc{dc}",
            "fn": "experiments.bb_strategies.run_strategy_45",
            "kwargs": {**P, "asymmetry_min": asym, "dispersion_cap": dc}
        })

# Baselines
baselines = [
    ("S1_BB", "experiments.bb_strategies.run_strategy_1"),
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
print("BEST PER GROUP (by Test Sharpe x n_te^0.3)")
print("=" * 130)
groups = {"S44_Asymmetry": "S44_", "S45_Combined": "S45_", "BL_S1_BB": "BL_S1_BB",
          "BL_S20_OB": "BL_S20_OB", "BL_S38_LowVol_OB": "BL_S38_LowVol_OB"}
for label, prefix in groups.items():
    subset = [r for n, r in results.items() if n.startswith(prefix)]
    if not subset:
        continue
    best = max(subset, key=lambda r: (r.get("test", {}).get("sharpe", 0) or 0) * (r.get("test", {}).get("n", 0) or 0) ** 0.3)
    r_tr, r_te = best.get("train", {}), best.get("test", {})
    print(f"  {label:<22} Tr={r_tr.get('n',0):>4}t/{r_tr.get('sharpe',0):.2f}/{r_tr.get('total_return_pct',0):+.1f}%  "
          f"Te={r_te.get('n',0):>4}t/{r_te.get('sharpe',0):.2f}/{r_te.get('total_return_pct',0):+.1f}%")
