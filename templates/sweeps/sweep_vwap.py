"""
TEMPLATE: Test VWAP strategies 33-37 with parameter sweeps.
Source: kronos-artifact/alpha/experiments/test_vwap_sweep.py
Purpose: Reference example for VWAP strategy family parameter sweep
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

# 33: VWAP Bands Reversion — sweep k & lookback
for k in [1.5, 2.0, 2.5]:
    for lb in [48, 96]:
        configs.append({
            "name": f"VWAP-Rev k={k} lb={lb}",
            "fn": "experiments.bb_strategies.run_strategy_33",
            "kwargs": {**P, "k": k, "lookback": lb}
        })

# 34: VWAP + OB — sweep k
for k in [1.0, 1.5, 2.0]:
    configs.append({
        "name": f"VWAP+OB k={k}",
        "fn": "experiments.bb_strategies.run_strategy_34",
        "kwargs": {**P, "k": k, "lookback": 96}
    })

# 35: VWAP Cross
configs.append({
    "name": "VWAP Cross",
    "fn": "experiments.bb_strategies.run_strategy_35",
    "kwargs": {**P, "lookback": 96}
})

# 36: Anchored VWAP — sweep fractal_max_age
for fa in [12, 24]:
    configs.append({
        "name": f"Anchor-VWAP fa={fa}",
        "fn": "experiments.bb_strategies.run_strategy_36",
        "kwargs": {**P, "fractal_max_age": fa, "lookback": 192}
    })

# 37: VWAP Volume — sweep volume_mult
for vm in [1.5, 2.0, 3.0]:
    configs.append({
        "name": f"VWAP-Vol vm={vm}",
        "fn": "experiments.bb_strategies.run_strategy_37",
        "kwargs": {**P, "lookback": 96, "volume_mult": vm}
    })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# Summary
print("\n" + "=" * 130)
print(f"{'Config':<30} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'PF(T)':>5} {'PF(Te)':>5} {'AvgRet(T)':>9} {'AvgRet(Te)':>9} {'Ret(T)':>7} {'Ret(Te)':>7}")
print("=" * 130)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        avg_tr = r_tr.get("avg_return_pct", 0)
        avg_te = r_te.get("avg_return_pct", 0)
        print(f"{name:<30} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('profit_factor',0):>5.2f} {r_te.get('profit_factor',0):>5.2f} "
              f"{avg_tr:>+7.4f} {avg_te:>+7.4f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f}")

print("\n" + "=" * 130)
print("BEST OF EACH:")
print("=" * 130)
for family, key in [(33, "VWAP-Rev"), (34, "VWAP+OB"), (35, "VWAP Cross"),
                     (36, "Anchor-VWAP"), (37, "VWAP-Vol")]:
    subset = [(n, r) for n, r in results.items() if key in n]
    if not subset:
        continue
    best = max(subset, key=lambda x: (x[1].get("test", {}).get("sharpe", 0) or 0) *
               (x[1].get("test", {}).get("n", 0) or 0) ** 0.3)
    r_tr, r_te = best[1].get("train", {}), best[1].get("test", {})
    print(f"  Best {family}: {best[0]}")
    print(f"    Train: {r_tr.get('n',0)}t Sh={r_tr.get('sharpe',0):.2f} PF={r_tr.get('profit_factor',0):.2f} "
          f"Ret={r_tr.get('total_return_pct',0):+.1f}%")
    print(f"    Test:  {r_te.get('n',0)}t Sh={r_te.get('sharpe',0):.2f} PF={r_te.get('profit_factor',0):.2f} "
          f"Ret={r_te.get('total_return_pct',0):+.1f}%")

print("\n=== REFERENCE ===")
print("  OB mt=2 ma=12:     750/163t  T1.57 Te5.88  Ret +18.5%/+6.4%")
print("  VolOB vm=2.0:      119/52t   T2.03 Te7.46  Ret +4.0%/+2.3%")
print("  BB Narrow:         718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
