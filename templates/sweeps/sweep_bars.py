"""
TEMPLATE: Test bar type strategies 31-32 with parameter sweep.
Source: kronos-artifact/alpha/experiments/test_bars_sweep.py
Purpose: Reference example for bar-type strategy parameter sweep
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

# 31: Bar Type + Kronos — sweep min_edge
for me in [0.02, 0.03, 0.05, 0.08]:
    configs.append({
        "name": f"Bar+Kr edge={me}",
        "fn": "experiments.bb_strategies.run_strategy_31",
        "kwargs": {**P, "min_edge": me, "bar_lookback": 5000}
    })

# 32: Bar-filtered OB — sweep min_edge
for me in [0.02, 0.03, 0.05]:
    configs.append({
        "name": f"BarOB edge={me}",
        "fn": "experiments.bb_strategies.run_strategy_32",
        "kwargs": {**P, "min_edge": me, "bar_lookback": 5000}
    })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# Summary
print("\n" + "=" * 110)
print(f"{'Config':<25} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'PF(T)':>5} {'PF(Te)':>5} {'AvgRet(T)':>9} {'AvgRet(Te)':>9} {'Ret(T)':>7} {'Ret(Te)':>7}")
print("=" * 110)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        print(f"{name:<25} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('profit_factor',0):>5.2f} {r_te.get('profit_factor',0):>5.2f} "
              f"{r_tr.get('avg_return_pct',0):>+7.4f} {r_te.get('avg_return_pct',0):>+7.4f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f}")

print("\n=== REFERENCE ===")
print("  OB mt=2 ma=12:     750/163t  T1.57 Te5.88  Ret +18.5%/+6.4%")
print("  VolOB vm=2.0:      119/52t   T2.03 Te7.46  Ret +4.0%/+2.3%")
print("  BB Narrow:         718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  %%B Rev:            435/157t  T2.20 Te3.66  Ret +15.7%/+4.5%")
