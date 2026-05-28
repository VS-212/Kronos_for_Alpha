"""
TEMPLATE: Test FVG (single) vs Multi-FVG with full per-quarter metrics.
Source: kronos-artifact/alpha/experiments/test_fvg_sweep.py
Purpose: Reference example for FVG strategy sweep with quarterly breakdown
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

# Strategy 22: current FVG — sweep lookback
for lb in [12, 24, 48, 96]:
    configs.append({
        "name": f"FVG lb={lb}",
        "fn": "experiments.bb_strategies.run_strategy_22",
        "kwargs": {**P, "lookback": lb}
    })

# Strategy 30: Multi-FVG — sweep lookback x min_fvgs x max_age
for lb in [12, 24, 48, 96]:
    for mf in [2, 3]:
        for ma in [24, 48]:
            configs.append({
                "name": f"M-FVG lb={lb} mf={mf} ma={ma}",
                "fn": "experiments.bb_strategies.run_strategy_30",
                "kwargs": {**P, "lookback": lb, "min_fvgs": mf, "max_age": ma}
            })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# --- Summary Table ---
print("\n" + "=" * 130)
print(f"{'Config':<35} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'PF(T)':>6} {'PF(Te)':>6} {'AvgRet(T)':>9} {'AvgRet(Te)':>9} {'Ret(T)':>7} {'Ret(Te)':>7}")
print("=" * 130)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        avg_tr = r_tr.get("avg_return_pct", 0)
        avg_te = r_te.get("avg_return_pct", 0)
        print(f"{name:<35} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('profit_factor',0):>5.2f} {r_te.get('profit_factor',0):>5.2f} "
              f"{avg_tr:>+7.4f} {avg_te:>+7.4f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f}")

# --- Per-quarter breakdowns ---
print("\n" + "=" * 130)
print("PER-QUARTER BREAKDOWNS (best of each)")
print("=" * 130)

# Best single FVG and best multi-FVG
best_single = max(
    [(n, r) for n, r in results.items() if n.startswith("FVG ")],
    key=lambda x: x[1].get("test", {}).get("sharpe", 0) or 0
)
best_multi = max(
    [(n, r) for n, r in results.items() if n.startswith("M-FVG")],
    key=lambda x: x[1].get("test", {}).get("sharpe", 0) or 0
)

for label, (name, r) in [("BEST FVG", best_single), ("BEST Multi-FVG", best_multi)]:
    print(f"\n{label}: {name}")
    print(f"  Overall: n(Tr)={r['train'].get('n',0)} n(Te)={r['test'].get('n',0)}  "
          f"Sh(T)={r['train'].get('sharpe',0):.2f} Sh(Te)={r['test'].get('sharpe',0):.2f}  "
          f"Ret(T)={r['train'].get('total_return_pct',0):+.1f}% Ret(Te)={r['test'].get('total_return_pct',0):+.1f}%")
    for set_label, qdata in [("Train", r.get("train_q", {})), ("Test", r.get("test_q", {}))]:
        if qdata:
            print(f"  {set_label} by quarter:")
            print(f"    {'Quarter':<10} {'n':>5} {'Sharpe':>7} {'PF':>6} {'AvgRet':>9} {'Ret':>8} {'MDD':>7}")
            for q, d in sorted(qdata.items()):
                print(f"    {q:<10} {d.get('n',0):>5} {d.get('sharpe',0):>7.2f} "
                      f"{d.get('profit_factor',0):>5.2f} "
                      f"{d.get('avg_return_pct',0):>+7.4f} "
                      f"{d.get('total_return_pct',0):>+6.2f}% "
                      f"{d.get('mdd_pct',0):>+6.2f}%")

print("\n" + "=" * 130)
print("BEST OF EACH (sorted by Test Sharpe x n^0.3):")
print("=" * 130)
for family, key_prefix in [("FVG", "FVG "), ("Multi-FVG", "M-FVG")]:
    subset = [(n, r) for n, r in results.items() if n.startswith(key_prefix)]
    if not subset:
        continue
    best = max(subset, key=lambda x: (x[1].get("test", {}).get("sharpe", 0) or 0) *
               (x[1].get("test", {}).get("n", 0) or 0) ** 0.3)
    r_tr, r_te = best[1].get("train", {}), best[1].get("test", {})
    print(f"  Best {family}: {best[0]}")
    print(f"    Train: {r_tr.get('n',0)}t Sh={r_tr.get('sharpe',0):.2f} PF={r_tr.get('profit_factor',0):.2f} "
          f"AvgRet={r_tr.get('avg_return_pct',0):+.4f} Ret={r_tr.get('total_return_pct',0):+.1f}%")
    print(f"    Test:  {r_te.get('n',0)}t Sh={r_te.get('sharpe',0):.2f} PF={r_te.get('profit_factor',0):.2f} "
          f"AvgRet={r_te.get('avg_return_pct',0):+.4f} Ret={r_te.get('total_return_pct',0):+.1f}%")

print("\n=== REFERENCE ===")
print("  OB (mt=2,ma=12):  750/163t  T1.57 Te5.88  Ret +18.5%/+6.4%")
print("  VolOB vm=2.0:     119/52t   T2.03 Te7.46  Ret +4.0%/+2.3%")
print("  #5 BB Narrow:     718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
print("  #8 %%B Rev:        435/157t  T2.20 Te3.66  Ret +15.7%/+4.5%")
