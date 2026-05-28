"""
TEMPLATE: ATR TP/SL sweep — reduced to ~100 configs for speed.
Source: kronos-artifact/alpha/experiments/test_tpsl_sweep.py
Purpose: Reference example for ATR-based TP/SL + deferred entry parameter sweep
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

STRATEGIES = {
    1:  ("S1_BB_Consensus", "experiments.bb_strategies.run_strategy_1"),
    5:  ("S5_BB_Narrow",    "experiments.bb_strategies.run_strategy_5"),
    20: ("S20_OB",          "experiments.bb_strategies.run_strategy_20"),
    28: ("S28_VolOB",       "experiments.bb_strategies.run_strategy_28"),
    34: ("S34_VWAP_OB",     "experiments.bb_strategies.run_strategy_34"),
    38: ("S38_LowVol_OB",   "experiments.bb_strategies.run_strategy_38"),
}

configs = []

# Phase 1: Baseline (Kronos q90/q10)
for sid, (sname, fn) in STRATEGIES.items():
    configs.append({"name": f"BL_{sname}", "fn": fn, "kwargs": {**P}})

# Phase 2: ATR-only — tp=[0.5,1.0,2.0,3.0] x sl=[1.0,2.0]
for sid, (sname, fn) in STRATEGIES.items():
    for tp in [0.5, 1.0, 2.0, 3.0]:
        for sl in [1.0, 2.0]:
            configs.append({
                "name": f"ATR_{sname}_tp{tp}_sl{sl}",
                "fn": fn,
                "kwargs": {**P, "atr_tp_mult": tp, "atr_sl_mult": sl}
            })

# Phase 3: Deferred — save_first_n=[1,2,3]
for sid, (sname, fn) in STRATEGIES.items():
    for n in [1, 2, 3]:
        configs.append({
            "name": f"DEF_{sname}_n{n}",
            "fn": fn,
            "kwargs": {**P, "save_first_n": n}
        })

# Phase 4: Combined — top-3 strategies x tp=[1.0,2.0] x sl=[1.0,2.0] x n=[1,2]
BEST_3_SIDS = [20, 34, 38]  # top-3 by baseline Sharpe(Te)
for sid, (sname, fn) in [(s, STRATEGIES[s]) for s in BEST_3_SIDS]:
    for tp in [1.0, 2.0]:
        for sl in [1.0, 2.0]:
            for n in [1, 2]:
                configs.append({
                    "name": f"COMB_{sname}_tp{tp}_sl{sl}_n{n}",
                    "fn": fn,
                    "kwargs": {**P, "atr_tp_mult": tp, "atr_sl_mult": sl,
                               "save_first_n": n}
                })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# --- Table ---
rows = []
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        rows.append({
            "name": name,
            "n_tr": r_tr["n"], "n_te": r_te.get("n", 0),
            "sh_tr": r_tr.get("sharpe", 0), "sh_te": r_te.get("sharpe", 0),
            "ret_tr": r_tr.get("total_return_pct", 0),
            "ret_te": r_te.get("total_return_pct", 0),
            "mdd_tr": r_tr.get("mdd_pct", 0), "mdd_te": r_te.get("mdd_pct", 0),
        })

print("\n" + "=" * 140)
print(f"{'Name':<45} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'Ret(T)':>7} {'Ret(Te)':>7} {'MDD(T)':>7} {'MDD(Te)':>7}")
print("=" * 140)
for row in rows:
    print(f"{row['name']:<45} {row['n_tr']:>5} {row['n_te']:>5} "
          f"{row['sh_tr']:>6.2f} {row['sh_te']:>6.2f} "
          f"{row['ret_tr']:>+6.1f} {row['ret_te']:>+6.1f} "
          f"{row['mdd_tr']:>6.1f} {row['mdd_te']:>6.1f}")

# --- Best per phase per strategy ---
print("\n" + "=" * 140)
print("BEST PER STRATEGY (ranked by Sharpe(Te) x n_te^0.3)")
print("=" * 140)

for sid, (sname, fn) in STRATEGIES.items():
    print(f"\n  [{sid}] {sname}:  Baseline Sh(Te)={[r['sh_te'] for r in rows if r['name']==f'BL_{sname}'][0]:.2f}")
    for phase, prefix in [("ATR-only", "ATR_"), ("Deferred", "DEF_"),
                          ("Combined", "COMB_")]:
        subset = [r for r in rows if r["name"].startswith(f"{prefix}{sname}")]
        if not subset:
            continue
        best = max(subset, key=lambda r: r["sh_te"] * (r["n_te"] ** 0.3))
        imp = best["sh_te"] - [r['sh_te'] for r in rows if r['name']==f'BL_{sname}'][0]
        print(f"    {phase:>8}: {best['name'][len(prefix):]:<45} "
              f"Tr={best['n_tr']}t/{best['sh_tr']:.2f}  "
              f"Te={best['n_te']}t/{best['sh_te']:.2f} ({imp:+.2f} vs BL)  "
              f"Ret={best['ret_te']:+.1f}%")
