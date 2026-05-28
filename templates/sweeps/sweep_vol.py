"""
TEMPLATE: Test volatility strategies 38-42 with parameter sweep.
Source: kronos-artifact/alpha/experiments/test_vol_sweep.py
Purpose: Reference example for volatility-based strategy parameter sweep
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

# 38: Low-vol OB — sweep adr_pct
for ap in [0.2, 0.3, 0.5]:
    configs.append({
        "name": f"LowOB ap={ap}",
        "fn": "experiments.bb_strategies.run_strategy_38",
        "kwargs": {**P, "adr_lookback": 800, "ob_lookback": 48, "adr_period": 20, "adr_pct": ap}
    })

# 39: High-vol BB — sweep atr_pct
for ap in [0.5, 0.7]:
    configs.append({
        "name": f"HiBB ap={ap}",
        "fn": "experiments.bb_strategies.run_strategy_39",
        "kwargs": {**P, "lookback": 96, "atr_period": 14, "atr_pct": ap}
    })

# 40: Regime Switch
configs.append({
    "name": "Regime",
    "fn": "experiments.bb_strategies.run_strategy_40",
    "kwargs": {**P, "adr_lookback": 800, "atr_period": 14, "adr_period": 20}
})

# 41: ATR Expansion — sweep rising_n
for rn in [2, 3, 5]:
    configs.append({
        "name": f"ATR-Exp rn={rn}",
        "fn": "experiments.bb_strategies.run_strategy_41",
        "kwargs": {**P, "lookback": 96, "atr_period": 14, "rising_n": rn}
    })

# 42: ADR Compression — sweep adr_pct
for ap in [0.1, 0.2, 0.3]:
    configs.append({
        "name": f"ADR-Cmp ap={ap}",
        "fn": "experiments.bb_strategies.run_strategy_42",
        "kwargs": {**P, "adr_lookback": 800, "adr_period": 20, "adr_pct": ap}
    })

print(f"Total configs: {len(configs)}")
results = run_parallel(configs, PROJECT_ROOT, SAMPLES_PATH, max_workers=12)

# Summary
print("\n" + "=" * 130)
hd = f"{'Config':<25} {'n(Tr)':>5} {'n(Te)':>5} {'Sh(T)':>6} {'Sh(Te)':>6} {'PF(T)':>5} {'PF(Te)':>5} {'AvgRet(T)':>9} {'AvgRet(Te)':>9} {'Ret(T)':>7} {'Ret(Te)':>7}"
print(hd)
print("=" * 130)
for name in sorted(results.keys()):
    r = results[name]
    r_tr, r_te = r.get("train", {}), r.get("test", {})
    if r_tr and r_tr.get("n", 0) > 0:
        avg_tr = r_tr.get("avg_return_pct", 0)
        avg_te = r_te.get("avg_return_pct", 0)
        print(f"{name:<25} {r_tr['n']:>5} {r_te.get('n',0):>5} "
              f"{r_tr.get('sharpe',0):>6.2f} {r_te.get('sharpe',0):>6.2f} "
              f"{r_tr.get('profit_factor',0):>5.2f} {r_te.get('profit_factor',0):>5.2f} "
              f"{avg_tr:>+7.4f} {avg_te:>+7.4f} "
              f"{r_tr.get('total_return_pct',0):>+6.1f} {r_te.get('total_return_pct',0):>+6.1f}")

print("\n" + "=" * 130)
print("BEST OF EACH:")
print("=" * 130)
for family, key in [(38, "LowOB"), (39, "HiBB"), (40, "Regime"),
                     (41, "ATR-Exp"), (42, "ADR-Cmp")]:
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
print("  VWAP+OB:           539/158t  T1.67 Te5.40  Ret +13.6%/+5.8%")
print("  VolOB vm=2.0:      119/52t   T2.03 Te7.46  Ret +4.0%/+2.3%")
print("  BB Narrow:         718/154t  T1.91 Te4.18  Ret +21.9%/+4.9%")
