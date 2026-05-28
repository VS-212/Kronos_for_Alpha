"""
TEMPLATE: Hour x Quarter breakdown for #5 BB Narrow Mean Rev.
Source: kronos-artifact/alpha/experiments/test_hour5.py
Purpose: Reference example for temporal distribution analysis (hour/quarter pivot)
Usage: Adapt to your instrument/asset before production use.
Status: Example — not production code
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.evaluation.output import load_samples
from src.strategies import load_config, load_mamba_sber, run_strategy_5

cfg = load_config()
mamba = load_mamba_sber(cfg)
df = load_samples("/tmp/opencode/SBER_samples_pl6_sc5.parquet")
df["year"] = df["month"].str[:4]
train_df = df[df["year"] == "2025"]
test_df = df[df["year"] == "2026"]

tr = run_strategy_5(train_df, mamba, 6, 5, 0.90, 0.10)
te = run_strategy_5(test_df, mamba, 6, 5, 0.90, 0.10)

for label, td in [("Train", tr), ("Test", te)]:
    if len(td) == 0:
        continue
    t = td.copy()
    t["hour"] = t["pred_ts"].dt.hour
    print(f"\n{'='*80}")
    print(f"  #5 BB Narrow Mean Rev — {label}")
    print(f"{'='*80}")

    pivots = []
    for h in sorted(t["hour"].unique()):
        for q, grp in t[t["hour"] == h].groupby("quarter", sort=True):
            n = len(grp)
            if n < 3:
                continue
            avg_ret = grp["return"].mean()
            std_ret = grp["return"].std()
            sharpe = avg_ret / max(std_ret, 1e-10) * np.sqrt(252 * 6 / 6)
            wins = int((grp["return"] > 0).sum())
            losses = n - wins
            sum_wins = float(grp.loc[grp["return"] > 0, "return"].sum()) if wins else 0.0
            sum_losses = abs(float(grp.loc[grp["return"] <= 0, "return"].sum())) if losses else 0.0
            pf = sum_wins / max(sum_losses, 1e-10)
            pivots.append({"hour": h, "quarter": q, "n": n, "sharpe": sharpe, "pf": pf})

    pivots = pd.DataFrame(pivots)
    if len(pivots) == 0:
        continue

    # Pivot tables
    for metric in ["sharpe", "pf"]:
            pt = pivots.pivot_table(index="hour", columns="quarter", values=metric, aggfunc="first")
            pt = pt.round(2)

            # Correct weighted average per hour across all quarters
            hour_wavg = {}
            for h in sorted(t["hour"].unique()):
                hr = pivots[pivots["hour"] == h]
                if len(hr) > 0:
                    wavg = np.average(hr[metric], weights=hr["n"])
                    hour_wavg[h] = (wavg, hr["n"].sum())

            print(f"\n  {metric.upper()} by Hour x Quarter:")
            cols = sorted(pivots["quarter"].unique())
            header = f"  {'Hour':>5}"
            for c in cols:
                header += f" {c:>10}"
            header += f" {'Avg':>8}"
            print(header)
            print(f"  {'─'*70}")

            for h in sorted(pt.index):
                line = f"  {h:>5}"
                hr = pivots[pivots["hour"] == h]
                for c in cols:
                    if c in pt.columns and pd.notna(pt.loc[h, c]):
                        line += f" {pt.loc[h, c]:>9.2f} "
                    else:
                        line += f" {'':>10}"
                wavg, total_n = hour_wavg.get(h, (0, 0))
                line += f" {wavg:>7.2f} ({total_n})"
                print(line)

    # Overall per quarter (all hours)
    print(f"\n  ALL HOURS by Quarter:")
    print(f"  {'Quarter':>10} {'n':>5} {'Sharpe':>7} {'PF':>6}")
    for q, grp in t.groupby("quarter", sort=True):
        n = len(grp)
        avg_ret = grp["return"].mean()
        std_ret = grp["return"].std()
        sharpe = avg_ret / max(std_ret, 1e-10) * np.sqrt(252 * 6 / 6)
        wins = int((grp["return"] > 0).sum())
        losses = n - wins
        sum_wins = float(grp.loc[grp["return"] > 0, "return"].sum()) if wins else 0.0
        sum_losses = abs(float(grp.loc[grp["return"] <= 0, "return"].sum())) if losses else 0.0
        pf = sum_wins / max(sum_losses, 1e-10)
        print(f"  {q:>10} {n:>5} {sharpe:>7.2f} {pf:>6.2f}")
