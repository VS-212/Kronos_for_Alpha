"""
M-QUARTERLY: Quarterly performance breakdown + CSV output
Contract: labels, per_bar arrays, timestamps → quarterly tables + CSV files
Status: ✅ ready
"""

import csv
import os

import numpy as np

# Metric key lists (from backtest_sber_v2 STRAT_METRICS + TRADE_METRICS)
STRAT_METRICS_KEYS = [
    "sharpe", "sortino", "max_dd", "profit_factor", "win_rate",
    "calmar", "avg_return", "total_return", "aver_ret",
    "trade_pct", "psr", "dsr",
]
TRADE_METRICS_KEYS = ["n_trades", "n_long", "n_short", "n_bar_active"]
ALL_METRICS_KEYS = STRAT_METRICS_KEYS + TRADE_METRICS_KEYS


def compute_quarterly_tables(labels, per_bars, ts, results=None):
    """Group bars by quarter, print AverRet + Sharpe tables, return (bars_per_qtr, sorted_qtrs)."""
    bars_per_qtr = {}
    for bi, t in enumerate(ts):
        parts = str(t).split('-')
        y, m = int(parts[0]), int(parts[1])
        qtr = f"{y}-Q{(m-1)//3+1}"
        bars_per_qtr.setdefault(qtr, []).append(bi)

    for qtr, indices in bars_per_qtr.items():
        bars_per_qtr[qtr] = np.array(indices)

    sorted_qtrs = sorted(bars_per_qtr.keys())
    nq = len(sorted_qtrs)

    ANNUAL_BARS = 6 * 24 * 252

    print(f"\n  {'─'* (22 + nq * 24)}")
    print(f"  QUARTERLY AverRet (% per active bar)")
    print(f"  {'─'* (22 + nq * 24)}")
    print(f"  {'Strategy':<22s}", end="")
    for qtr in sorted_qtrs:
        print(f"  {qtr:>11s}      ", end="")
    print(f"  {'overall':>11s}")
    print(f"  {'':22s}{'─'* (nq * 18)}")

    for l, pb in zip(labels, per_bars):
        print(f"  {l:<22s}", end="")
        for qtr in sorted_qtrs:
            mask = bars_per_qtr[qtr]
            active = pb[mask] != 0
            if active.sum() < 3:
                s = f"{'low':>11s}"
            else:
                ar = pb[mask][active].mean()
                s = f"{ar:>11.4%}"
            print(f"  {s}  ", end="")
        active_all = pb != 0
        overall_ar = pb[active_all].mean() if active_all.sum() > 3 else 0.0
        print(f"  {overall_ar:>11.4%}")

    print(f"  {'─'* (22 + nq * 24)}")
    print(f"  QUARTERLY Sharpe (annualized)")
    print(f"  {'─'* (22 + nq * 24)}")
    print(f"  {'Strategy':<22s}", end="")
    for qtr in sorted_qtrs:
        print(f"  {qtr:>11s}      ", end="")
    print(f"  {'overall':>11s}")
    print(f"  {'':22s}{'─'* (nq * 18)}")

    for l, pb in zip(labels, per_bars):
        print(f"  {l:<22s}", end="")
        for qtr in sorted_qtrs:
            mask = bars_per_qtr[qtr]
            active = pb[mask] != 0
            if active.sum() < 10:
                s = f"{'low':>11s}"
            else:
                r = pb[mask][active]
                sr = r.mean() / max(r.std(), 1e-10) * np.sqrt(ANNUAL_BARS)
                s = f"{sr:>11.2f}"
            print(f"  {s}  ", end="")
        active_all = pb != 0
        if active_all.sum() > 10:
            ra = pb[active_all]
            overall_sr = ra.mean() / max(ra.std(), 1e-10) * np.sqrt(ANNUAL_BARS)
        else:
            overall_sr = 0.0
        print(f"  {overall_sr:>11.2f}")

    return bars_per_qtr, sorted_qtrs


def save_results_csv(results, labels):
    """Save strategy metrics to data/v3/results/sber_backtest_v2.csv."""
    os.makedirs("data/v3/results", exist_ok=True)
    with open("data/v3/results/sber_backtest_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy"] + ALL_METRICS_KEYS)
        for r, l in zip(results, labels):
            row = [l]
            for k in ALL_METRICS_KEYS:
                v = r.get(k, 0)
                if isinstance(v, (int, np.integer)):
                    row.append(int(v))
                elif isinstance(v, (float, np.floating)):
                    row.append(round(float(v), 6))
                else:
                    row.append(str(v))
            w.writerow(row)
    print(f"Saved: data/v3/results/sber_backtest_v2.csv")


def save_quarterly_csv(labels, per_bars, bars_per_qtr, sorted_qtrs, ts, ANNUAL_BARS=6*24*252):
    """Save quarterly breakdown + overall to CSV."""
    os.makedirs("data/v3/results", exist_ok=True)
    with open("data/v3/results/sber_backtest_v2_quarterly.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "quarter", "n_bars", "n_active", "total_pnl", "profit_factor", "aver_ret", "sharpe"])
        for l, pb in zip(labels, per_bars):
            for qtr in sorted_qtrs:
                mask = bars_per_qtr[qtr]
                active = pb[mask] != 0
                n_active = int(active.sum())
                if n_active < 10:
                    continue
                rtrn = pb[mask][active]
                total_pnl = round(float(rtrn.sum()), 6)
                pos = rtrn[rtrn > 0].sum()
                neg = abs(rtrn[rtrn < 0].sum())
                pf = round(float(pos / max(neg, 1e-12)), 4)
                ar = round(float(rtrn.mean()), 8)
                sr = round(float(rtrn.mean() / max(rtrn.std(), 1e-10) * np.sqrt(ANNUAL_BARS)), 2)
                w.writerow([l, qtr, len(mask), n_active, total_pnl, pf, ar, sr])
            active_all = pb != 0
            if active_all.sum() > 10:
                ra = pb[active_all]
                pos = ra[ra > 0].sum()
                neg = abs(ra[ra < 0].sum())
                pf = round(float(pos / max(neg, 1e-12)), 4)
                w.writerow([l, "overall", len(pb), int(active_all.sum()),
                            round(float(ra.sum()), 6), pf,
                            round(float(ra.mean()), 8),
                            round(float(ra.mean() / max(ra.std(), 1e-10) * np.sqrt(ANNUAL_BARS)), 2)])
    print(f"Saved: data/v3/results/sber_backtest_v2_quarterly.csv")


def compute_h1_2026_metrics(labels, per_bars, bars_per_qtr, sorted_qtrs, ts, ANNUAL_BARS=6*24*252):
    """Filter H1 2026 (Q1+Q2), compute metrics, print top 15 table, return sorted list of dicts."""
    n_bars = len(ts)
    h1_2026_mask = np.zeros(n_bars, dtype=bool)
    for qtr in sorted_qtrs:
        if qtr in ("2026-Q1", "2026-Q2"):
            indices = bars_per_qtr[qtr]
            h1_2026_mask[indices[indices < n_bars]] = True

    h1_metrics = []
    for l, pb in zip(labels, per_bars):
        pb_h1 = pb.copy()
        pb_h1[~h1_2026_mask] = 0.0
        active = pb_h1 != 0
        n_active = int(active.sum())
        if n_active < 20:
            continue
        r = pb_h1[active]
        cumul = np.cumprod(1 + pb_h1)
        sharpe = float(r.mean() / max(r.std(), 1e-10) * np.sqrt(ANNUAL_BARS))
        tot_ret = float(cumul[-1] / cumul[0] - 1)
        pos_sum = r[r > 0].sum()
        neg_sum = abs(r[r < 0].sum())
        pf = float(pos_sum / max(neg_sum, 1e-12))
        wr = float((r > 0).sum() / max(n_active, 1))
        aver_ret = float(r.sum() / max(n_active, 1))
        dd = float((cumul / np.maximum.accumulate(cumul) - 1).min())
        trades = n_active
        h1_metrics.append({"strategy": l, "sharpe": sharpe, "tot_ret": tot_ret, "pf": pf,
                            "wr": wr, "aver_ret": aver_ret, "max_dd": dd, "trades": trades})

    h1_metrics.sort(key=lambda d: -d["sharpe"])

    print(f"\n  {'═'*100}")
    print(f"  H1 2026 (Q1+Q2) — TOP 15 by Sharpe")
    print(f"  {'═'*100}")
    print(f"  {'Rank':<5} {'Strategy':<50} {'Sharpe':<9} {'TotRet':<9} {'PF':<8} {'WR':<7} {'AverRet':<10} {'MaxDD':<10} {'Trades':<7}")
    print(f"  {'─'*110}")
    for i, m in enumerate(h1_metrics[:15]):
        print(f"  {i+1:<5} {m['strategy']:<50} {m['sharpe']:>7.3f}  {m['tot_ret']:>7.4%}  {m['pf']:>6.4f}  {m['wr']:>5.2%}  {m['aver_ret']:>8.5%}  {m['max_dd']:>8.4%}  {m['trades']:<7d}")

    return h1_metrics


def save_all_results(results, labels, per_bars, bars_per_qtr, sorted_qtrs, ts, ANNUAL_BARS=6*24*252):
    """Combined re-save: both CSV files from results + per_bars."""
    labels_all = [r["label"] for r in results]
    os.makedirs("data/v3/results", exist_ok=True)

    with open("data/v3/results/sber_backtest_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy"] + ALL_METRICS_KEYS)
        for r in results:
            w.writerow([r["label"]] + [r.get(k, 0) for k in ALL_METRICS_KEYS])

    with open("data/v3/results/sber_backtest_v2_quarterly.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "quarter", "n_bars", "n_active", "total_pnl", "profit_factor", "aver_ret", "sharpe"])
        for l, pb in zip(labels_all, per_bars):
            for qtr in sorted_qtrs:
                mask = bars_per_qtr[qtr]
                active = pb[mask] != 0
                n_active = int(active.sum())
                if n_active < 10:
                    continue
                rtrn = pb[mask][active]
                total_pnl = round(float(rtrn.sum()), 6)
                pos = rtrn[rtrn > 0].sum()
                neg = abs(rtrn[rtrn < 0].sum())
                pf = round(float(pos / max(neg, 1e-12)), 4)
                ar = round(float(rtrn.mean()), 8)
                sr = round(float(rtrn.mean() / max(rtrn.std(), 1e-10) * np.sqrt(ANNUAL_BARS)), 2)
                w.writerow([l, qtr, len(mask), n_active, total_pnl, pf, ar, sr])
            active_all = pb != 0
            if active_all.sum() > 10:
                ra = pb[active_all]
                pos = ra[ra > 0].sum()
                neg = abs(ra[ra < 0].sum())
                pf = round(float(pos / max(neg, 1e-12)), 4)
                w.writerow([l, "overall", len(pb), int(active_all.sum()),
                            round(float(ra.sum()), 6), pf,
                            round(float(ra.mean()), 8),
                            round(float(ra.mean() / max(ra.std(), 1e-10) * np.sqrt(ANNUAL_BARS)), 2)])

    print(f"Re-saved: CSV + quarterly with regime filter results")
