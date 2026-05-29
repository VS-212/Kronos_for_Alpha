"""
Сравнение: baseline vs confidence-фильтры для SBER backtest.
Тестирует, улучшает ли фильтрация по уверенности модели метрики.
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.evaluation.metrics import (
    sharpe_ratio, max_drawdown, profit_factor, win_rate, calmar_ratio,
    sortino_ratio, psr, direction_accuracy, ic_rank,
)

DATA_DIR = "data/v3/predictions/10min_sber_mini"
RAW_DIR = "data/tickers/SBER"
LK = 500; PL = 12; COMM = 0.0006

preds = np.load(f"{DATA_DIR}/SBER_preds_pl12_sc5.npy")
belief = np.load(f"{DATA_DIR}/SBER_belief_pl12_sc5.npy")
raw = np.load(f"{RAW_DIR}/feats_test_raw.npy")

N = preds.shape[0]
conf = belief[:, :, :, 0].mean(axis=(1, 2))  # mean confidence per window

# Per-window returns
entry_prices = np.zeros(N)
pred_ret = np.zeros(N)
actual_ret = np.zeros(N)
for i in range(N):
    ep = raw[i + LK - 1, 3]
    entry_prices[i] = ep
    pred_ret[i] = (preds[i, :, PL - 1, 3].mean() - ep) / max(ep, 1e-8)
    ci = min(i + LK + PL - 1, len(raw) - 1)
    actual_ret[i] = (raw[ci, 3] - ep) / max(ep, 1e-8)

# Quantile thresholds from ALL windows (baseline)
q90 = float(np.quantile(pred_ret, 0.90))
q10 = float(np.quantile(pred_ret, 0.10))

# ── Per-bar PnL для одного набора сигналов ──
def run_backtest(signals):
    per_bar = np.zeros(len(raw))
    for i in range(N):
        if signals[i] == 0:
            continue
        sig = signals[i]
        ep = raw[i + LK - 1, 3]
        for step in range(PL):
            bi = i + LK + step
            if bi >= len(raw): break
            per_bar[bi] += sig * (raw[bi, 3] - ep) / max(ep, 1e-8) / PL
    active = per_bar != 0
    per_bar[active] -= COMM / PL
    cumul = np.cumprod(1 + per_bar)
    tot = cumul[-1] - 1
    n_act = int(active.sum())
    act_ret = per_bar[active]
    return {
        "total_return": tot,
        "sharpe": sharpe_ratio(per_bar),
        "sortino": sortino_ratio(per_bar),
        "max_dd": max_drawdown(cumul),
        "win_rate": win_rate(act_ret) if len(act_ret) > 0 else 0,
        "profit_factor": profit_factor(act_ret) if len(act_ret) > 0 else 0,
        "psr": psr(act_ret) if len(act_ret) > 0 else 0,
        "n_active_bars": n_act,
        "n_long": int((signals == 1).sum()),
        "n_short": int((signals == -1).sum()),
        "n_flat": N - int((signals != 0).sum()),
    }

results = []

# ── Baseline: quantile 90/10, NO confidence filter ──
sig = np.zeros(N, dtype=int)
sig[pred_ret > q90] = 1
sig[pred_ret < q10] = -1
r = run_backtest(sig)
r["label"] = "baseline (no filter)"
r["n_windows"] = N
r["n_traded"] = int((sig != 0).sum())
results.append(r)

# ── С различными confidence thresholds ──
for th in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
    mask = conf >= th
    if mask.sum() < 50:
        continue
    # Recompute quantiles WITHIN the filtered windows
    local_q90 = float(np.quantile(pred_ret[mask], 0.90))
    local_q10 = float(np.quantile(pred_ret[mask], 0.10))
    sig = np.zeros(N, dtype=int)
    sig[(pred_ret > local_q90) & mask] = 1
    sig[(pred_ret < local_q10) & mask] = -1
    r = run_backtest(sig)
    r["label"] = f"conf>={th:.2f}"
    r["n_windows"] = int(mask.sum())
    r["n_traded"] = int((sig != 0).sum())
    r["conf_thresh"] = th
    results.append(r)

# ── Prediction quality ──
# Baseline dir_acc/IC на ВСЕХ окнах
da_all = direction_accuracy(actual_ret, pred_ret)
ic_all = ic_rank(actual_ret, pred_ret)
print(f"Baseline prediction quality (all {N} windows):")
print(f"  dir_acc = {da_all:.4%}")
print(f"  IC      = {ic_all:.4f}")
print()

# ── Вывод таблицы ──
print(f"{'='*90}")
h = f"  {'label':<22s} {'win_rate':>8s} {'sharpe':>8s} {'sortino':>8s} {'max_dd':>9s} {'ret':>8s} {'PF':>6s} {'PSR':>6s} {'n_trade':>9s} {'n_win':>6s}"
print(h)
print(f"{'='*90}")
for r in results:
    print(f"  {r['label']:<22s} {r['win_rate']:>8.2%} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} "
          f"{r['max_dd']:>9.2%} {r['total_return']:>8.2%} {r['profit_factor']:>6.3f} "
          f"{r['psr']:>6.3f} {r['n_traded']:>9d} {r['n_windows']:>6d}")

# ── Сравнение prediction quality c фильтром ──
print(f"\n{'─'*70}")
print(f"  PREDICTION QUALITY с confidence-фильтром")
print(f"{'─'*70}")
print(f"  {'threshold':>10s} {'windows':>8s} {'dir_acc':>8s} {'IC':>8s} {'IC_change':>10s}")
for th in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
    mask = conf >= th
    if mask.sum() < 30: continue
    da = direction_accuracy(actual_ret[mask], pred_ret[mask])
    ic = ic_rank(actual_ret[mask], pred_ret[mask])
    chg = ic / ic_all - 1 if ic_all != 0 else 0
    print(f"  {'conf>=' + str(th):>10s} {mask.sum():>8d} {da:>8.2%} {ic:>8.4f} {chg:>+10.1%}")

# ── Сохраняем результаты ──
os.makedirs("data/v3/results", exist_ok=True)
with open("data/v3/results/sber_confidence_comparison.json", "w") as f:
    json.dump([{k: float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v
                for k, v in r.items()} for r in results], f, indent=2, default=str)
print(f"\nSaved: data/v3/results/sber_confidence_comparison.json")
