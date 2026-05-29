"""
Single-asset backtest: SBER predictions with 90/10 quantile thresholds.

Strategy: 
  - Compute mean predicted return across 5 MC samples per window
  - LONG when pred_ret > 90th percentile of all windows
  - SHORT when pred_ret < 10th percentile of all windows
  - Hold for pred_len=12 bars (~2h)
  - Per-bar return simulation with commission
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.evaluation.metrics import (
    sharpe_ratio, max_drawdown, profit_factor, win_rate, calmar_ratio,
    sortino_ratio, avg_return, n_trades, trade_pct, psr, dsharpe_ratio,
    direction_accuracy, return_correlation, ic_rank, bias, mae,
)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = "data/v3/predictions/10min_sber_mini"
RAW_DIR = "data/tickers/SBER"
LOOKBACK = 500
PRED_LEN = 12
Q_LONG = 0.90
Q_SHORT = 0.10
COMMISSION = 0.0006

# ── Load ────────────────────────────────────────────────────────────────────
preds = np.load(f"{DATA_DIR}/SBER_preds_pl12_sc5.npy")
belief = np.load(f"{DATA_DIR}/SBER_belief_pl12_sc5.npy")
raw = np.load(f"{RAW_DIR}/feats_test_raw.npy")
ts = np.load(f"{RAW_DIR}/timestamps_test_raw.npy", allow_pickle=True)

N = preds.shape[0]
assert N == raw.shape[0] - LOOKBACK - PRED_LEN + 1

# ── Compute returns ─────────────────────────────────────────────────────────
entry_prices = np.zeros(N)
pred_returns = np.zeros((N, 5))
actual_returns = np.zeros(N)

for i in range(N):
    entry_prices[i] = raw[i + LOOKBACK - 1, 3]
    ci = i + LOOKBACK + PRED_LEN - 1
    actual_returns[i] = (raw[ci, 3] - entry_prices[i]) / max(entry_prices[i], 1e-8)
    for s in range(5):
        pred_returns[i, s] = (preds[i, s, PRED_LEN - 1, 3] - entry_prices[i]) / max(entry_prices[i], 1e-8)

mean_pred_ret = pred_returns.mean(axis=1)
q90 = float(np.quantile(mean_pred_ret, Q_LONG))
q10 = float(np.quantile(mean_pred_ret, Q_SHORT))

signals = np.zeros(N, dtype=int)
signals[mean_pred_ret > q90] = 1
signals[mean_pred_ret < q10] = -1

# ── Per-bar PnL ─────────────────────────────────────────────────────────────
per_bar = np.zeros(len(raw))
for i in range(N):
    if signals[i] == 0:
        continue
    sig = signals[i]
    ep = raw[i + LOOKBACK - 1, 3]
    for step in range(PRED_LEN):
        bi = i + LOOKBACK + step
        if bi >= len(raw): break
        sr = sig * (raw[bi, 3] - ep) / max(ep, 1e-8)
        per_bar[bi] += sr / PRED_LEN

active_mask = per_bar != 0
per_bar[active_mask] -= COMMISSION / PRED_LEN

# ── Equity curve ────────────────────────────────────────────────────────────
cumul = np.cumprod(1 + per_bar)
total_ret = cumul[-1] - 1
active = per_bar[active_mask]

# ── Report ──────────────────────────────────────────────────────────────────
print(f"N_windows={N}  range={ts[LOOKBACK]}→{ts[LOOKBACK+N-1+PRED_LEN-1]}")
print(f"q90={q90:.6f}  q10={q10:.6f}")
print(f"Signals: LONG={int((signals==1).sum())} SHORT={int((signals==-1).sum())} "
      f"FLAT={N-int((signals!=0).sum())}")

print(f"\n{'='*60}")
print(f"  SBER BACKTEST — Q={Q_LONG*100:.0f}/{Q_SHORT*100:.0f} "
      f"pl={PRED_LEN} sc=5")
print(f"{'='*60}")
print(f"  Total return:       {total_ret:.4%}")
print(f"  Sharpe (ann):       {sharpe_ratio(per_bar):.4f}")
print(f"  Sortino (ann):      {sortino_ratio(per_bar):.4f}")
print(f"  Max drawdown:       {max_drawdown(cumul):.4%}")
print(f"  Win rate (active):  {win_rate(active):.2%}")
print(f"  Profit factor:      {profit_factor(active):.2f}")
print(f"  Calmar ratio:       {calmar_ratio(active, cumul):.4f}")
print(f"  PSR:                {psr(active):.4f}")
print(f"  DSR:                {dsharpe_ratio(active):.4f}")
print(f"  Active bars:        {active_mask.sum()} / {len(per_bar)}")
print(f"  Mean active ret:    {active.mean():.8f}")
print(f"  Std active ret:     {active.std():.8f}")

# ── Prediction quality ──────────────────────────────────────────────────────
da = direction_accuracy(actual_returns, mean_pred_ret)
rc = return_correlation(actual_returns, mean_pred_ret)
ic = ic_rank(actual_returns, mean_pred_ret)
print(f"\n{'─'*60}")
print(f"  PREDICTION QUALITY")
print(f"{'─'*60}")
print(f"  Direction accuracy: {da:.4%}")
print(f"  Return correlation: {rc:.4f}")
print(f"  IC (rank):          {ic:.4f}")
print(f"  Bias:               {bias(actual_returns, mean_pred_ret):.6f}")
print(f"  MAE:                {mae(actual_returns, mean_pred_ret):.6f}")

# ── Belief ──────────────────────────────────────────────────────────────────
conf = belief[:, :, :, 0].mean(axis=(1, 2))
print(f"\n{'─'*60}")
print(f"  BELIEF ANALYSIS")
print(f"{'─'*60}")
print(f"  Confidence: {conf.mean():.4f} ± {conf.std():.4f}")
print(f"  Entropy:    {belief[:,:,:,1].mean():.4f} ± {belief[:,:,:,1].std():.4f}")
print(f"  Top3 mass:  {belief[:,:,:,2].mean():.4f} ± {belief[:,:,:,2].std():.4f}")

for th in np.arange(0.15, 0.60, 0.05):
    mask = conf > th
    if mask.sum() > 30:
        ds = ic_rank(actual_returns[mask], mean_pred_ret[mask])
        wr = float(np.mean(np.sign(mean_pred_ret[mask]) == np.sign(actual_returns[mask])))
        print(f"    conf>{th:.2f} ({mask.sum():5d}w) IC={ds:.4f} dir_acc={wr:.2%}")

# ── Quality gates ───────────────────────────────────────────────────────────
sr_val = sharpe_ratio(per_bar)
wr_val = win_rate(active)
mdd_val = max_drawdown(cumul)
psr_val = psr(active)
ic_val = ic_rank(actual_returns, mean_pred_ret)

print(f"\n{'─'*60}")
print(f"  QUALITY GATES (config/global.yaml §backtest.quality_gates)")
print(f"{'─'*60}")
gates = [
    ("Sharpe ≥ 0.5",          sr_val, sr_val >= 0.5),
    ("Win rate ≥ 52%",        wr_val, wr_val >= 0.52),
    ("MaxDD ≤ 25%",          abs(mdd_val), abs(mdd_val) <= 0.25),
    ("PSR ≥ 0.7",             psr_val, psr_val >= 0.7),
    ("IC mean ≥ 0.02",        ic_val, ic_val >= 0.02),
]
for name, val, passed in gates:
    r = "✅" if passed else "❌"
    print(f"  {r} {name:<25s}  {val:.4f}")

# ── Save ────────────────────────────────────────────────────────────────────
data_dict = {
    "sharpe": float(sr_val), "sortino": float(sortino_ratio(per_bar)), "max_dd": float(mdd_val),
    "win_rate": float(wr_val), "profit_factor": float(profit_factor(active)),
    "calmar": float(calmar_ratio(active, cumul)), "psr": float(psr_val), "dsr": float(dsharpe_ratio(active)),
    "total_return": float(total_ret), "n_active_bars": int(active_mask.sum()),
    "n_long": int((signals==1).sum()), "n_short": int((signals==-1).sum()),
    "dir_acc": float(da), "ic_rank": float(ic), "return_corr": float(rc),
    "bias": float(bias(actual_returns, mean_pred_ret)), "mae": float(mae(actual_returns, mean_pred_ret)),
    "q90": float(q90), "q10": float(q10), "windows": N,
    "gates": {g[0]: {"value": float(g[1]), "passed": bool(g[2])} for g in gates},
}

with open("data/v3/results/sber_backtest_q90_q10.json", "w") as f:
    json.dump(data_dict, f, indent=2, default=str)
print(f"\nSaved: data/v3/results/sber_backtest_q90_q10.json")
