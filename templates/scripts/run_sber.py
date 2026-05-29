"""
Orchestrator: SBER backtest using modular architecture.
Replaces backtest_sber_v2.py (1556 lines -> clean module imports).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from src.data.loader_sber import load_sber_data
from src.evaluation.engine import (
    PRED_METRICS,
    STRAT_METRICS,
    TRADE_METRICS,
    run_backtest,
    run_backtest_custom,
)
from src.evaluation.quarterly import (
    compute_h1_2026_metrics,
    compute_quarterly_tables,
    save_all_results,
    save_quarterly_csv,
    save_results_csv,
)
from src.evaluation.simulation import (
    _weighted_q,
    get_tp_sl,
    get_tp_sl_no_sl,
    get_tp_sl_no_tp,
    get_tp_sl_w,
)
from src.signals.filters import (
    compute_atr_filter,
    compute_bb,
    compute_bb_momentum,
    compute_conf_trend,
    compute_lr,
    compute_mc_breadth,
    compute_pred_z,
    compute_roll_wr,
    compute_rsi14,
    compute_volume_filter,
)

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
data = load_sber_data()
LK = 500; PL = 12; COMM = 0.0
Q_LONG = 0.90; Q_SHORT = 0.10
TP_Q = 0.80; SL_Q = 0.20
data.update(LK=LK, PL=PL, COMM=COMM, Q_LONG=Q_LONG, Q_SHORT=Q_SHORT, TP_Q=TP_Q, SL_Q=SL_Q)

N = data['N']; raw = data['raw']; ts = data['ts']
preds = data['preds']; belief = data['belief']; conf = data['conf']
entry_close = data['entry_close']; entry_open = data['entry_open']
pred_ret = data['pred_ret']; actual_ret = data['actual_ret']
g_tp = data['g_tp']; g_sl = data['g_sl']
wf_q90 = data['wf_q90']; wf_q10 = data['wf_q10']
close_arr = data['close_arr']; conf_per_mc = data['conf_per_mc']

pred_close_horizon = preds[:, :, PL - 1, 3]
data['pred_close_horizon'] = pred_close_horizon

results = []
per_bars = []

# ── Helper: run & collect ───────────────────────────────────────────────────
def run_custom(sig, name, fn=get_tp_sl, ref=None, verbose=True):
    r, pb = run_backtest_custom(sig, name, fn, data, pred_ret_ref=ref if ref is not None else pred_ret, verbose=verbose)
    results.append(r); per_bars.append(pb)

def run_std(sig, name, verbose=True):
    r, pb = run_backtest(sig, name, data, verbose=verbose)
    results.append(r); per_bars.append(pb)

print(f"\n{'='*90}")
print(f"  SBER BACKTEST — Q={Q_LONG*100:.0f}/{Q_SHORT*100:.0f}  pl={PL}  sc=5  COMM={COMM}")
print(f"  TP/SL: q{TP_Q*100:.0f}/q{SL_Q*100:.0f} from MC dist at horizon")
print(f"{'='*90}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. WF baseline
# ═══════════════════════════════════════════════════════════════════════════
sig_wf = np.zeros(N, dtype=int)
for i in range(N):
    if pred_ret[i] > wf_q90[i]:
        sig_wf[i] = 1
    elif pred_ret[i] < wf_q10[i]:
        sig_wf[i] = -1
run_std(sig_wf, "wf (FIX)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. WF + confidence thresholds
# ═══════════════════════════════════════════════════════════════════════════
CONF_THRESHOLDS = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
for th in CONF_THRESHOLDS:
    mask = conf >= th
    if mask.sum() < 30:
        continue
    sig = np.zeros(N, dtype=int)
    for i in range(N):
        if not mask[i]:
            continue
        if pred_ret[i] > wf_q90[i]:
            sig[i] = 1
        elif pred_ret[i] < wf_q10[i]:
            sig[i] = -1
    run_std(sig, f"wf+conf≥{th:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# H1: MC agreement (consensus)
# ═══════════════════════════════════════════════════════════════════════════
pred_close_mc = preds[:, :, PL - 1, 3]
pred_dir_mc = np.sign(pred_close_mc - entry_close[:, None])
mc_agreement = np.abs(pred_dir_mc.sum(axis=1))
pred_ret_mc_arr = (pred_close_mc - entry_close[:, None]) / np.maximum(entry_close[:, None], 1e-8)

for min_agree in [3, 5]:
    sig = np.zeros(N, dtype=int)
    for i in range(N):
        if mc_agreement[i] < min_agree:
            continue
        if pred_ret[i] > wf_q90[i]:
            sig[i] = 1
        elif pred_ret[i] < wf_q10[i]:
            sig[i] = -1
    run_std(sig, f"H1: MC agree≥{min_agree}")

# ═══════════════════════════════════════════════════════════════════════════
# H2: Conf-weighted pred_ret + weighted TP/SL
# ═══════════════════════════════════════════════════════════════════════════
conf_weights = conf_per_mc / (conf_per_mc.sum(axis=1, keepdims=True) + 1e-8)
pred_ret_w = np.sum(pred_ret_mc_arr * conf_weights, axis=1)

wf_q90_w = np.zeros(N)
wf_q10_w = np.zeros(N)
for i in range(N):
    if i < 100:
        wf_q90_w[i] = float(np.quantile(pred_ret_w[:i+1], Q_LONG)) if i > 10 else 0.001
        wf_q10_w[i] = float(np.quantile(pred_ret_w[:i+1], Q_SHORT)) if i > 10 else -0.001
    else:
        wf_q90_w[i] = float(np.quantile(pred_ret_w[:i], Q_LONG))
        wf_q10_w[i] = float(np.quantile(pred_ret_w[:i], Q_SHORT))

g_tp_w = _weighted_q(pred_close_mc, conf_weights, TP_Q)
g_sl_w = _weighted_q(pred_close_mc, conf_weights, SL_Q)
data['g_tp_w'] = g_tp_w; data['g_sl_w'] = g_sl_w

sig_w = np.zeros(N, dtype=int)
sig_w[pred_ret_w > wf_q90_w] = 1
sig_w[pred_ret_w < wf_q10_w] = -1
# Wrap get_tp_sl_w to use weighted TP/SL from data dict
def _get_tp_sl_w_wrapped(i, sig, ec, _gt, _gs):
    return get_tp_sl_w(i, sig, ec, data['g_tp_w'], data['g_sl_w'])
run_custom(sig_w, "H2: weighted by conf", _get_tp_sl_w_wrapped, ref=pred_ret_w)

# ═══════════════════════════════════════════════════════════════════════════
# V: TP/SL asymmetry ratio
# ═══════════════════════════════════════════════════════════════════════════
can_long = g_tp > entry_close
can_short = g_sl < entry_close

for rr_th in [1.0, 1.5, 2.0, 3.0, 5.0]:
    sig = np.zeros(N, dtype=int)
    for i in range(N):
        if can_long[i] and (g_tp[i] - entry_close[i]) > rr_th * (entry_close[i] - g_sl[i]):
            sig[i] = 1
        elif can_short[i] and (entry_close[i] - g_sl[i]) > rr_th * (g_tp[i] - entry_close[i]):
            sig[i] = -1
    run_std(sig, f"V: TP/SL ratio>{rr_th:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# V+WF: ratio>3.0 + close momentum
# ═══════════════════════════════════════════════════════════════════════════
close_prev = raw[LK - 2: LK - 2 + N, 3]
close_last = entry_close
close_rise = close_last > close_prev

for rr_th in [3.0]:
    sig = np.zeros(N, dtype=int)
    for i in range(N):
        if can_long[i] and close_rise[i] and (g_tp[i] - entry_close[i]) > rr_th * (entry_close[i] - g_sl[i]):
            sig[i] = 1
        elif can_short[i] and not close_rise[i] and (entry_close[i] - g_sl[i]) > rr_th * (g_tp[i] - entry_close[i]):
            sig[i] = -1
    run_std(sig, f"V+WF: ratio>{rr_th:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# H4: Best MC per window
# ═══════════════════════════════════════════════════════════════════════════
best_mc_idx = conf_per_mc.argmax(axis=1)
pred_ret_best = pred_ret_mc_arr[np.arange(N), best_mc_idx]

wf_q90_best = np.zeros(N)
wf_q10_best = np.zeros(N)
for i in range(N):
    if i < 100:
        wf_q90_best[i] = float(np.quantile(pred_ret_best[:i+1], Q_LONG)) if i > 10 else 0.001
        wf_q10_best[i] = float(np.quantile(pred_ret_best[:i+1], Q_SHORT)) if i > 10 else -0.001
    else:
        wf_q90_best[i] = float(np.quantile(pred_ret_best[:i], Q_LONG))
        wf_q10_best[i] = float(np.quantile(pred_ret_best[:i], Q_SHORT))

sig_best = np.zeros(N, dtype=int)
sig_best[pred_ret_best > wf_q90_best] = 1
sig_best[pred_ret_best < wf_q10_best] = -1
run_custom(sig_best, "H4: best MC", get_tp_sl, ref=pred_ret_best)

# ═══════════════════════════════════════════════════════════════════════════
# H5: Drop low-confidence MC
# ═══════════════════════════════════════════════════════════════════════════
for drop_th in [0.4, 0.6, 0.8, 1.0]:
    mask_good = conf_per_mc > drop_th
    n_good = mask_good.sum(axis=1)
    pred_ret_drop = np.where(n_good > 0,
        (pred_ret_mc_arr * mask_good).sum(axis=1) / n_good,
        pred_ret_mc_arr.mean(axis=1))

    wf_q90_d = np.zeros(N)
    wf_q10_d = np.zeros(N)
    for i in range(N):
        if i < 100:
            wf_q90_d[i] = float(np.quantile(pred_ret_drop[:i+1], Q_LONG)) if i > 10 else 0.001
            wf_q10_d[i] = float(np.quantile(pred_ret_drop[:i+1], Q_SHORT)) if i > 10 else -0.001
        else:
            wf_q90_d[i] = float(np.quantile(pred_ret_drop[:i], Q_LONG))
            wf_q10_d[i] = float(np.quantile(pred_ret_drop[:i], Q_SHORT))

    sig_d = np.zeros(N, dtype=int)
    sig_d[pred_ret_drop > wf_q90_d] = 1
    sig_d[pred_ret_drop < wf_q10_d] = -1
    run_custom(sig_d, f"H5: drop MC<{drop_th:.1f} conf", get_tp_sl, ref=pred_ret_drop)

# ═══════════════════════════════════════════════════════════════════════════
# H1xV: MC agree>=5 + ratio>3.0
# ═══════════════════════════════════════════════════════════════════════════
sig_h5 = np.zeros(N, dtype=int)
for i in range(N):
    if mc_agreement[i] < 5:
        continue
    if pred_ret[i] > wf_q90[i]:
        sig_h5[i] = 1
    elif pred_ret[i] < wf_q10[i]:
        sig_h5[i] = -1

sig_h5v = np.zeros(N, dtype=int)
for i in range(N):
    if sig_h5[i] == 0:
        continue
    if sig_h5[i] == 1 and can_long[i] and (g_tp[i] - entry_close[i]) > 3.0 * (entry_close[i] - g_sl[i]):
        sig_h5v[i] = 1
    if sig_h5[i] == -1 and can_short[i] and (entry_close[i] - g_sl[i]) > 3.0 * (g_tp[i] - entry_close[i]):
        sig_h5v[i] = -1
run_std(sig_h5v, "H1×V: agree≥5 + ratio>3.0")

# ═══════════════════════════════════════════════════════════════════════════
# BOLLINGER BANDS
# ═══════════════════════════════════════════════════════════════════════════
bb_mid, bb_std, bb_upper, bb_lower, bb_width, bb_pct, wf_width, \
    bb_band_touch_lower, bb_band_touch_upper = compute_bb(data)

BB_PERIOD = 20; BB_K = 2.0

# BB filters
bb_band_touch_lower = entry_close <= bb_lower + 0.5 * bb_std
bb_band_touch_upper = entry_close >= bb_upper - 0.5 * bb_std

# ── Base signals for BB combination testing ────────────────────────────────
base_signals = {}
base_signals["WF"] = sig_wf.copy()

# H1>=5 (from above)
base_signals["H1≥5"] = sig_h5.copy()

# H5<0.4 (recompute)
mask_good = conf_per_mc > 0.4
n_good = mask_good.sum(axis=1)
pred_ret_d40 = np.where(n_good > 0,
    (pred_ret_mc_arr * mask_good).sum(axis=1) / n_good,
    pred_ret_mc_arr.mean(axis=1))
wf_q90_d40 = np.zeros(N); wf_q10_d40 = np.zeros(N)
for i in range(N):
    if i < 100:
        wf_q90_d40[i] = float(np.quantile(pred_ret_d40[:i+1], Q_LONG)) if i > 10 else 0.001
        wf_q10_d40[i] = float(np.quantile(pred_ret_d40[:i+1], Q_SHORT)) if i > 10 else -0.001
    else:
        wf_q90_d40[i] = float(np.quantile(pred_ret_d40[:i], Q_LONG))
        wf_q10_d40[i] = float(np.quantile(pred_ret_d40[:i], Q_SHORT))
sig_h5_d40 = np.zeros(N, dtype=int)
sig_h5_d40[pred_ret_d40 > wf_q90_d40] = 1
sig_h5_d40[pred_ret_d40 < wf_q10_d40] = -1
base_signals["H5<0.4"] = sig_h5_d40.copy()

# V>3.0 (recompute)
sig_v3 = np.zeros(N, dtype=int)
for i in range(N):
    if can_long[i] and (g_tp[i] - entry_close[i]) > 3.0 * (entry_close[i] - g_sl[i]):
        sig_v3[i] = 1
    elif can_short[i] and (entry_close[i] - g_sl[i]) > 3.0 * (g_tp[i] - entry_close[i]):
        sig_v3[i] = -1
base_signals["V>3.0"] = sig_v3.copy()

# ── BB filter applications ─────────────────────────────────────────────────
BB_FILTERS = [
    ("width", lambda i, sig: bb_width[i] >= wf_width[i]),
    ("%B",    lambda i, sig: (sig == 1 and bb_pct[i] <= 0.3) or (sig == -1 and bb_pct[i] >= 0.7)),
    ("touch", lambda i, sig: (sig == 1 and bb_band_touch_lower[i]) or (sig == -1 and bb_band_touch_upper[i])),
]

for base_name, base_sig in base_signals.items():
    for bb_label, _bb_check in BB_FILTERS:
        sig = np.zeros(N, dtype=int)
        for i in range(N):
            if base_sig[i] == 0:
                continue
            if bb_check(i, base_sig[i]):
                sig[i] = base_sig[i]
        name = f"{base_name}+BB{bb_label}"
        run_std(sig, name)

# ═══════════════════════════════════════════════════════════════════════════
# NO-SL strategies
# ═══════════════════════════════════════════════════════════════════════════
NO_SL_STRATS = [
    ("WF+BBwidth",       base_signals["WF"],          True),
    ("H1≥5+BBwidth",     base_signals["H1≥5"],        True),
    ("H5<0.4+BBwidth",   base_signals["H5<0.4"],      True),
    ("V>3.0+BBwidth",    base_signals["V>3.0"],       True),
    ("V>3.0",            base_signals["V>3.0"],       False),
]

for label, base_sig, use_bb in NO_SL_STRATS:
    sig = base_sig.copy()
    if use_bb:
        sig[bb_width < wf_width] = 0
    run_custom(sig, f"{label} noSL", get_tp_sl_no_sl)

# ═══════════════════════════════════════════════════════════════════════════
# NO-TP strategies
# ═══════════════════════════════════════════════════════════════════════════
NO_TP_STRATS = [
    ("WF",           base_signals["WF"],          False),
    ("H5<0.4",       base_signals["H5<0.4"],      False),
    ("H1≥5",         base_signals["H1≥5"],        False),
    ("V>3.0",        base_signals["V>3.0"],       False),
    ("WF+BBwidth",   base_signals["WF"],          True),
    ("H5<0.4+BBwidth", base_signals["H5<0.4"],    True),
    ("H1≥5+BBwidth", base_signals["H1≥5"],        True),
    ("V>3.0+BBwidth", base_signals["V>3.0"],      True),
    ("WF+BB%B",      base_signals["WF"],          False),
    ("V>3.0+BB%B",   base_signals["V>3.0"],       False),
]

for label, base_sig, _use_bb in NO_TP_STRATS:
    sig = base_sig.copy()
    if "BBwidth" in label:
        sig[bb_width < wf_width] = 0
    elif "BB%B" in label:
        for i in range(N):
            if sig[i] == 0:
                continue
            if (sig[i] == 1 and bb_pct[i] > 0.3) or (sig[i] == -1 and bb_pct[i] < 0.7):
                sig[i] = 0
    run_custom(sig, f"{label} noTP", get_tp_sl_no_tp)

# ═══════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
lr_slope, lr_r2, lr_mid, lr_std, lr_upper, lr_lower, lr_channel_pct, wf_r2 = compute_lr(data)

LR_BASE = [
    ("WF",     base_signals["WF"]),
    ("H5<0.4", base_signals["H5<0.4"]),
    ("H1≥5",   base_signals["H1≥5"]),
    ("V>3.0",  base_signals["V>3.0"]),
]

for base_name, base_sig in LR_BASE:
    sig = base_sig.copy()
    sig[lr_r2 < wf_r2] = 0
    run_custom(sig, f"{base_name}+LRr2 noTP", get_tp_sl_no_tp)

    sig = base_sig.copy()
    for i in range(N):
        if sig[i] == 0:
            continue
        if (sig[i] == 1 and lr_channel_pct[i] > 0.3) or (sig[i] == -1 and lr_channel_pct[i] < 0.7):
            sig[i] = 0
    run_custom(sig, f"{base_name}+LR%B noTP", get_tp_sl_no_tp)

    sig = base_sig.copy()
    sig[lr_r2 < wf_r2] = 0
    for i in range(N):
        if sig[i] == 0:
            continue
        if (sig[i] == 1 and lr_channel_pct[i] > 0.3) or (sig[i] == -1 and lr_channel_pct[i] < 0.7):
            sig[i] = 0
    run_custom(sig, f"{base_name}+LRr2%B noTP", get_tp_sl_no_tp)

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: ATR, Volume, BB Momentum
# ═══════════════════════════════════════════════════════════════════════════
atr, wf_atr = compute_atr_filter(data)
vol_at_close, wf_vol = compute_volume_filter(data)
bb_mom_arr = compute_bb_momentum(bb_width)

TIER1_BASE = [
    ("WF",     base_signals["WF"]),
    ("H5<0.4", base_signals["H5<0.4"]),
    ("H1≥5",   base_signals["H1≥5"]),
    ("V>3.0",  base_signals["V>3.0"]),
]

for base_name, base_sig in TIER1_BASE:
    sig_a = base_sig.copy(); sig_a[atr < wf_atr] = 0
    run_custom(sig_a, f"{base_name}+ATR noTP", get_tp_sl_no_tp)

    sig_v = base_sig.copy(); sig_v[vol_at_close < wf_vol] = 0
    run_custom(sig_v, f"{base_name}+vol noTP", get_tp_sl_no_tp)

    sig_b = base_sig.copy(); sig_b[~bb_mom_arr] = 0
    run_custom(sig_b, f"{base_name}+BBmom noTP", get_tp_sl_no_tp)

# ── Combos on best BB strategies ──────────────────────────────────────────
BB_BEST = [
    ("WF+BB%B",     base_signals["WF"]),
    ("H5<0.4+BBwidth", base_signals["H5<0.4"]),
]

for label, base_sig in BB_BEST:
    for bb_label, _bb_check in BB_FILTERS:
        if bb_label == "width" and "width" in label:
            sig = base_sig.copy()
            sig[bb_width < wf_width] = 0
            sub_label = label
            break
        elif bb_label == "%B" and "%B" in label:
            sig = base_sig.copy()
            for i in range(N):
                if sig[i] == 0: continue
                if (sig[i] == 1 and bb_pct[i] > 0.3) or (sig[i] == -1 and bb_pct[i] < 0.7):
                    sig[i] = 0
            sub_label = label
            break
    else:
        continue

    s = sig.copy(); s[atr < wf_atr] = 0
    run_custom(s, f"{sub_label}+ATR noTP", get_tp_sl_no_tp)

    s = sig.copy(); s[vol_at_close < wf_vol] = 0
    run_custom(s, f"{sub_label}+vol noTP", get_tp_sl_no_tp)

    s = sig.copy(); s[~bb_mom_arr] = 0
    run_custom(s, f"{sub_label}+BBmom noTP", get_tp_sl_no_tp)

    s = sig.copy(); s[atr < wf_atr] = 0; s[vol_at_close < wf_vol] = 0
    run_custom(s, f"{sub_label}+ATR+vol noTP", get_tp_sl_no_tp)

# ── Top5 + Volume filter ──────────────────────────────────────────────────
TOP5_VOL = [
    ("WF+BB%B+BBmom",    base_signals["WF"]),
    ("H5<0.4+BBwidth+BBmom", base_signals["H5<0.4"]),
    ("H5<0.4+BBmom",     base_signals["H5<0.4"]),
    ("WF+BBmom",         base_signals["WF"]),
    ("WF+BB%B",          base_signals["WF"]),
]

for label, base_sig in TOP5_VOL:
    sig = base_sig.copy()
    for i in range(N):
        if sig[i] == 0:
            continue
        if "BB%B" in label and ((sig[i] == 1 and bb_pct[i] > 0.3) or (sig[i] == -1 and bb_pct[i] < 0.7)):
            sig[i] = 0
        if "BBwidth" in label and bb_width[i] < wf_width[i]:
            sig[i] = 0
        if "BBmom" in label and not bb_mom_arr[i]:
            sig[i] = 0
        if vol_at_close[i] < wf_vol[i]:
            sig[i] = 0
    run_custom(sig, f"{label}+vol noTP", get_tp_sl_no_tp)

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: Conf trend, MC breadth, RSI, pred_z, rolling WR
# ═══════════════════════════════════════════════════════════════════════════

# ── MC confidence trend ──────────────────────────────────────────────────────
conf_trend = compute_conf_trend(data)

# ── MC path dispersion ──────────────────────────────────────────────────────
mc_std, mc_small, mc_big = compute_mc_breadth(data)

# ── RSI(14) ──────────────────────────────────────────────────────────────────
rsi14 = compute_rsi14(data)

# ── Pred_ret z-score ─────────────────────────────────────────────────────────
pred_z = compute_pred_z(data)

# ── Rolling WR ───────────────────────────────────────────────────────────────
data['wf_signal'] = sig_wf.copy()
roll_wr_ok = compute_roll_wr(data)

# ── Apply Tier 2 filters to WF+BB%B+BBmom (best strategy) ──────────────────
T2_BASE_SIG = base_signals["WF"]
t2_sig_ref = {}
for i in range(N):
    sig = T2_BASE_SIG[i]
    if sig == 0: continue
    if (sig == 1 and bb_pct[i] > 0.3) or (sig == -1 and bb_pct[i] < 0.7):
        sig = 0
    if not bb_mom_arr[i]:
        sig = 0
    t2_sig_ref[i] = sig

def apply_t2_to_sig(t2_filter, label_suffix):
    sig = np.zeros(N, dtype=np.int32)
    for i, v in t2_sig_ref.items():
        if v == 0: continue
        if t2_filter(i):
            sig[i] = v
    run_custom(sig, f"WF+BB%B+BBmom+{label_suffix} noTP", get_tp_sl_no_tp)

apply_t2_to_sig(lambda i: conf_trend[i], "confTrend")
apply_t2_to_sig(lambda i: mc_small[i], "mcSmall")
apply_t2_to_sig(lambda i: mc_big[i], "mcBig")
apply_t2_to_sig(lambda i: not ((t2_sig_ref[i] == 1 and rsi14[i] > 70) or (t2_sig_ref[i] == -1 and rsi14[i] < 30)), "rsi14")
apply_t2_to_sig(lambda i: abs(pred_z[i]) > 1.0, "zGt1")
apply_t2_to_sig(lambda i: roll_wr_ok[i], "rollWR")
apply_t2_to_sig(lambda i: (t2_sig_ref[i] == 1 and rsi14[i] < 30) or (t2_sig_ref[i] == -1 and rsi14[i] > 70), "rsiExtreme")

# ── T2 combinations ─────────────────────────────────────────────────────────
sig = np.zeros(N, dtype=np.int32)
for i, v in t2_sig_ref.items():
    if v == 0: continue
    if conf_trend[i] and mc_small[i]:
        sig[i] = v
run_custom(sig, "WF+BB%B+BBmom+confTrend+mcSmall noTP", get_tp_sl_no_tp)

sig = np.zeros(N, dtype=np.int32)
for i, v in t2_sig_ref.items():
    if v == 0: continue
    skip_rsi = (v == 1 and rsi14[i] > 70) or (v == -1 and rsi14[i] < 30)
    if abs(pred_z[i]) > 1.0 and not skip_rsi:
        sig[i] = v
run_custom(sig, "WF+BB%B+BBmom+zGt1+rsi14 noTP", get_tp_sl_no_tp)

sig = np.zeros(N, dtype=np.int32)
for i, v in t2_sig_ref.items():
    if v == 0: continue
    skip_rsi = (v == 1 and rsi14[i] > 70) or (v == -1 and rsi14[i] < 30)
    if conf_trend[i] and mc_small[i] and not skip_rsi:
        sig[i] = v
run_custom(sig, "WF+BB%B+BBmom+confTrend+mcSmall+rsi14 noTP", get_tp_sl_no_tp)

# ── Apply T2 filters to H5<0.4+BBwidth+BBmom (2nd best) ────────────────────
T2_BASE_SIG2 = base_signals["H5<0.4"]
t2_sig_ref2 = {}
for i in range(N):
    sig = T2_BASE_SIG2[i]
    if sig == 0: continue
    if bb_width[i] < wf_width[i]: sig = 0
    if not bb_mom_arr[i]: sig = 0
    t2_sig_ref2[i] = sig

sig = np.zeros(N, dtype=np.int32)
for i, v in t2_sig_ref2.items():
    if v == 0: continue
    if conf_trend[i]:
        sig[i] = v
run_custom(sig, "H5+BBwidth+BBmom+confTrend noTP", get_tp_sl_no_tp)

sig = np.zeros(N, dtype=np.int32)
for i, v in t2_sig_ref2.items():
    if v == 0: continue
    if mc_small[i]:
        sig[i] = v
run_custom(sig, "H5+BBwidth+BBmom+mcSmall noTP", get_tp_sl_no_tp)

sig = np.zeros(N, dtype=np.int32)
for i, v in t2_sig_ref2.items():
    if v == 0: continue
    skip_rsi = (v == 1 and rsi14[i] > 70) or (v == -1 and rsi14[i] < 30)
    if not skip_rsi and conf_trend[i] and mc_small[i]:
        sig[i] = v
run_custom(sig, "H5+BBwidth+BBmom+confTrend+mcSmall+rsi14 noTP", get_tp_sl_no_tp)

# ═══════════════════════════════════════════════════════════════════════════
# METRIC TABLES
# ═══════════════════════════════════════════════════════════════════════════
strat_keys = [k for k, _, _ in STRAT_METRICS]
trade_keys = [k for k, _, _ in TRADE_METRICS]
all_keys = strat_keys + trade_keys

print(f"\n  {'─'*80}")
print(f"  STRATEGY METRICS")
print(f"  {'─'*80}")
labels = [r["label"] for r in results]
col_w = max(len(l) for l in labels) + 2
print(f"  {'':<{col_w}s}", end="")
print("  ".join(f"{k:<15s}" for k in all_keys))
print(f"  {'':<{col_w}s}{'─'* (len(all_keys) * 17 - 1)}")

for r, l in zip(results, labels):
    print(f"  {l:<{col_w}s}", end="")
    for k in all_keys:
        v = r.get(k, 0)
        if k in ("n_trades", "n_long", "n_short", "n_bar_active"):
            s = f"{int(v):<15d}"
        elif isinstance(v, float) and v != v:
            s = f"{'nan':<15}"
        elif isinstance(v, float) and k in ("psr", "dsr", "sharpe", "sortino", "calmar", "avg_return", "total_return", "return_corr", "ic_rank", "bias", "mae", "dir_sharpe", "profit_factor"):
            s = f"{v:<15.4f}"
        elif isinstance(v, float):
            s = f"{v:<15.4%}"
        else:
            s = f"{v:<15}"
        print("  " + s, end="")
    print()

# ── Prediction quality ──────────────────────────────────────────────────────
print(f"\n  {'─'*60}")
print(f"  PREDICTION QUALITY (all {N} windows)")
print(f"  {'─'*60}")
for k, n, _ in PRED_METRICS:
    v = results[0][k]
    if k == "dir_acc":
        print(f"  {n:<20s}  {v:.4%}")
    elif k in ("bias", "mae"):
        print(f"  {n:<20s}  {v:.6f}")
    else:
        print(f"  {n:<20s}  {v:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# QUARTERLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════
bars_per_qtr, sorted_qtrs = compute_quarterly_tables(labels, per_bars, ts)

# ── Save ────────────────────────────────────────────────────────────────────
for r in results:
    r["q90_wf"] = float(wf_q90[-1]); r["q10_wf"] = float(wf_q10[-1])

os.makedirs("data/v3/results", exist_ok=True)
with open("data/v3/results/sber_backtest_v2.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: data/v3/results/sber_backtest_v2.json")

save_results_csv(results, labels)
save_quarterly_csv(labels, per_bars, bars_per_qtr, sorted_qtrs, ts)

# ═══════════════════════════════════════════════════════════════════════════
# H1 2026 metrics
# ═══════════════════════════════════════════════════════════════════════════
h1_metrics = compute_h1_2026_metrics(labels, per_bars, bars_per_qtr, sorted_qtrs, ts)

# ═══════════════════════════════════════════════════════════════════════════
# CHAMPION BREAKDOWN (conditional)
# ═══════════════════════════════════════════════════════════════════════════
CHAMPION_LABEL = "WF+BB%B+BBmom+rollWR noTP"
idx = labels.index(CHAMPION_LABEL) if CHAMPION_LABEL in labels else -1

if idx >= 0:
    pb_champ = per_bars[idx]
    from datetime import datetime
    bar_hour = np.zeros(len(raw), dtype=int)
    bar_dow = np.zeros(len(raw), dtype=int)
    for bi, t in enumerate(ts):
        dt = datetime.fromisoformat(str(t))
        bar_hour[bi] = dt.hour
        bar_dow[bi] = dt.weekday()

    h1_2026_mask = np.zeros(len(raw), dtype=bool)
    for qtr in sorted_qtrs:
        if qtr in ("2026-Q1", "2026-Q2"):
            indices = bars_per_qtr[qtr]
            h1_2026_mask[indices[indices < len(raw)]] = True

    active_h1 = h1_2026_mask & (pb_champ != 0)
    bar_returns = pb_champ[active_h1]
    bar_hours = bar_hour[active_h1]
    bar_dows = bar_dow[active_h1]

    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    # Hour breakdown
    print(f"\n  {'═'*80}")
    print(f"  {CHAMPION_LABEL} — H1 2026 by hour")
    print(f"  {'═'*80}")
    print(f"  {'Hour':<6} {'Bars':<7} {'TotRet':<9} {'PF':<8} {'AverRet':<10}")
    print(f"  {'─'*40}")
    for hr in sorted(set(bar_hours)):
        mask = bar_hours == hr
        n = int(mask.sum())
        if n < 5: continue
        rr = bar_returns[mask]
        tot = float(rr.sum())
        pos = rr[rr > 0].sum()
        neg = abs(rr[rr < 0].sum())
        pf = float(pos / max(neg, 1e-12))
        ar = float(rr.mean())
        print(f"  {hr:>2}h   {n:<7} {tot:>7.4%}  {pf:>6.4f}  {ar:>8.5%}")

    # DOW breakdown
    print(f"\n  {'═'*80}")
    print(f"  {CHAMPION_LABEL} — H1 2026 by day of week")
    print(f"  {'═'*80}")
    print(f"  {'Day':<6} {'Bars':<7} {'TotRet':<9} {'PF':<8} {'AverRet':<10}")
    print(f"  {'─'*40}")
    for dw in range(5):
        mask = bar_dows == dw
        n = int(mask.sum())
        if n < 5: continue
        rr = bar_returns[mask]
        tot = float(rr.sum())
        pos = rr[rr > 0].sum()
        neg = abs(rr[rr < 0].sum())
        pf = float(pos / max(neg, 1e-12))
        ar = float(rr.mean())
        print(f"  {DOW_NAMES[dw]:<6} {n:<7} {tot:>7.4%}  {pf:>6.4f}  {ar:>8.5%}")

    # Regime breakdown
    raw_bb_width = np.zeros(len(raw))
    raw_bb_pct = np.zeros(len(raw))
    raw_wf_width = np.zeros(len(raw))
    raw_bb_mom = np.zeros(len(raw), dtype=bool)
    for bi in range(len(raw)):
        wi = bi - LK
        if 0 <= wi < N:
            raw_bb_width[bi] = bb_width[wi]
            raw_bb_pct[bi] = bb_pct[wi]
            raw_wf_width[bi] = wf_width[wi]
            raw_bb_mom[bi] = bb_mom_arr[wi]
        else:
            raw_bb_width[bi] = bb_width[0]
            raw_bb_pct[bi] = 0.5
            raw_wf_width[bi] = wf_width[0]
            raw_bb_mom[bi] = True

    regimes = {}
    for bi in np.where(active_h1)[0]:
        rr = pb_champ[bi]
        vol_reg = "high_vol" if raw_bb_width[bi] >= raw_wf_width[bi] else "low_vol"
        zone = "oversold" if raw_bb_pct[bi] < 0.3 else ("overbought" if raw_bb_pct[bi] > 0.7 else "middle")
        mom = "expanding" if raw_bb_mom[bi] else "contracting"
        reg = f"{vol_reg}_{zone}_{mom}"
        regimes.setdefault(reg, []).append(rr)

    print(f"\n  {'═'*100}")
    print(f"  {CHAMPION_LABEL} — H1 2026 by BB regime")
    print(f"  {'═'*100}")
    print(f"  {'Regime':<35} {'Bars':<7} {'TotRet':<10} {'PF':<8} {'AverRet':<10}")
    print(f"  {'─'*70}")
    for reg in sorted(regimes.keys(), key=lambda k: -abs(sum(regimes[k]))):
        rr = np.array(regimes[reg])
        n = len(rr)
        if n < 5: continue
        tot = float(rr.sum())
        pos = rr[rr > 0].sum()
        neg = abs(rr[rr < 0].sum())
        pf = float(pos / max(neg, 1e-12))
        ar = float(rr.mean())
        print(f"  {reg:<35} {n:<7} {tot:>8.4%}  {pf:>6.4f}  {ar:>8.5%}")

    # Regime masks
    reg_bad = np.zeros(N, dtype=bool)
    reg_best = np.zeros(N, dtype=bool)
    for i in range(N):
        pos = i + LK - 1
        if pos >= len(raw): continue
        hv = raw_bb_width[pos] >= raw_wf_width[pos]
        os = raw_bb_pct[pos] < 0.3
        ex = raw_bb_mom[pos]
        md = not os and raw_bb_pct[pos] < 0.7
        reg_bad[i] = hv and os and ex
        reg_best[i] = hv and md and not ex

    # ── Apply_filters helper (uses local vars, matches monolith) ──────────────
    def apply_filters_monolith(sig, has_bbpct, has_bbwidth, has_bbmom, has_conftrend,
                               has_rsiextreme, has_vol, has_mcbig):
        out = sig.copy()
        for i in range(N):
            if out[i] == 0: continue
            if has_bbpct and ((out[i] == 1 and bb_pct[i] > 0.3) or (out[i] == -1 and bb_pct[i] < 0.7)):
                out[i] = 0
            if has_bbwidth and bb_width[i] < wf_width[i]:
                out[i] = 0
            if has_bbmom and not bb_mom_arr[i]:
                out[i] = 0
            if has_conftrend and not conf_trend[i]:
                out[i] = 0
            if has_rsiextreme and ((out[i] == 1 and rsi14[i] > 70) or (out[i] == -1 and rsi14[i] < 30)):
                out[i] = 0
            if has_vol and vol_at_close[i] < wf_vol[i]:
                out[i] = 0
            if has_mcbig and i > 10 and mc_std[i] < np.median(mc_std[:i]):
                out[i] = 0
        return out

    # Standalone: WF + regime filter
    sig_w = np.zeros(N, dtype=int)
    sig_w[pred_ret > wf_q90] = 1
    sig_w[pred_ret < wf_q10] = -1

    s = sig_w.copy(); s[reg_bad] = 0
    run_custom(s, "WF+skipBadReg noTP", get_tp_sl_no_tp)

    s = np.zeros(N, dtype=int)
    for i in range(N):
        if reg_best[i] and pred_ret[i] > wf_q90[i]:
            s[i] = 1
        if reg_best[i] and pred_ret[i] < wf_q10[i]:
            s[i] = -1
    run_custom(s, "WF+onlyBestReg noTP", get_tp_sl_no_tp)

    s = apply_filters_monolith(sig_w, has_bbpct=True, has_bbwidth=False, has_bbmom=False,
                               has_conftrend=False, has_rsiextreme=False, has_vol=False, has_mcbig=False)
    s[reg_bad] = 0
    run_custom(s, "WF+BB%B+skipBadReg noTP", get_tp_sl_no_tp)

    # ── 4-regime filter ────────────────────────────────────────────────────
    reg_bad4 = np.zeros(N, dtype=bool)
    for i in range(N):
        pos = i + LK - 1
        if pos >= len(raw): continue
        hv = raw_bb_width[pos] >= raw_wf_width[pos]
        os = raw_bb_pct[pos] < 0.3
        ob = raw_bb_pct[pos] > 0.7
        ex = raw_bb_mom[pos]
        cntr = not ex
        reg_bad4[i] = (hv and ob and cntr) or (hv and os and cntr) or (hv and os and ex) or (not hv and os and ex)

    TOP5 = [
        ("WF+BB%B+BBmom noTP",           "WF",    True,  False, True,  False, False, False, False),
        ("WF+BB%B+BBmom+vol noTP",       "WF",    True,  False, True,  False, False, True,  False),
        ("WF+BB%B+BBmom+mcBig noTP",     "WF",    True,  False, True,  False, False, False, True),
        ("WF+BB%B+BBmom+confTrend noTP", "WF",    True,  False, True,  True,  False, False, False),
        ("H5<0.4+BBwidth+BBmom noTP",    "H5<0.4", False, True,  True,  False, False, False, False),
    ]

    def compute_quarterly(pb):
        rows = []
        for qtr in sorted_qtrs:
            mask = bars_per_qtr[qtr]
            active = pb[mask] != 0
            n_active = int(active.sum())
            if n_active < 5: continue
            rtrn = pb[mask][active]
            pos_sum = rtrn[rtrn > 0].sum()
            neg_sum = abs(rtrn[rtrn < 0].sum())
            pf = float(pos_sum / max(neg_sum, 1e-12))
            ar = float(rtrn.mean())
            wr = float((rtrn > 0).sum() / max(n_active, 1))
            rows.append((qtr, ar, pf, wr, n_active))
        active_all = pb != 0
        if active_all.sum() > 5:
            ra = pb[active_all]
            pos_sum = ra[ra > 0].sum()
            neg_sum = abs(ra[ra < 0].sum())
            pf = float(pos_sum / max(neg_sum, 1e-12))
            ar = float(ra.mean())
            wr = float((ra > 0).sum() / max(int(active_all.sum()), 1))
            rows.append(("overall", ar, pf, wr, int(active_all.sum())))
        return rows

    print(f"\n  {'═'*140}")
    print(f"  4-REGIME FILTER — top 5 strategies: before vs after")
    print(f"  Filtered: high_vol_overbought/oversold_contracting, high_vol_oversold_expanding, low_vol_oversold_expanding")
    print(f"  {'═'*140}")
    for label, base_key, hbbp, hbbw, hbm, hct, hre, hv, hmc in TOP5:
        base_sig_src = base_signals[base_key].copy()
        sig0 = apply_filters_monolith(base_sig_src, hbbp, hbbw, hbm, hct, hre, hv, hmc)
        sig1 = sig0.copy(); sig1[reg_bad4] = 0
        r0, pb0 = run_backtest_custom(sig0, f"{label}", get_tp_sl_no_tp, data, pred_ret_ref=pred_ret, verbose=False)
        r1, pb1 = run_backtest_custom(sig1, f"{label}+4regFilter", get_tp_sl_no_tp, data, pred_ret_ref=pred_ret, verbose=False)
        results.append(r1); per_bars.append(pb1)

        q0 = compute_quarterly(pb0)
        q1 = compute_quarterly(pb1)
        n0d = {k: v for k, *v in [row for row in q0]}
        n1d = {k: v for k, *v in [row for row in q1]}

        print(f"\n  ┌─ {label}")
        print(f"  │ {'Quarter':<12} {'AverRet bef':>10} {'AverRet aft':>10} {'Δ AverRet':>10}   {'PF bef':>8} {'PF aft':>8} {'Δ PF':>8}   {'WR bef':>7} {'WR aft':>7} {'Δ WR':>7}")
        print(f"  ├─{'─'*98}")
        for qtr in sorted_qtrs + (["overall"] if "overall" in n0d else []):
            if qtr not in n0d: continue
            ar0, pf0, wr0, _ = n0d[qtr]
            ar1, pf1, wr1, _ = n1d.get(qtr, (0, 0, 0, 0))
            dar = ar1 - ar0; dpf = pf1 - pf0; dwr = wr1 - wr0
            print(f"  │ {qtr:<12} {ar0:>10.6%} {ar1:>10.6%} {dar:>+10.6%}   {pf0:>8.4f} {pf1:>8.4f} {dpf:>+8.4f}   {wr0:>6.2%} {wr1:>6.2%} {dwr:>+6.2%}")
        ov0 = n0d.get("overall", (0,0,0,0))
        ov1 = n1d.get("overall", (0,0,0,0))
        if ov0[0] != 0:
            print(f"  │ {'─'*98}")
            print(f"  │ SHARPE: {r0['sharpe']:.3f} → {r1['sharpe']:.3f}  ({r1['sharpe'] - r0['sharpe']:+5.3f})  |  TRADES: {int(r0['n_trades'])} → {int(r1['n_trades'])}  ({int(r1['n_trades']) - int(r0['n_trades']):+4d})")
        print(f"  └─")

    # ── TOP5 15h/16h by day-of-week per quarter ────────────────────────────
    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    hr15_16_mask = (bar_hour == 15) | (bar_hour == 16)
    bar_qtr = np.array([f"{str(t)[:4]}-Q{(int(str(t)[5:7])-1)//3+1}" for t in ts])
    print(f"\n  {'═'*140}")
    print(f"  TOP 5 — AverRet & PF at 15h/16h, by day-of-week, per quarter")
    print(f"  {'═'*140}")
    for label, base_key, hbbp, hbbw, hbm, hct, hre, hv, hmc in TOP5:
        base_sig_src = base_signals[base_key].copy()
        sig = apply_filters_monolith(base_sig_src, hbbp, hbbw, hbm, hct, hre, hv, hmc)
        _, pb = run_backtest_custom(sig, label, get_tp_sl_no_tp, data, pred_ret_ref=pred_ret, verbose=False)
        active_idx = np.where((pb != 0) & hr15_16_mask)[0]
        if len(active_idx) == 0: continue
        bar_vals = pb[active_idx]

        print(f"\n  ┌─ {label}")
        print(f"  │ {'Day':<6} {'Quarter':<10} {'AverRet':>10} {'PF':>8} {'Bars':<6}")
        print(f"  ├─{'─'*45}")
        for dw in range(5):
            for qtr in sorted_qtrs:
                dw_mask = bar_dow[active_idx] == dw
                qtr_mask = bar_qtr[active_idx] == qtr
                mask = dw_mask & qtr_mask
                n = int(mask.sum())
                if n < 3: continue
                rr = bar_vals[mask]
                ar = float(rr.mean())
                pos = rr[rr > 0].sum()
                neg = abs(rr[rr < 0].sum())
                pf = float(pos / max(neg, 1e-12))
                print(f"  │ {DOW_NAMES[dw]:<6} {qtr:<10} {ar:>9.6%}  {pf:>7.4f}  {n:<5d}")
        print(f"  │ {'─'*45}")
        for dw in range(5):
            mask = bar_dow[active_idx] == dw
            n = int(mask.sum())
            if n < 3: continue
            rr = bar_vals[mask]
            ar = float(rr.mean())
            pos = rr[rr > 0].sum()
            neg = abs(rr[rr < 0].sum())
            pf = float(pos / max(neg, 1e-12))
            print(f"  │ {DOW_NAMES[dw]:<6} {'all':<10} {ar:>9.6%}  {pf:>7.4f}  {n:<5d}")
        print(f"  └─")

    # ── Resave CSV ──────────────────────────────────────────────────────────
    labels_all = [r["label"] for r in results]
    save_all_results(results, labels_all, per_bars, bars_per_qtr, sorted_qtrs, ts)
    print(f"Re-saved: CSV + quarterly with regime filter results")
