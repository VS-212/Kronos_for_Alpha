"""
M-SIM: Single-asset trade simulation with TP/SL.

Purpose: Pure functions for trade execution, TP/SL level selection, and weighted
  quantile computation — no global state, all data passed explicitly.

Input: Parameters via function arguments (i, sig, entry_open, raw, LK, PL, etc.)
Output: Returns (bar_idx, trade_return, holding_period) or (tp, sl) tuple.

Guarantees: Stateless, deterministic (no random), no lookahead, no side effects.
Known failures: entry_px <= 0 -> return None (invalid entry).
"""

import numpy as np


def simulate_trade(i, sig, tp_level, sl_level, entry_open, raw, LK, PL):
    """Run a single trade with TP/SL. Returns (bar_idx, trade_return, holding_period)."""
    entry_px = entry_open[i]
    if entry_px <= 0:
        return None, 0.0, 0
    for s in range(PL):
        bi = i + LK + s
        if bi >= len(raw):
            break
        hi, lo, _ = raw[bi, 1], raw[bi, 2], raw[bi, 3]
        if sig == 1:
            sl_hit = lo <= sl_level
            tp_hit = hi >= tp_level
        else:
            sl_hit = hi >= sl_level
            tp_hit = lo <= tp_level
        if sl_hit:
            return bi, sig * (sl_level - entry_px) / entry_px, s + 1
        if tp_hit:
            return bi, sig * (tp_level - entry_px) / entry_px, s + 1
    bi = min(i + LK + PL - 1, len(raw) - 1)
    exit_px = raw[bi, 3]
    return bi, sig * (exit_px - entry_px) / entry_px, PL


def get_tp_sl(i, sig, entry_close, g_tp, g_sl):
    """Return (tp, sl) for window i and signal direction using global TP/SL levels."""
    ec = entry_close[i]
    if sig == 1:
        tp = g_tp[i] if g_tp[i] > ec else np.inf
        sl = g_sl[i] if g_sl[i] < ec else -np.inf
    else:
        tp = g_sl[i] if g_sl[i] < ec else -np.inf
        sl = g_tp[i] if g_tp[i] > ec else np.inf
    return tp, sl


def get_tp_sl_no_sl(i, sig, entry_close, g_tp, g_sl):
    """Return (tp, sl) with SL disabled (TP + horizon exit only)."""
    ec = entry_close[i]
    if sig == 1:
        tp = g_tp[i] if g_tp[i] > ec else np.inf
        sl = -np.inf
    else:
        tp = g_sl[i] if g_sl[i] < ec else -np.inf
        sl = np.inf
    return tp, sl


def get_tp_sl_no_tp(i, sig, entry_close, g_tp, g_sl):
    """Return (tp, sl) with TP disabled (SL + horizon exit only)."""
    ec = entry_close[i]
    if sig == 1:
        tp = np.inf
        sl = g_sl[i] if g_sl[i] < ec else -np.inf
    else:
        tp = -np.inf
        sl = g_tp[i] if g_tp[i] > ec else np.inf
    return tp, sl


def _weighted_q(vals, weights, q):
    """Weighted quantile q (0..1) over 5 MC samples. vals, weights: (N,5)."""
    res = np.zeros(vals.shape[0])
    for i in range(len(res)):
        idx = np.argsort(vals[i])
        sv = vals[i][idx]
        sw = weights[i][idx]
        cum = np.cumsum(sw)
        res[i] = np.interp(q, cum - sw / 2, sv)
    return res


def get_tp_sl_w(i, sig, entry_close, g_tp_w, g_sl_w):
    """Return (tp, sl) for window i using WEIGHTED TP/SL levels."""
    ec = entry_close[i]
    if sig == 1:
        tp = g_tp_w[i] if g_tp_w[i] > ec else np.inf
        sl = g_sl_w[i] if g_sl_w[i] < ec else -np.inf
    else:
        tp = g_sl_w[i] if g_sl_w[i] < ec else -np.inf
        sl = g_tp_w[i] if g_tp_w[i] > ec else np.inf
    return tp, sl
