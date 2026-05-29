"""
M-FILTERS: Signal filter computation for SBER backtest
Contract: data dict → filter arrays (bool/int masks)
Status: ✅ ready
"""

import numpy as np


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


def compute_bb(data, BB_PERIOD=20, BB_K=2.0):
    """Returns bb_mid, bb_std, bb_upper, bb_lower, bb_width, bb_pct, wf_width"""
    N = data['N']
    LK = data['LK']
    close_arr = data['close_arr']
    entry_close = data['entry_close']

    bb_mid = np.zeros(N)
    bb_std = np.zeros(N)
    for i in range(N):
        pos = i + LK - 1
        if pos >= BB_PERIOD - 1:
            seg = close_arr[pos - BB_PERIOD + 1 : pos + 1]
        else:
            seg = close_arr[:pos + 1]
        bb_mid[i] = seg.mean()
        bb_std[i] = seg.std(ddof=1) if len(seg) > 1 else seg.std() if len(seg) > 0 else 0.0

    bb_upper = bb_mid + BB_K * bb_std
    bb_lower = bb_mid - BB_K * bb_std

    wf_width = np.zeros(N)
    for i in range(N):
        if i < 11:
            wf_width[i] = 0.001
        else:
            w = (bb_upper[:i] - bb_lower[:i]) / np.maximum(bb_mid[:i], 1e-10)
            wf_width[i] = float(np.median(w))

    bb_width = (bb_upper - bb_lower) / np.maximum(bb_mid, 1e-10)
    bb_pct = (entry_close - bb_lower) / np.maximum(bb_upper - bb_lower, 1e-10)
    bb_band_touch_lower = entry_close <= bb_lower + 0.5 * bb_std
    bb_band_touch_upper = entry_close >= bb_upper - 0.5 * bb_std

    return bb_mid, bb_std, bb_upper, bb_lower, bb_width, bb_pct, wf_width, bb_band_touch_lower, bb_band_touch_upper


def bb_width_ok(i, sig, bb_width, wf_width):
    return bb_width[i] >= wf_width[i]


def bb_pct_ok(i, sig, bb_pct):
    return (sig == 1 and bb_pct[i] <= 0.3) or (sig == -1 and bb_pct[i] >= 0.7)


def bb_touch_ok(i, sig, bb_band_touch_lower, bb_band_touch_upper):
    return (sig == 1 and bb_band_touch_lower[i]) or (sig == -1 and bb_band_touch_upper[i])


def compute_lr(data, LR_PERIOD=20, LR_K=2.0):
    """Returns lr_slope, lr_r2, lr_mid, lr_std, lr_upper, lr_lower, lr_channel_pct, wf_r2"""
    N = data['N']
    LK = data['LK']
    close_arr = data['close_arr']
    entry_close = data['entry_close']

    lr_slope = np.zeros(N)
    lr_r2 = np.zeros(N)
    lr_mid = np.zeros(N)
    lr_std = np.zeros(N)
    x_vals = np.arange(LR_PERIOD)

    for i in range(N):
        pos = i + LK - 1
        if pos >= LR_PERIOD - 1:
            seg = close_arr[pos - LR_PERIOD + 1 : pos + 1]
        else:
            seg = close_arr[:pos + 1]
        x = x_vals[:len(seg)]
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, seg, rcond=None)[0]
        lr_slope[i] = slope
        predicted = intercept + slope * x
        residuals = seg - predicted
        ss_res = (residuals ** 2).sum()
        ss_tot = ((seg - seg.mean()) ** 2).sum()
        lr_r2[i] = 1.0 - ss_res / max(ss_tot, 1e-10)
        lr_mid[i] = float(predicted[-1])
        lr_std[i] = float(np.std(residuals, ddof=2) if len(residuals) > 2 else 0.0)

    lr_upper = lr_mid + LR_K * lr_std
    lr_lower = lr_mid - LR_K * lr_std
    lr_channel_pct = (entry_close - lr_lower) / np.maximum(lr_upper - lr_lower, 1e-10)

    wf_r2 = np.zeros(N)
    for i in range(N):
        if i < 11:
            wf_r2[i] = 0.001
        else:
            wf_r2[i] = float(np.median(lr_r2[:i]))

    return lr_slope, lr_r2, lr_mid, lr_std, lr_upper, lr_lower, lr_channel_pct, wf_r2


def compute_atr_filter(data, ATR_PERIOD=14):
    """Returns atr, wf_atr"""
    N = data['N']
    LK = data['LK']
    raw = data['raw']

    tr = np.zeros(len(raw))
    for b in range(1, len(raw)):
        hl = raw[b, 1] - raw[b, 2]
        hc = abs(raw[b, 1] - raw[b - 1, 3])
        lc = abs(raw[b, 2] - raw[b - 1, 3])
        tr[b] = max(hl, hc, lc)

    atr = np.zeros(N)
    for i in range(N):
        pos = i + LK - 1
        if pos >= ATR_PERIOD:
            atr[i] = tr[pos - ATR_PERIOD + 1 : pos + 1].mean()
        else:
            atr[i] = tr[1:pos + 1].mean() if pos > 1 else 0.0

    wf_atr = np.zeros(N)
    for i in range(N):
        wf_atr[i] = float(np.median(atr[:i])) if i > 10 else 0.001

    return atr, wf_atr


def compute_volume_filter(data):
    """Returns vol_at_close, wf_vol"""
    N = data['N']
    LK = data['LK']
    raw = data['raw']

    vol_at_close = raw[LK - 1 : LK - 1 + N, 4]
    wf_vol = np.zeros(N)
    for i in range(N):
        wf_vol[i] = float(np.median(vol_at_close[:i])) if i > 10 else 0.001

    return vol_at_close, wf_vol


def compute_bb_momentum(bb_width):
    """Returns bb_mom boolean array"""
    N = len(bb_width)
    bb_mom = np.zeros(N, dtype=bool)
    for i in range(2, N):
        bb_mom[i] = bb_width[i] > bb_width[i - 1] > bb_width[i - 2]
    return bb_mom


def compute_conf_trend(data):
    """Returns conf_trend bool array"""
    N = data['N']
    conf = data['conf']

    conf_trend = np.ones(N, dtype=bool)
    for i in range(1, N):
        conf_trend[i] = conf[i] > conf[i - 1]
    return conf_trend


def compute_mc_breadth(data):
    """Returns mc_std, mc_small, mc_big"""
    N = data['N']
    pred_close_horizon = data['pred_close_horizon']

    mc_std = pred_close_horizon.std(axis=1)
    mc_breadth = np.ones(N, dtype=bool)
    mc_small = np.ones(N, dtype=bool)
    mc_big = np.ones(N, dtype=bool)
    for i in range(N):
        mc_breadth[i] = mc_std[i] > 0.0
        mc_small[i] = mc_std[i] < np.median(mc_std[:i]) if i > 10 else True
        mc_big[i] = mc_std[i] > np.median(mc_std[:i]) if i > 10 else True

    return mc_std, mc_small, mc_big


def compute_rsi14(data, RSI_PERIOD=14):
    """Returns rsi14"""
    N = data['N']
    LK = data['LK']
    raw = data['raw']

    rsi14 = np.ones(N) * 50.0
    for i in range(N):
        pos = i + LK - 1
        if pos < RSI_PERIOD:
            continue
        closes = raw[pos - RSI_PERIOD : pos + 1, 3]
        gains = np.maximum(np.diff(closes), 0)
        losses = np.maximum(-np.diff(closes), 0)
        avg_gain = np.mean(gains[-RSI_PERIOD:])
        avg_loss = np.mean(losses[-RSI_PERIOD:])
        if avg_loss == 0:
            rsi14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi14[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi14


def compute_pred_z(data):
    """Returns pred_z"""
    N = data['N']
    pred_ret = data['pred_ret']

    pred_z = np.ones(N) * 0.5
    for i in range(N):
        hist = pred_ret[:i]
        if len(hist) > 20:
            m, s = np.mean(hist), np.std(hist)
            pred_z[i] = (pred_ret[i] - m) / max(s, 1e-10)
    return pred_z


def compute_roll_wr(data, WR_WINDOW=50):
    """Returns roll_wr_ok bool array.
    Uses data['wf_signal'] as base signal for trade simulation.
    """
    N = data['N']
    LK = data['LK']
    PL = data['PL']
    raw = data['raw']
    entry_close = data['entry_close']
    entry_open = data['entry_open']
    g_tp = data['g_tp']
    g_sl = data['g_sl']
    COMM = data.get('COMM', 0.0)
    base_signal = data.get('wf_signal')

    if base_signal is None:
        pred_ret = data['pred_ret']
        wf_q90 = data['wf_q90']
        wf_q10 = data['wf_q10']
        base_signal = np.zeros(N, dtype=int)
        for i in range(N):
            if pred_ret[i] > wf_q90[i]:
                base_signal[i] = 1
            elif pred_ret[i] < wf_q10[i]:
                base_signal[i] = -1

    roll_wr_ok = np.ones(N, dtype=bool)
    entry_bar_for_signal = np.zeros(N, dtype=int)
    for i in range(N):
        _sig = base_signal[i]
        if _sig == 0:
            entry_bar_for_signal[i] = -1
        else:
            ec = entry_close[i]
            eo = entry_open[i]
            sl_level = g_sl[i] if _sig == 1 else g_tp[i]
            tp_level = g_tp[i] if _sig == 1 else g_sl[i]
            direction = _sig
            pnl = 0.0
            for t in range(PL):
                idx = i + t + 1
                if idx >= N:
                    break
                bar_open = raw[LK + idx - 1, 0]
                bar_high = raw[LK + idx - 1, 1]
                bar_low = raw[LK + idx - 1, 2]
                in_trade = True
                if direction == 1:
                    if bar_low <= sl_level:
                        exit_price = sl_level
                        in_trade = False
                    elif bar_high >= tp_level and t > 0:
                        exit_price = tp_level
                        in_trade = False
                    else:
                        exit_price = raw[LK + idx - 1, 3] if t == PL - 1 else None
                else:
                    if bar_high >= sl_level:
                        exit_price = sl_level
                        in_trade = False
                    elif bar_low <= tp_level and t > 0:
                        exit_price = tp_level
                        in_trade = False
                    else:
                        exit_price = raw[LK + idx - 1, 3] if t == PL - 1 else None
                if not in_trade or t == PL - 1:
                    pnl = direction * (exit_price - eo) / max(eo, 1e-8) - COMM
                    break
            entry_bar_for_signal[i] = pnl > 0

    mock_wins = np.zeros(N, dtype=bool)
    for i in range(N):
        if entry_bar_for_signal[i] < 0:
            roll_wr_ok[i] = True
            continue
        mock_wins[i] = entry_bar_for_signal[i] > 0
        lo = max(0, i - WR_WINDOW)
        wins_in_window = mock_wins[lo:i+1].sum()
        total_in_window = (entry_bar_for_signal[lo:i+1] >= 0).sum()
        wr = wins_in_window / max(total_in_window, 1)
        roll_wr_ok[i] = wr >= 0.5

    return roll_wr_ok


def compute_mc_agreement(data):
    """Returns mc_agreement array"""
    N = data['N']
    PL = data['PL']
    preds = data['preds']
    entry_close = data['entry_close']

    pred_close_mc = preds[:, :, PL - 1, 3]
    pred_dir_mc = np.sign(pred_close_mc - entry_close[:, None])
    mc_agreement = np.abs(pred_dir_mc.sum(axis=1))
    return mc_agreement


def compute_weighted_quantiles(data):
    """Returns g_tp_w, g_sl_w, pred_ret_w, wf_q90_w, wf_q10_w"""
    N = data['N']
    PL = data['PL']
    preds = data['preds']
    belief = data['belief']
    entry_close = data['entry_close']
    Q_LONG = data.get('Q_LONG', 0.90)
    Q_SHORT = data.get('Q_SHORT', 0.10)
    TP_Q = data.get('TP_Q', 0.80)
    SL_Q = data.get('SL_Q', 0.20)

    pred_close_mc = preds[:, :, PL - 1, 3]

    conf_per_mc = belief[:, :, :, 0].mean(axis=2)
    conf_weights = conf_per_mc / (conf_per_mc.sum(axis=1, keepdims=True) + 1e-8)

    pred_ret_mc_arr = (pred_close_mc - entry_close[:, None]) / np.maximum(entry_close[:, None], 1e-8)

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

    return g_tp_w, g_sl_w, pred_ret_w, wf_q90_w, wf_q10_w


def compute_best_mc(data):
    """Returns pred_ret_best, wf_q90_best, wf_q10_best"""
    N = data['N']
    PL = data['PL']
    preds = data['preds']
    belief = data['belief']
    entry_close = data['entry_close']
    Q_LONG = data.get('Q_LONG', 0.90)
    Q_SHORT = data.get('Q_SHORT', 0.10)

    pred_close_mc = preds[:, :, PL - 1, 3]
    conf_per_mc = belief[:, :, :, 0].mean(axis=2)
    pred_ret_mc_arr = (pred_close_mc - entry_close[:, None]) / np.maximum(entry_close[:, None], 1e-8)

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

    return pred_ret_best, wf_q90_best, wf_q10_best


def compute_drop_low_conf(data, drop_th=0.4):
    """Returns pred_ret_drop, wf_q90_d, wf_q10_d"""
    N = data['N']
    PL = data['PL']
    preds = data['preds']
    belief = data['belief']
    entry_close = data['entry_close']
    Q_LONG = data.get('Q_LONG', 0.90)
    Q_SHORT = data.get('Q_SHORT', 0.10)

    pred_close_mc = preds[:, :, PL - 1, 3]
    conf_per_mc = belief[:, :, :, 0].mean(axis=2)
    pred_ret_mc_arr = (pred_close_mc - entry_close[:, None]) / np.maximum(entry_close[:, None], 1e-8)

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

    return pred_ret_drop, wf_q90_d, wf_q10_d


def compute_asymmetry_ratio(data):
    """Returns can_long, can_short (bool arrays)"""
    entry_close = data['entry_close']
    g_tp = data['g_tp']
    g_sl = data['g_sl']

    can_long = g_tp > entry_close
    can_short = g_sl < entry_close
    return can_long, can_short


def apply_filters(sig, data, bb_pct=None, bb_width=None, wf_width=None, bb_mom=None,
                  conf_trend=None, rsi14=None, vol_at_close=None, wf_vol=None, mc_std=None,
                  has_bbpct=False, has_bbwidth=False, has_bbmom=False,
                  has_conftrend=False, has_rsiextreme=False, has_vol=False, has_mcbig=False):
    """Apply filters to a base signal array. Returns filtered signal copy."""
    N = data['N']
    out = sig.copy()
    for i in range(N):
        if out[i] == 0:
            continue
        if has_bbpct and ((out[i] == 1 and bb_pct[i] > 0.3) or (out[i] == -1 and bb_pct[i] < 0.7)):
            out[i] = 0
        if has_bbwidth and bb_width[i] < wf_width[i]:
            out[i] = 0
        if has_bbmom and not bb_mom[i]:
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
