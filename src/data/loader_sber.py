"""
M-LOAD-SBER: SBER prediction and price data loader.

Purpose: Load SBER predictions, belief, raw OHLCV, timestamps from disk and
  compute per-window returns, TP/SL quantiles, walk-forward quantiles.

Input: Hardcoded paths to data/v3/predictions/10min_sber_mini and
  data/tickers/SBER.
Output: dict with keys:
  preds, belief, raw, ts, N, conf, entry_close, entry_open, pred_ret,
  actual_ret, g_tp, g_sl, wf_q90, wf_q10, close_arr, conf_per_mc.

Guarantees: Deterministic (same files → same output). No side effects.
Known failures: Missing .npy files on disk → np.load raises FileNotFoundError.
"""

import numpy as np

DATA_DIR = "data/v3/predictions/10min_sber_mini"
RAW_DIR = "data/tickers/SBER"
LK = 500
PL = 12
COMM = 0.0
Q_LONG = 0.90
Q_SHORT = 0.10
TP_Q = 0.80
SL_Q = 0.20


def load_sber_data():
    """Load SBER data and compute all per-window values.

    Returns
    -------
    dict with keys:
        preds       (N, 5, 12, 6)  — MC predictions
        belief      (N, 5, 12, 4)  — belief metrics per step
        raw         (T, 6)         — raw OHLCV
        ts          (T,)           — timestamps
        N           int            — number of windows
        conf        (N,)           — mean confidence per window
        entry_close (N,)           — close[T] reference
        entry_open  (N,)           — open[T+1] trade entry
        pred_ret    (N,)           — mean predicted return over PL
        actual_ret  (N,)           — actual return over PL
        g_tp        (N,)           — q80 TP level
        g_sl        (N,)           — q20 SL level
        wf_q90      (N,)           — walk-forward q90
        wf_q10      (N,)           — walk-forward q10
        close_arr   (T,)           — raw close prices
        conf_per_mc (N, 5)         — mean confidence per MC path
    """
    preds = np.load(f"{DATA_DIR}/SBER_preds_pl12_sc5.npy")
    belief = np.load(f"{DATA_DIR}/SBER_belief_pl12_sc5.npy")
    raw = np.load(f"{RAW_DIR}/feats_test_raw.npy")
    ts = np.load(f"{RAW_DIR}/timestamps_test_raw.npy", allow_pickle=True)

    N = preds.shape[0]
    conf = belief[:, :, :, 0].mean(axis=(1, 2))

    entry_close = np.zeros(N)
    entry_open = np.zeros(N)
    pred_ret = np.zeros(N)
    actual_ret = np.zeros(N)

    for i in range(N):
        ec = raw[i + LK - 1, 3]
        entry_close[i] = ec
        entry_open[i] = raw[i + LK, 0]
        pred_ret[i] = (preds[i, :, PL - 1, 3].mean() - ec) / max(ec, 1e-8)
        ci = min(i + LK + PL - 1, len(raw) - 1)
        actual_ret[i] = (raw[ci, 3] - ec) / max(ec, 1e-8)

    pred_close_horizon = preds[:, :, PL - 1, 3]
    g_tp = np.quantile(pred_close_horizon, TP_Q, axis=1)
    g_sl = np.quantile(pred_close_horizon, SL_Q, axis=1)

    wf_q90 = np.zeros(N)
    wf_q10 = np.zeros(N)
    for i in range(N):
        if i < 100:
            wf_q90[i] = float(np.quantile(pred_ret[:i + 1], Q_LONG)) if i > 10 else 0.001
            wf_q10[i] = float(np.quantile(pred_ret[:i + 1], Q_SHORT)) if i > 10 else -0.001
        else:
            wf_q90[i] = float(np.quantile(pred_ret[:i], Q_LONG))
            wf_q10[i] = float(np.quantile(pred_ret[:i], Q_SHORT))

    close_arr = raw[:, 3]
    conf_per_mc = belief[:, :, :, 0].mean(axis=2)

    return {
        "preds": preds,
        "belief": belief,
        "raw": raw,
        "ts": ts,
        "N": N,
        "conf": conf,
        "entry_close": entry_close,
        "entry_open": entry_open,
        "pred_ret": pred_ret,
        "actual_ret": actual_ret,
        "g_tp": g_tp,
        "g_sl": g_sl,
        "wf_q90": wf_q90,
        "wf_q10": wf_q10,
        "close_arr": close_arr,
        "conf_per_mc": conf_per_mc,
    }
