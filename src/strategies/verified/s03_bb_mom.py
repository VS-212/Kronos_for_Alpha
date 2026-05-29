"""WF+BB%B+BBmom noTP: BB%B + BB momentum expansion"""
import numpy as np

from src.evaluation.engine import run_backtest_custom
from src.evaluation.simulation import get_tp_sl_no_tp
from src.signals.filters import bb_pct_ok, compute_bb, compute_bb_momentum


def run(data, verbose=True):
    bb_mid, bb_std, bb_upper, bb_lower, bb_width, bb_pct_vals, wf_width, bbt_lower, bbt_upper = compute_bb(data)
    bb_mom = compute_bb_momentum(bb_width)
    base = _build_wf_signal(data)
    sig = np.zeros(data['N'], dtype=int)
    for i in range(data['N']):
        if base[i] == 0:
            continue
        if not bb_pct_ok(i, base[i], bb_pct_vals):
            continue
        if not bb_mom[i]:
            continue
        sig[i] = base[i]
    return run_backtest_custom(sig, "WF+BB%B+BBmom noTP", get_tp_sl_no_tp, data, pred_ret_ref=data['pred_ret'], verbose=verbose)

def _build_wf_signal(data):
    N = data['N']; wf_q90 = data['wf_q90']; wf_q10 = data['wf_q10']; pr = data['pred_ret']
    sig = np.zeros(N, dtype=int)
    for i in range(N):
        if pr[i] > wf_q90[i]:
            sig[i] = 1
        elif pr[i] < wf_q10[i]:
            sig[i] = -1
    return sig
