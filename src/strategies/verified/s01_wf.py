"""WF baseline: walk-forward q90/q10 signal + TP/SL"""
import numpy as np

from src.evaluation.engine import run_backtest


def run(data, verbose=True):
    sig = _build_signal(data)
    return run_backtest(sig, "WF baseline", data, verbose=verbose)

def _build_signal(data):
    N = data['N']; wf_q90 = data['wf_q90']; wf_q10 = data['wf_q10']; pred_ret = data['pred_ret']
    sig = np.zeros(N, dtype=int)
    for i in range(N):
        if pred_ret[i] > wf_q90[i]:
            sig[i] = 1
        elif pred_ret[i] < wf_q10[i]:
            sig[i] = -1
    return sig
