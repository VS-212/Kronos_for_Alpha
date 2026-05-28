"""
M-STRATEGY-VANILLA: Pure Kronos consensus entry
Contract: samples DF + mamba DF → trades DataFrame
Status: ✅ ready
"""

"""Vanilla: pure consensus strategy — no external filter, just Kronos q90/q10 TP/SL."""
import pandas as pd

from src.evaluation.output import reconstruct
from src.signals.atoms import consensus
from src.strategies.core import _enrich_trade, _simulate_trade


def run(
    df,
    mamba,
    pred_len,
    sample_count,
    tp_q=0.90,
    sl_q=0.10,
    atr_tp_mult=None,
    atr_sl_mult=None,
    save_first_n=1,
    consensus_threshold=0.8,
):
    trades = []
    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        cons = consensus(close_only, prev_close, threshold=consensus_threshold)
        if not cons["has_consensus"][0]:
            continue
        pred_dir = int(cons["consensus_dir"][0])

        trade = _simulate_trade(
            close_only,
            actuals,
            prev_close,
            pred_dir,
            pred_len,
            tp_q,
            sl_q,
            mamba=mamba,
            pred_ts=row["pred_ts"],
            atr_tp_mult=atr_tp_mult,
            atr_sl_mult=atr_sl_mult,
            save_first_n=save_first_n,
            consensus_threshold=consensus_threshold,
            bypass_consensus=True,
        )
        if trade is not None:
            trades.append(_enrich_trade(trade, row))
    return pd.DataFrame(trades)
