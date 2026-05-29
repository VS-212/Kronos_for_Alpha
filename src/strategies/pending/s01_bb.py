"""
M-STRATEGY-S01: Bollinger Band + consensus filter
Contract: samples DF + mamba DF → trades DataFrame (BB extremes + consensus)
Status: ✅ ready
"""

"""S1: BB + Consensus Filter — enter only if price outside BB + consensus matches."""
import numpy as np
import pandas as pd

from src.evaluation.output import reconstruct
from src.signals.atoms import consensus
from src.strategies.pending.core import _enrich_trade, _simulate_trade, lookup_bb


def run(
    df,
    mamba,
    pred_len,
    sample_count,
    tp_q=0.90,
    sl_q=0.10,
    atr_tp_mult=None,
    atr_sl_mult=None,
    save_first_n=0,
    consensus_threshold=0.8,
):
    trades = []
    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        bb = lookup_bb(mamba, row["pred_ts"])
        if np.isnan(bb["bb_upper"]):
            continue

        entry_price = float(actuals[0, 0])
        outside_upper = entry_price >= bb["bb_upper"]
        outside_lower = entry_price <= bb["bb_lower"]

        cons = consensus(close_only, prev_close, threshold=consensus_threshold)
        if not cons["has_consensus"][0]:
            continue
        pred_dir = int(cons["consensus_dir"][0])

        if not ((outside_upper and pred_dir == -1) or (outside_lower and pred_dir == 1)):
            continue

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
        )
        if trade is not None:
            trades.append(_enrich_trade(trade, row))
    return pd.DataFrame(trades)
