"""
M-STRATEGY-S02: Bollinger Band mean reversion
Contract: samples DF + mamba DF → trades DataFrame (BB zone + consensus)
Status: ✅ ready
"""

"""S2: BB Mean Reversion — enter when price is near BB extreme + consensus confirms."""
import numpy as np
import pandas as pd

from src.evaluation.output import reconstruct
from src.signals.atoms import consensus
from src.strategies.core import _enrich_trade, _simulate_trade, lookup_bb


def run(
    df, mamba, pred_len, sample_count, tp_q=0.90, sl_q=0.10, bb_zone=0.5, consensus_threshold=0.8
):
    trades = []
    for _, row in df.iterrows():
        close_only = reconstruct(row["samples_blob"], sample_count, pred_len)
        actuals = reconstruct(row["actual_blob"], pred_len, 4)
        prev_close = float(row["prev_close"])
        if prev_close == 0:
            continue

        bb = lookup_bb(mamba, row["pred_ts"])
        if np.isnan(bb["bb_upper"]) or bb["bb_width"] == 0:
            continue

        entry_price = float(actuals[0, 0])
        zone_dist = bb_zone * (bb["bb_upper"] - bb["bb_lower"])
        near_lower = entry_price <= bb["bb_lower"] + zone_dist
        near_upper = entry_price >= bb["bb_upper"] - zone_dist

        cons = consensus(close_only, prev_close, threshold=consensus_threshold)
        if not cons["has_consensus"][0]:
            continue
        pred_dir = int(cons["consensus_dir"][0])

        if not ((near_lower and pred_dir == 1) or (near_upper and pred_dir == -1)):
            continue

        trade = _simulate_trade(
            close_only,
            actuals,
            prev_close,
            pred_dir,
            pred_len,
            tp_q,
            sl_q,
            consensus_threshold=consensus_threshold,
        )
        if trade is not None:
            trades.append(_enrich_trade(trade, row))
    return pd.DataFrame(trades)
