"""
M-STRATEGY-S28: Volume-filtered Order Block
Contract: samples DF + mamba DF → trades DataFrame (volume-filtered OB + consensus)
Status: ✅ ready
"""

"""S28: Volume-filtered OB + Kronos — only OB candles with high volume."""
import pandas as pd

from src.evaluation.output import reconstruct
from src.signals.atoms import consensus
from src.signals.ict import detect_volume_ob
from src.strategies.pending.core import _enrich_trade, _simulate_trade, lookup_mamba_window


def run(
    df,
    mamba,
    pred_len,
    sample_count,
    tp_q=0.90,
    sl_q=0.10,
    lookback=48,
    move_threshold=0.002,
    max_age=24,
    volume_mult=1.5,
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

        cons = consensus(close_only, prev_close, threshold=consensus_threshold)
        if not cons["has_consensus"][0]:
            continue
        pred_dir = int(cons["consensus_dir"][0])

        w = lookup_mamba_window(mamba, row["pred_ts"], lookback)
        if not w["has_data"]:
            continue

        vob = detect_volume_ob(
            w["open"],
            w["high"],
            w["low"],
            w["close"],
            w["volume"],
            lookback=lookback,
            move_threshold=move_threshold,
            max_age=max_age,
            volume_mult=volume_mult,
        )
        if vob["signal"] == "none":
            continue
        if not (
            (vob["signal"] == "bullish" and pred_dir == 1)
            or (vob["signal"] == "bearish" and pred_dir == -1)
        ):
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
