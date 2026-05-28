"""
M-STRATEGY-S34: VWAP-confirmed Order Block
Contract: samples DF + mamba DF → trades DataFrame (VWAP + OB + consensus)
Status: ✅ ready
"""

"""S34: VWAP + OB — only enter OB when VWAP position confirms."""
import pandas as pd

from src.evaluation.output import reconstruct
from src.signals.atoms import consensus
from src.signals.ict import detect_order_block
from src.signals.vwap import compute_vwap
from src.strategies.core import _enrich_trade, _simulate_trade, lookup_mamba_window


def run(
    df,
    mamba,
    pred_len,
    sample_count,
    tp_q=0.90,
    sl_q=0.10,
    k=1.0,
    lookback=96,
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
        if not w["has_data"] or len(w["close"]) < 2:
            continue

        ob = detect_order_block(
            w["open"],
            w["high"],
            w["low"],
            w["close"],
            lookback=48,
            move_threshold=0.002,
            max_age=24,
        )
        if ob["signal"] == "none":
            continue
        if not (
            (ob["signal"] == "bullish" and pred_dir == 1)
            or (ob["signal"] == "bearish" and pred_dir == -1)
        ):
            continue

        vw = compute_vwap(w["high"], w["low"], w["close"], w["volume"], w["is_day_start"], k=k)
        if pred_dir == 1 and not vw["below_vwap"]:
            continue
        if pred_dir == -1 and not vw["above_vwap"]:
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
