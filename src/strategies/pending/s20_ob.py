"""
M-STRATEGY-S20: Order Block + Kronos consensus
Contract: samples DF + mamba DF → trades DataFrame (OB zone + Kronos consensus)
Status: ✅ ready
"""

"""S20: Order Block + Kronos — enter when price is at an OB zone."""
from src.signals.ict import detect_order_block
from src.strategies.pending.core import _run_ict_strategy


def run(
    df,
    mamba,
    pred_len,
    sample_count,
    tp_q=0.90,
    sl_q=0.10,
    lookback=48,
    move_threshold=0.005,
    max_age=24,
    atr_tp_mult=None,
    atr_sl_mult=None,
    save_first_n=0,
    consensus_threshold=0.8,
):
    return _run_ict_strategy(
        df,
        mamba,
        pred_len,
        sample_count,
        tp_q,
        sl_q,
        lambda **kw: detect_order_block(
            kw["open"],
            kw["high"],
            kw["low"],
            kw["close"],
            lookback=kw["lookback"],
            move_threshold=move_threshold,
            max_age=max_age,
        ),
        lookback,
        atr_tp_mult=atr_tp_mult,
        atr_sl_mult=atr_sl_mult,
        save_first_n=save_first_n,
        consensus_threshold=consensus_threshold,
    )
