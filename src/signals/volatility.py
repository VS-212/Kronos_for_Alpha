"""
M-SIGNAL-VOL: Volatility signals — ATR, ADR, regime
Contract: OHLCV arrays + is_day_start → volatility diagnostics dict
Status: ✅ ready
"""

"""Volatility detection using ADR (Average Daily Range) and ATR (Average True Range).

Used as a filter for OB/BB strategies — entering only during favorable
volatility regimes (low vol → breakout, high vol → reversion).
"""

import numpy as np


def compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, lookback: int = 500
) -> dict:
    """Compute ATR and its percentile rank for the last candle.

    ATR[t] = EMA(period) of True Range[t].
    True Range[t] = max(high-low, |high-close[t-1]|, |low-close[t-1]|).

    Returns:
        atr: last ATR value
        atr_pct: percentile rank (0-1) of current ATR within lookback window
        low_vol: atr_pct < 0.3
        high_vol: atr_pct > 0.7
        rising: ATR increased for 3+ consecutive periods
        falling: ATR decreased for 3+ consecutive periods
    """
    n = len(close)
    if n < period + 1:
        return {
            "atr": 0,
            "atr_pct": 0.5,
            "low_vol": False,
            "high_vol": False,
            "rising": False,
            "falling": False,
        }

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # EMA of TR
    atr = np.zeros(n)
    atr[period] = float(np.mean(tr[1 : period + 1]))
    alpha = 2.0 / (period + 1)
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    last_atr = float(atr[-1])
    start = max(period, n - lookback)
    atr_window = atr[start:-1]  # exclude current
    if len(atr_window) < 20:
        atr_pct = 0.5
    else:
        atr_pct = float(np.mean(atr_window < last_atr))

    low_vol = atr_pct < 0.3
    high_vol = atr_pct > 0.7

    # Rising/falling detection
    rising = False
    falling = False
    if len(atr) >= period + 4:
        if all(atr[-3:] > atr[-4:-1]):
            rising = True
        if all(atr[-3:] < atr[-4:-1]):
            falling = True

    return {
        "atr": last_atr,
        "atr_pct": atr_pct,
        "low_vol": low_vol,
        "high_vol": high_vol,
        "rising": rising,
        "falling": falling,
    }


def compute_adr(
    high: np.ndarray,
    low: np.ndarray,
    is_day_start: np.ndarray,
    adr_period: int = 20,
    percentile_threshold: float = 0.3,
) -> dict:
    """Compute Average Daily Range and current day's range percentile.

    ADR = mean of daily ranges over adr_period full days.
    Current day's range percentile = rank within historical daily ranges.
    low_range = percentile_rank < pct_threshold (bottom 30% of days).
    high_range = percentile_rank > 1-pct_threshold (top 30% of days).

    Returns:
        adr: ADR value (mean daily range)
        current_range: today's range so far
        range_ratio: current_range / adr (1.0 = typical)
        low_range: current day's range is in bottom pct_threshold percentile
        high_range: current day's range is in top pct_threshold percentile
        above_adr: current_range > adr
        percentile: rank of current_range within daily_ranges (0-1)
    """
    n = len(high)
    day_indices = np.where(is_day_start)[0]

    if len(day_indices) < 2:
        return {
            "adr": 0,
            "current_range": 0,
            "range_ratio": 1.0,
            "low_range": False,
            "high_range": False,
            "above_adr": False,
        }

    # Daily ranges
    daily_ranges = []
    last_start = int(day_indices[-1])  # current day start
    for i in range(len(day_indices) - 2, max(-1, len(day_indices) - adr_period - 2), -1):
        if i < 0:
            break
        di = int(day_indices[i])
        di_next = int(day_indices[i + 1])
        if di_next >= n:
            di_next = n - 1
        day_range = float(np.max(high[di:di_next]) - np.min(low[di:di_next]))
        daily_ranges.append(day_range)

    if len(daily_ranges) < 3:
        return {
            "adr": 0,
            "current_range": 0,
            "range_ratio": 1.0,
            "low_range": False,
            "high_range": False,
            "above_adr": False,
        }

    adr = float(np.mean(daily_ranges))

    # Current day's range (from last_start to end)
    current_range = float(np.max(high[last_start:]) - np.min(low[last_start:]))
    range_ratio = current_range / max(adr, 1e-10)

    # Percentile rank: fraction of historical daily ranges below today's range
    dr_arr = np.array(daily_ranges, dtype=np.float64)
    cnt_lower = float(np.sum(dr_arr < current_range))
    percentile = cnt_lower / max(len(dr_arr), 1)
    low_range = percentile < percentile_threshold
    high_range = percentile > (1.0 - percentile_threshold)
    above_adr = current_range > adr

    return {
        "adr": adr,
        "current_range": current_range,
        "range_ratio": range_ratio,
        "low_range": low_range,
        "high_range": high_range,
        "above_adr": above_adr,
        "percentile": percentile,
    }


def volatility_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    is_day_start: np.ndarray,
    atr_period: int = 14,
    adr_period: int = 20,
    atr_pct_low: float = 0.3,
    atr_pct_high: float = 0.7,
    adr_pct: float = 0.3,
) -> str:
    """Classify current volatility regime.

    Returns:
        "compression": low ADR (day is quiet) + low ATR percentile
        "expansion": high ADR + high ATR percentile
        "neutral": everything else
    """
    atr = compute_atr(high, low, close, atr_period)
    adr = compute_adr(high, low, is_day_start, adr_period, adr_pct)

    if adr["low_range"] and atr["low_vol"]:
        return "compression"
    if adr["high_range"] and atr["high_vol"]:
        return "expansion"

    return "neutral"
