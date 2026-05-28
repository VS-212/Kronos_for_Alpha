"""
M-SIGNAL-BB: Bollinger Bands signal generation
Contract: OHLCV DataFrame → BB values (upper, lower, width, position, squeeze)
Status: ✅ ready
"""

import numpy as np
import pandas as pd


def compute_bb(
    close: np.ndarray,
    period: int = 20,
    std_mult: float = 2.0,
) -> dict:
    """Compute Bollinger Bands from a close price series.

    Args:
        close: (N,) float array of close prices
        period: SMA window
        std_mult: standard deviation multiplier

    Returns:
        sma: (N,) simple moving average
        upper: (N,) upper band = sma + std_mult * std
        lower: (N,) lower band = sma - std_mult * std
        width: (N,) relative band width = (upper - lower) / sma
        last: dict with last values {sma, upper, lower, width}
    """
    sma = pd.Series(close).rolling(period, min_periods=period).mean().values
    std = pd.Series(close).rolling(period, min_periods=period).std(ddof=1).values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / (sma + 1e-8)

    return {
        "sma": sma,
        "upper": upper,
        "lower": lower,
        "width": width,
        "last": {
            "sma": float(sma[-1]) if not np.isnan(sma[-1]) else float("nan"),
            "upper": float(upper[-1]) if not np.isnan(upper[-1]) else float("nan"),
            "lower": float(lower[-1]) if not np.isnan(lower[-1]) else float("nan"),
            "width": float(width[-1]) if not np.isnan(width[-1]) else float("nan"),
        },
    }


def bb_position(close: float, upper: float, lower: float, sma: float) -> dict:
    """Price position relative to Bollinger Bands.

    Returns:
        zone: one of {above_upper, near_upper, middle, near_lower, below_lower}
        pct_bandwidth: (close - lower) / (upper - lower), clipped [0,1]
        is_extreme: True if price is outside bands
        is_near_upper: True if price in upper 25% of band
        is_near_lower: True if price in lower 25% of band
    """
    if np.isnan(upper) or np.isnan(lower) or upper <= lower:
        return {
            "zone": "unknown",
            "pct_bandwidth": 0.5,
            "is_extreme": False,
            "is_near_upper": False,
            "is_near_lower": False,
        }

    bw = upper - lower
    pct = max(0.0, min(1.0, (close - lower) / bw))

    if close > upper:
        zone = "above_upper"
    elif close < lower:
        zone = "below_lower"
    elif pct >= 0.75:
        zone = "near_upper"
    elif pct <= 0.25:
        zone = "near_lower"
    else:
        zone = "middle"

    return {
        "zone": zone,
        "pct_bandwidth": pct,
        "is_extreme": close > upper or close < lower,
        "is_near_upper": zone == "near_upper",
        "is_near_lower": zone == "near_lower",
        "above_upper": close > upper,
        "below_lower": close < lower,
    }


def bb_squeeze(width: np.ndarray, lookback: int = 20, percentile: float = 20.0) -> dict:
    """Detect Bollinger Band squeeze (low volatility compression).

    A squeeze occurs when current band width is in the bottom percentile of recent width.

    Args:
        width: (N,) band width array
        lookback: how many bars to compare against
        percentile: squeeze threshold (e.g. 20 means current width < 20th percentile)

    Returns:
        is_squeeze: True if current width is in squeeze zone
        current_pct: current width as percentile of recent widths
    """
    if len(width) < lookback:
        return {"is_squeeze": False, "current_pct": 50.0}

    recent = width[-lookback:]
    current = recent[-1]
    valid = recent[~np.isnan(recent)]
    if len(valid) < lookback // 2:
        return {"is_squeeze": False, "current_pct": 50.0}

    current_pct = np.mean(valid < current) * 100.0
    is_squeeze = current_pct < percentile

    return {"is_squeeze": is_squeeze, "current_pct": float(current_pct)}


def bb_signal(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> dict:
    """Full Bollinger Band signal at the latest bar.

    Combines compute_bb, bb_position, and bb_squeeze into one call.

    Returns:
        sma, upper, lower, width: last band values
        zone: position zone
        pct_bandwidth: relative position within bands
        is_extreme: price outside bands
        is_squeeze: low-volatility compression
        squeeze_pct: current width percentile
        mean_reversion: bool — near upper and above sma = bearish bias, or near lower and below sma = bullish bias
    """
    bb = compute_bb(close, period, std_mult)
    last = bb["last"]
    pos = bb_position(close[-1], last["upper"], last["lower"], last["sma"])
    squeeze = bb_squeeze(bb["width"])

    mr_long = pos["is_near_lower"] and close[-1] < last["sma"]
    mr_short = pos["is_near_upper"] and close[-1] > last["sma"]

    return {
        **last,
        **pos,
        **squeeze,
        "mr_long": mr_long,
        "mr_short": mr_short,
    }
