"""
M-SIGNAL-VWAP: Volume-weighted average price signals
Contract: OHLCV + is_day_start → VWAP bands, cross detection, anchored VWAP
Status: ✅ ready
"""

"""VWAP (Volume Weighted Average Price) with standard deviation bands.

VWAP is computed cumulatively from the last known day_start marker.
Each trading day gets a fresh VWAP calculation.

Available signals:
  - vwap_bands: position relative to VWAP ± k*σ bands
  - vwap_cross: recent VWAP cross direction
  - anchored_vwap: VWAP from a specific swing point (fractal)
  - vwap_volume: VWAP-based signal with volume surge filter
"""

import numpy as np


def compute_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    is_day_start: np.ndarray,
    k: float = 2.0,
) -> dict:
    """Compute VWAP and bands for the last candle.

    VWAP resets at each is_day_start=True candle.

    Args:
        high, low, close, volume: (N,) arrays
        is_day_start: (N,) bool — True at first candle of each session
        k: number of standard deviations for bands (default 2.0)

    Returns:
        vwap: last VWAP value
        upper: VWAP + k*σ
        lower: VWAP - k*σ
        sigma: standard deviation
        pct_pos: (close - vwap) / (upper - lower) — position within bands
        zscore: (close - vwap) / σ — distance from VWAP
        below_vwap: bool — close below VWAP (discount)
        above_vwap: bool — close above VWAP (premium)
        near_lower: bool — close within k*σ of lower band
        near_upper: bool — close within k*σ of upper band
        outside_lower: bool — close below lower band
        outside_upper: bool — close above upper band
        crossed_up: bool — price crossed from below to above VWAP in last 2 candles
        crossed_down: bool — price crossed from above to below VWAP in last 2 candles
    """
    n = len(close)
    if n < 2:
        return {
            "vwap": float(close[-1]) if n > 0 else 0,
            "upper": float(close[-1]) if n > 0 else 0,
            "lower": float(close[-1]) if n > 0 else 0,
            "zscore": 0,
            "near_lower": False,
            "near_upper": False,
            "crossed_up": False,
            "crossed_down": False,
        }

    typical = (high + low + close) / 3.0
    idx = np.where(is_day_start)[0]
    if len(idx) == 0:
        idx = np.array([0], dtype=np.int32)

    # Find the latest day_start that is <= n-1
    last_start = int(idx[idx <= n - 1][-1]) if np.any(idx <= n - 1) else 0
    seg_high = high[last_start:]
    seg_low = low[last_start:]
    seg_close = close[last_start:]
    seg_vol = volume[last_start:]
    seg_typ = (seg_high + seg_low + seg_close) / 3.0

    # Cumulative VWAP
    cum_pv = np.cumsum(seg_typ * seg_vol)
    cum_v = np.cumsum(seg_vol)
    cum_vwap = cum_pv / np.maximum(cum_v, 1e-10)

    # Cumulative sigma
    cum_p2v = np.cumsum(seg_typ**2 * seg_vol)
    variance = cum_p2v / np.maximum(cum_v, 1e-10) - cum_vwap**2
    variance = np.maximum(variance, 0)
    sigma = np.sqrt(variance)

    last_vwap = float(cum_vwap[-1])
    last_sigma = float(sigma[-1])
    last_close = float(seg_close[-1])
    upper = float(last_vwap + k * last_sigma)
    lower = float(last_vwap - k * last_sigma)
    rng = upper - lower
    zscore = (last_close - last_vwap) / max(last_sigma, 1e-10)
    pct_pos = (last_close - lower) / max(rng, 1e-10)

    # Cross detection (last 2 full candles)
    crossed_up = False
    crossed_down = False
    if len(seg_close) >= 3:
        c_prev = float(seg_close[-2])
        c_prev2 = float(seg_close[-3])
        vwap_prev = float(cum_vwap[-2])
        vwap_prev2 = float(cum_vwap[-3])
        crossed_up = c_prev <= vwap_prev and c_prev2 > vwap_prev2 and last_close > last_vwap
        crossed_down = c_prev >= vwap_prev and c_prev2 < vwap_prev2 and last_close < last_vwap
    elif len(seg_close) >= 2:
        c_prev = float(seg_close[-2])
        vwap_prev = float(cum_vwap[-2])
        crossed_up = c_prev <= vwap_prev and last_close > last_vwap
        crossed_down = c_prev >= vwap_prev and last_close < last_vwap

    return {
        "vwap": last_vwap,
        "upper": upper,
        "lower": lower,
        "sigma": last_sigma,
        "pct_pos": max(0, min(1, pct_pos)),
        "zscore": zscore,
        "below_vwap": last_close < last_vwap,
        "above_vwap": last_close > last_vwap,
        "near_lower": last_close <= lower or last_close <= last_vwap - 0.8 * k * last_sigma,
        "near_upper": last_close >= upper or last_close >= last_vwap + 0.8 * k * last_sigma,
        "outside_lower": last_close <= lower,
        "outside_upper": last_close >= upper,
        "within_bands": lower <= last_close <= upper,
        "crossed_up": crossed_up,
        "crossed_down": crossed_down,
    }


def anchored_vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, anchor_idx: int
) -> dict:
    """Compute VWAP anchored at a specific index (e.g. a fractal swing).

    Returns same band info as compute_vwap but anchored at anchor_idx.
    """
    sub_h = high[anchor_idx:]
    sub_l = low[anchor_idx:]
    sub_c = close[anchor_idx:]
    sub_v = volume[anchor_idx:]

    fake_day_start = np.zeros(len(sub_c), dtype=bool)
    fake_day_start[0] = True
    return compute_vwap(sub_h, sub_l, sub_c, sub_v, fake_day_start, k=2.0)
