"""
M-SIGNAL-DIV: RSI/OBV/MFI divergence detection
Contract: OHLCV arrays + is_day_start → indicator values + divergence flags
Status: ✅ ready
"""

"""Divergence detection atoms for OHLCV data.

Each atom takes raw price/volume arrays (not Kronos samples) and returns a dict
of derived indicators + divergence flags. Pure numpy — no pandas.
"""

import numpy as np


def rsi(close: np.ndarray, period: int = 14, is_day_start: np.ndarray = None) -> dict:
    """Intraday RSI with daily reset — no gap contamination.

    At each day boundary, RSI resets and builds from intraday moves only.
    First candle of each day is a fresh start (no delta from previous close).

    Args:
        close: (N,) float64 — close prices
        period: RSI lookback window (default 14)
        is_day_start: (N,) bool — True at first candle of each trading day.
                      If None, treats entire array as one contiguous day.

    Returns:
        rsi: float — last RSI value
        rsi_series: (N,) — full RSI series (NaN for first period of each day)
        oversold: bool — rsi < 30
        overbought: bool — rsi > 70
    """
    close = close.astype(np.float64)
    n = len(close)
    rsi_series = np.full(n, np.nan)

    if n < period + 1:
        return {"rsi": np.nan, "rsi_series": rsi_series, "oversold": False, "overbought": False}

    if is_day_start is None or not is_day_start.any():
        day_starts = np.array([0])
    else:
        day_starts = np.where(is_day_start)[0]
        if day_starts[0] != 0:
            day_starts = np.concatenate([[0], day_starts])

    for idx, start in enumerate(day_starts):
        end = day_starts[idx + 1] if idx + 1 < len(day_starts) else n
        seg = close[start:end]
        seg_len = len(seg)
        if seg_len < period + 1:
            continue

        delta = np.diff(seg)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.full(seg_len, np.nan)
        avg_loss = np.full(seg_len, np.nan)

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        rsi_series[start + period] = 100.0 - 100.0 / (
            1.0 + avg_gain[period] / max(avg_loss[period], 1e-10)
        )

        for i in range(period + 1, seg_len):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
            r_s = avg_gain[i] / max(avg_loss[i], 1e-10)
            rsi_series[start + i] = 100.0 - 100.0 / (1.0 + r_s)

    last_rsi = float(rsi_series[-1])
    return {
        "rsi": last_rsi,
        "rsi_series": rsi_series,
        "oversold": last_rsi < 30.0,
        "overbought": last_rsi > 70.0,
    }


def obv_data(close: np.ndarray, volume: np.ndarray) -> dict:
    """On-Balance Volume with trailing slope.

    Args:
        close: (N,) — close prices
        volume: (N,) — volume values

    Returns:
        obv_series: (N,) — full OBV
        obv_slope: float — slope of last 5 OBV values (linear regression)
    """
    obv = np.zeros_like(close, dtype=np.float64)
    obv[0] = volume[0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    n = 5
    seg = obv[-n:]
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = seg.mean()
    num = ((x - x_mean) * (seg - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    slope = float(num / max(den, 1e-10))

    return {
        "obv_series": obv,
        "obv_slope": slope,
    }


def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
    is_day_start: np.ndarray = None,
) -> dict:
    """Money Flow Index with daily reset.

    Args:
        high: (N,) — high prices
        low: (N,) — low prices
        close: (N,) — close prices
        volume: (N,) — volume
        period: lookback window (default 14)
        is_day_start: (N,) bool — True at first candle of each trading day.

    Returns:
        mfi: float — last MFI value
        mfi_series: (N,) — full MFI series (NaN for first period of each day)
    """
    n = len(close)
    mfi_series = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return {"mfi": np.nan, "mfi_series": mfi_series}

    if is_day_start is None or not is_day_start.any():
        day_starts = np.array([0])
    else:
        day_starts = np.where(is_day_start)[0]
        if day_starts[0] != 0:
            day_starts = np.concatenate([[0], day_starts])

    typical = (high + low + close) / 3.0
    raw_mf = typical * volume

    for idx, start in enumerate(day_starts):
        end = day_starts[idx + 1] if idx + 1 < len(day_starts) else n
        seg_len = end - start
        if seg_len < period + 1:
            continue

        for i in range(start + period, end):
            pos = 0.0
            neg = 0.0
            for j in range(i - period, i):
                if typical[j + 1] > typical[j]:
                    pos += raw_mf[j + 1]
                else:
                    neg += raw_mf[j + 1]
            mf_ratio = pos / max(neg, 1e-10)
            mfi_series[i] = 100.0 - 100.0 / (1.0 + mf_ratio)

    return {
        "mfi": float(mfi_series[-1]),
        "mfi_series": mfi_series,
    }


def _find_peaks(series: np.ndarray, order: int = 2) -> np.ndarray:
    """Find indices of local peaks (swing highs).

    A point is a peak if it is strictly higher than `order` neighbours
    on both sides. Only considers non-NaN values.
    """
    n = len(series)
    if n < 2 * order + 1:
        return np.array([], dtype=np.intp)
    peaks = []
    for i in range(order, n - order):
        if np.isnan(series[i]):
            continue
        left = series[i - order : i]
        right = series[i + 1 : i + order + 1]
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            continue
        if series[i] > left.max() and series[i] > right.max():
            peaks.append(i)
    return np.array(peaks, dtype=np.intp)


def _find_valleys(series: np.ndarray, order: int = 2) -> np.ndarray:
    """Find indices of local valleys (swing lows)."""
    n = len(series)
    if n < 2 * order + 1:
        return np.array([], dtype=np.intp)
    valleys = []
    for i in range(order, n - order):
        if np.isnan(series[i]):
            continue
        left = series[i - order : i]
        right = series[i + 1 : i + order + 1]
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            continue
        if series[i] < left.min() and series[i] < right.min():
            valleys.append(i)
    return np.array(valleys, dtype=np.intp)


def detect_divergence(
    price: np.ndarray, indicator: np.ndarray, lookback: int = 24, order: int = 2
) -> dict:
    """Detect classic price-indicator divergences using swing point analysis.

    Finds local peaks/valleys in both series, then compares the last two:
      - Bearish: price makes higher high (HH), indicator makes lower high (LH)
      - Bullish: price makes lower low (LL), indicator makes higher low (HL)

    Args:
        price: (N,) — price series (e.g. close)
        indicator: (N,) — indicator series (e.g. RSI)
        lookback: window to examine (default 24 candles)
        order: neighbours per side for swing detection (default 2)

    Returns:
        bearish_div: bool
        bullish_div: bool
        div_type: str — "bearish", "bullish", or "none"
    """
    if len(price) < lookback or len(indicator) < lookback:
        return {"bearish_div": False, "bullish_div": False, "div_type": "none"}

    seg_p = price[-lookback:]
    seg_i = indicator[-lookback:]

    peaks = _find_peaks(seg_p, order)
    if len(peaks) >= 2:
        p2, p1 = peaks[-2:]  # previous peak, current peak
        price_hh = seg_p[p1] > seg_p[p2]
        ind_lh = seg_i[p1] < seg_i[p2]
        bearish = bool(price_hh and ind_lh)
    else:
        bearish = False

    valleys = _find_valleys(seg_p, order)
    if len(valleys) >= 2:
        v2, v1 = valleys[-2:]  # previous valley, current valley
        price_ll = seg_p[v1] < seg_p[v2]
        ind_hl = seg_i[v1] > seg_i[v2]
        bullish = bool(price_ll and ind_hl)
    else:
        bullish = False

    if bearish:
        d_type = "bearish"
    elif bullish:
        d_type = "bullish"
    else:
        d_type = "none"

    return {
        "bearish_div": bearish,
        "bullish_div": bullish,
        "div_type": d_type,
    }


def compute_all(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    is_day_start: np.ndarray = None,
) -> dict:
    """Convenience wrapper: compute RSI, OBV, MFI and detect divergences.

    All divergence checks use the last 24 candles.

    Args:
        high: (N,) — high prices
        low: (N,) — low prices
        close: (N,) — close prices
        volume: (N,) — volume
        is_day_start: (N,) bool — True at first candle of each trading day.
                      Passed to RSI and MFI for daily reset.

    Returns:
        dict with all indicator values and divergence flags
    """
    r = rsi(close, is_day_start=is_day_start)
    o = obv_data(close, volume)
    m = mfi(high, low, close, volume, is_day_start=is_day_start)

    obv_s = o["obv_series"]
    div_close_vs_rsi = detect_divergence(close, r["rsi_series"], lookback=24)
    div_close_vs_obv = detect_divergence(close, obv_s, lookback=24)
    div_close_vs_mfi = detect_divergence(close, m["mfi_series"], lookback=24)

    return {
        "rsi": r["rsi"],
        "rsi_series": r["rsi_series"],
        "oversold": r["oversold"],
        "overbought": r["overbought"],
        "obv_series": o["obv_series"],
        "obv_slope": o["obv_slope"],
        "mfi": m["mfi"],
        "mfi_series": m["mfi_series"],
        "div_rsi": div_close_vs_rsi["div_type"],
        "div_obv": div_close_vs_obv["div_type"],
        "div_mfi": div_close_vs_mfi["div_type"],
        "bearish_div_rsi": div_close_vs_rsi["bearish_div"],
        "bullish_div_rsi": div_close_vs_rsi["bullish_div"],
        "bearish_div_obv": div_close_vs_obv["bearish_div"],
        "bullish_div_obv": div_close_vs_obv["bullish_div"],
        "bearish_div_mfi": div_close_vs_mfi["bearish_div"],
        "bullish_div_mfi": div_close_vs_mfi["bullish_div"],
    }
