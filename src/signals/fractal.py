"""
M-SIGNAL-FRACTAL: Williams Fractal pattern detection
Contract: OHLCV arrays → fractal signals (breakout, cluster, AO-confirmed)
Status: ✅ ready
"""

"""Williams Fractal detection and related indicators for OHLCV data.

Strategies:
  1. Fractal Breakout — price breaks through a fractal level
  2. Fractal Cluster — 2+ fractals at same price level = stronger signal
  3. Fractal + Awesome Oscillator — fractal confirmed by AO momentum
"""

import numpy as np

# ── Core Fractal Detection ──────────────────────────────────────────────


def find_fractals(high: np.ndarray, low: np.ndarray) -> dict:
    """Detect all Williams Fractals in a price series.

    Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    """
    n = len(high)
    b_idx, B_idx = [], []
    b_lvl, B_lvl = [], []

    for i in range(2, n - 2):
        if (
            low[i] < low[i - 2]
            and low[i] < low[i - 1]
            and low[i] < low[i + 1]
            and low[i] < low[i + 2]
        ):
            b_idx.append(i)
            b_lvl.append(low[i])

        if (
            high[i] > high[i - 2]
            and high[i] > high[i - 1]
            and high[i] > high[i + 1]
            and high[i] > high[i + 2]
        ):
            B_idx.append(i)
            B_lvl.append(high[i])

    return {
        "bullish_idx": np.array(b_idx, dtype=np.int32),
        "bearish_idx": np.array(B_idx, dtype=np.int32),
        "bullish_levels": np.array(b_lvl, dtype=np.float64),
        "bearish_levels": np.array(B_lvl, dtype=np.float64),
    }


def _most_recent(fractals: dict, n: int, max_age: int):
    """Return (signal, age, level) of the most recent fractal within max_age."""
    last_n = n - 1
    # Bearish first (more recent → higher index)
    for idx, lvl in reversed(list(zip(fractals["bearish_idx"], fractals["bearish_levels"]))):
        age = last_n - idx
        if age <= max_age:
            return "bearish", int(age), float(lvl)
    for idx, lvl in reversed(list(zip(fractals["bullish_idx"], fractals["bullish_levels"]))):
        age = last_n - idx
        if age <= max_age:
            return "bullish", int(age), float(lvl)
    return "none", -1, 0.0


def fractal_signal(high: np.ndarray, low: np.ndarray, max_age: int = 12) -> dict:
    """Most recent fractal signal (regardless of price action)."""
    n = len(high)
    if n < 5:
        return {"signal": "none", "age": -1, "level": 0.0}
    sig, age, lvl = _most_recent(find_fractals(high, low), n, max_age)
    return {"signal": sig, "age": age, "level": lvl}


# ── Fractal Breakout ────────────────────────────────────────────────────


def breakout_signal(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, max_age: int = 12
) -> dict:
    """Fractal breakout: price breaks THROUGH a fractal level.

    Bullish breakout: price closes above the most recent bearish fractal high
        (resistance broken → momentum up).
    Bearish breakdown: price closes below the most recent bullish fractal low
        (support broken → momentum down).

    Returns:
        signal: "bullish" | "bearish" | "none"
        age: candles since the triggering fractal
        level: the fractal level that was broken
    """
    n = len(high)
    if n < 5:
        return {"signal": "none", "age": -1, "level": 0.0}

    fractals = find_fractals(high, low)
    current_close = float(close[-1])
    last_n = n - 1

    # Check: has price broken above the most recent bearish fractal high?
    for idx, lvl in reversed(list(zip(fractals["bearish_idx"], fractals["bearish_levels"]))):
        age = last_n - idx
        if age <= max_age and current_close > lvl:
            return {"signal": "bullish", "age": int(age), "level": float(lvl)}

    # Check: has price broken below the most recent bullish fractal low?
    for idx, lvl in reversed(list(zip(fractals["bullish_idx"], fractals["bullish_levels"]))):
        age = last_n - idx
        if age <= max_age and current_close < lvl:
            return {"signal": "bearish", "age": int(age), "level": float(lvl)}

    return {"signal": "none", "age": -1, "level": 0.0}


# ── Fractal Cluster ─────────────────────────────────────────────────────


def _group_by_level(levels: np.ndarray, tolerance: float) -> list:
    """Group fractal indices by proximity in price level."""
    if len(levels) == 0:
        return []
    sorted_idx = np.argsort(levels)
    sorted_lvls = levels[sorted_idx]
    clusters = []
    current = [sorted_idx[0]]
    for i in range(1, len(sorted_lvls)):
        pct_diff = abs(sorted_lvls[i] - sorted_lvls[i - 1]) / max(abs(sorted_lvls[i - 1]), 1e-10)
        if pct_diff <= tolerance:
            current.append(sorted_idx[i])
        else:
            if len(current) >= 2:
                clusters.append((float(np.mean([levels[j] for j in current])), list(current)))
            current = [sorted_idx[i]]
    if len(current) >= 2:
        clusters.append((float(np.mean([levels[j] for j in current])), list(current)))
    return clusters


def cluster_signal(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    max_age: int = 24,
    tolerance: float = 0.002,
    min_cluster: int = 2,
) -> dict:
    """Fractal cluster: 2+ fractals at the same level = strong S/R.

    Returns bullish signal if a bearish cluster (resistance) is broken to the upside,
    or bearish signal if a bullish cluster (support) is broken to the downside.
    """
    n = len(high)
    if n < 5:
        return {"signal": "none", "age": -1, "level": 0.0}

    fractals = find_fractals(high, low)
    current_close = float(close[-1])
    start = max(0, n - max_age)

    # Filter fractals within max_age window
    mask_bull = fractals["bullish_idx"] >= start
    mask_bear = fractals["bearish_idx"] >= start

    bear_clusters = _group_by_level(fractals["bearish_levels"][mask_bear], tolerance)
    bull_clusters = _group_by_level(fractals["bullish_levels"][mask_bull], tolerance)

    # Breakout above bearish cluster (resistance broken → up)
    for lvl, idxs in reversed(bear_clusters):
        if len(idxs) >= min_cluster and current_close > lvl:
            return {"signal": "bullish", "age": 0, "level": lvl}

    # Breakdown below bullish cluster (support broken → down)
    for lvl, idxs in reversed(bull_clusters):
        if len(idxs) >= min_cluster and current_close < lvl:
            return {"signal": "bearish", "age": 0, "level": lvl}

    return {"signal": "none", "age": -1, "level": 0.0}


# ── Awesome Oscillator ──────────────────────────────────────────────────


def compute_ao(high: np.ndarray, low: np.ndarray) -> float:
    """Awesome Oscillator: SMA5(median) - SMA34(median).

    AO > 0 = upward momentum, AO < 0 = downward momentum.
    Returns the last value.
    """
    median = (high + low) / 2.0
    n = len(median)
    if n < 34:
        return 0.0
    sma5 = np.mean(median[-5:])
    sma34 = np.mean(median[-34:])
    return float(sma5 - sma34)


def ao_fractal_signal(high: np.ndarray, low: np.ndarray, max_age: int = 12) -> dict:
    """Fractal + AO: combine fractal signal with AO direction.

    Returns signal only when fractal and AO agree on direction.
    """
    n = len(high)
    if n < 34:
        return {"signal": "none", "age": -1, "level": 0.0, "ao": 0.0}

    fs = fractal_signal(high, low, max_age)
    if fs["signal"] == "none":
        return {"signal": "none", "age": -1, "level": 0.0, "ao": 0.0}

    ao_val = compute_ao(high, low)
    fs["ao"] = ao_val

    if (fs["signal"] == "bullish" and ao_val > 0) or (fs["signal"] == "bearish" and ao_val < 0):
        return fs

    return {"signal": "none", "age": -1, "level": 0.0, "ao": ao_val}
