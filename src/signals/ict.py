"""
M-SIGNAL-ICT: ICT/SMC signal generation toolkit
Contract: OHLCV arrays → dict of ICT signals (order block, FVG, liquidity sweep, MSS, etc.)
Status: ✅ ready
"""

"""ICT (Inner Circle Trader) / Smart Money Concepts for OHLCV data.

Combined with Williams Fractal/swing detection for confluence.

Detectors:
  - detect_swings: 3/5-bar ICT swings
  - detect_order_block: last opposing candle before strong move
  - detect_fvg: fair value gap (3-candle imbalance)
  - detect_liquidity_sweep: fractal broken + price reversed
  - detect_premium_discount: price within fractal-defined range
  - detect_mss: market structure shift via fractal breakout
  - detect_breaker_block: broken OB → reversed signal
  - detect_volume_ob: OB filtered by volume
  - detect_eqh_eql: equal swing levels breakout
"""

import numpy as np

# ── Helpers ─────────────────────────────────────────────────────────────


def _group_by_level(levels: np.ndarray, tolerance: float) -> list:
    """Group price levels within tolerance. Returns [(avg_level, [indices]), ...]."""
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


# ── ICT Swings ──────────────────────────────────────────────────────────


def detect_swings(high: np.ndarray, low: np.ndarray, n_bars: int = 3) -> dict:
    """Detect swing highs/lows (ICT-style). n_bars=3 (default) or 5 (Williams)."""
    n = len(high)
    half = n_bars // 2
    bullish_idx, bearish_idx = [], []
    bullish_levels, bearish_levels = [], []

    for i in range(half, n - half):
        if low[i] == min(low[i - half : i + half + 1]):
            bullish_idx.append(i)
            bullish_levels.append(low[i])
        if high[i] == max(high[i - half : i + half + 1]):
            bearish_idx.append(i)
            bearish_levels.append(high[i])

    return {
        "bullish_idx": np.array(bullish_idx, dtype=np.int32),
        "bearish_idx": np.array(bearish_idx, dtype=np.int32),
        "bullish_levels": np.array(bullish_levels, dtype=np.float64),
        "bearish_levels": np.array(bearish_levels, dtype=np.float64),
    }


def _recent_swings(high: np.ndarray, low: np.ndarray, n_bars: int = 3, max_age: int = 48) -> dict:
    """Get the most recent bullish and bearish swing within max_age."""
    sw = detect_swings(high, low, n_bars)
    last_n = len(high) - 1
    result = {"bullish": None, "bearish": None}

    for idx, lvl in reversed(list(zip(sw["bullish_idx"], sw["bullish_levels"]))):
        if last_n - idx <= max_age:
            result["bullish"] = {"idx": int(idx), "level": float(lvl), "age": int(last_n - idx)}
            break
    for idx, lvl in reversed(list(zip(sw["bearish_idx"], sw["bearish_levels"]))):
        if last_n - idx <= max_age:
            result["bearish"] = {"idx": int(idx), "level": float(lvl), "age": int(last_n - idx)}
            break
    return result


# ── Order Block ─────────────────────────────────────────────────────────


def detect_order_block(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 48,
    move_threshold: float = 0.003,
    max_age: int = 24,
) -> dict:
    """Find the most recent order block (ICT).

    Bullish OB: a bearish candle followed by strong up move.
    Bearish OB: a bullish candle followed by strong down move.
    Returns signal, ob_high, ob_low, age, idx (candle position).
    """
    n = len(close)
    if n < 3:
        return {"signal": "none", "ob_high": 0, "ob_low": 0, "age": -1, "idx": -1}
    start = max(1, n - lookback)

    for i in range(n - 2, start - 1, -1):
        if i + 1 >= n:
            continue
        if close[i] < open[i]:  # Bearish candle
            move_up = (high[i + 1] - low[i + 1]) / max(close[i], 1e-10)
            if move_up >= move_threshold:
                age = (n - 1) - i
                if age <= max_age:
                    return {
                        "signal": "bullish",
                        "ob_high": float(high[i]),
                        "ob_low": float(low[i]),
                        "age": age,
                        "idx": i,
                    }
        elif close[i] > open[i]:  # Bullish candle
            move_down = (high[i + 1] - low[i + 1]) / max(close[i], 1e-10)
            if move_down >= move_threshold:
                age = (n - 1) - i
                if age <= max_age:
                    return {
                        "signal": "bearish",
                        "ob_high": float(high[i]),
                        "ob_low": float(low[i]),
                        "age": age,
                        "idx": i,
                    }

    return {"signal": "none", "ob_high": 0, "ob_low": 0, "age": -1, "idx": -1}


# ── Fair Value Gap ──────────────────────────────────────────────────────


def detect_fvg(high: np.ndarray, low: np.ndarray, lookback: int = 48, max_age: int = 24) -> dict:
    """Find the most recent Fair Value Gap (ICT)."""
    n = len(high)
    if n < 3:
        return {"signal": "none", "fvg_high": 0, "fvg_low": 0, "age": -1}
    start = max(2, n - lookback)

    for i in range(n - 2, start - 1, -1):
        if i + 1 >= n or i - 1 < 0:
            continue
        if low[i - 1] > high[i + 1]:
            age = (n - 1) - i
            if age <= max_age:
                return {
                    "signal": "bullish",
                    "fvg_high": float(low[i - 1]),
                    "fvg_low": float(high[i + 1]),
                    "age": age,
                }
        if high[i - 1] < low[i + 1]:
            age = (n - 1) - i
            if age <= max_age:
                return {
                    "signal": "bearish",
                    "fvg_high": float(low[i + 1]),
                    "fvg_low": float(high[i - 1]),
                    "age": age,
                }

    return {"signal": "none", "fvg_high": 0, "fvg_low": 0, "age": -1}


def detect_fvg_multi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 48,
    min_fvgs: int = 2,
    max_age: int = 48,
) -> dict:
    """Multi-Fair Value Gap: finds confluence of ≥ min_fvgs overlapping FVG zones.

    A "confluence zone" is the intersection of multiple FVG zones at the same
    price level. Stronger than a single FVG — multiple imbalances at the same
    level act as stronger support/resistance.

    For each signal direction (bullish/bearish), finds overlapping FVG zones,
    computes their intersection, and counts how many zones cover it.
    If ≥ min_fvgs zones overlap and current close is inside → signal.

    Returns:
        signal: "bullish" | "bearish" | "none"
        confluence: number of overlapping FVGs
        fvg_low, fvg_high: the confluence zone boundaries
        age: max age of the most recent contributing FVG
    """
    n = len(high)
    if n < 3:
        return {"signal": "none", "confluence": 0, "fvg_low": 0, "fvg_high": 0, "age": -1}

    start = max(2, n - lookback)
    bull_zones = []  # [(low, high, age), ...]
    bear_zones = []

    for i in range(n - 2, start - 1, -1):
        age = (n - 1) - i
        if age > max_age:
            continue
        if low[i - 1] > high[i + 1]:
            bull_zones.append((float(high[i + 1]), float(low[i - 1]), age))
        if high[i - 1] < low[i + 1]:
            bear_zones.append((float(high[i - 1]), float(low[i + 1]), age))

    current_close = float(close[-1])

    def _best_confluence(zones, current_close, min_fvgs):
        if len(zones) < min_fvgs:
            return None
        zones.sort()
        best = {"overlaps": 0, "low": 0, "high": 0, "age": 0}
        for i in range(len(zones)):
            z_low_i, z_high_i, age_i = zones[i]
            for j in range(i + 1, len(zones)):
                z_low_j, z_high_j, age_j = zones[j]
                int_low = max(z_low_i, z_low_j)
                int_high = min(z_high_i, z_high_j)
                if int_low >= int_high:
                    continue
                count = 0
                min_age = 999
                for z in zones:
                    if z[0] < int_high and z[1] > int_low:
                        count += 1
                        min_age = min(min_age, z[2])
                if count >= min_fvgs and count > best["overlaps"]:
                    if int_low <= current_close <= int_high:
                        best = {"overlaps": count, "low": int_low, "high": int_high, "age": min_age}
        return best if best["overlaps"] > 0 else None

    b = _best_confluence(bull_zones, current_close, min_fvgs)
    if b:
        return {
            "signal": "bullish",
            "confluence": b["overlaps"],
            "fvg_low": b["low"],
            "fvg_high": b["high"],
            "age": b["age"],
        }

    b = _best_confluence(bear_zones, current_close, min_fvgs)
    if b:
        return {
            "signal": "bearish",
            "confluence": b["overlaps"],
            "fvg_low": b["low"],
            "fvg_high": b["high"],
            "age": b["age"],
        }

    return {"signal": "none", "confluence": 0, "fvg_low": 0, "fvg_high": 0, "age": -1}


# ── Liquidity Sweep ────────────────────────────────────────────────────


def detect_liquidity_sweep(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 48,
    fractal_n_bars: int = 3,
    fractal_max_age: int = 24,
) -> dict:
    """Detect liquidity sweep using fractal swings."""
    n = len(high)
    if n < 5:
        return {"signal": "none", "sweep_level": 0, "age": -1}

    sw = _recent_swings(high, low, fractal_n_bars, fractal_max_age)
    current_close = float(close[-1])

    if sw["bullish"] is not None:
        swing_low = sw["bullish"]["level"]
        min_c = float(np.min(close[sw["bullish"]["idx"] :]))
        if min_c < swing_low and current_close > swing_low:
            return {
                "signal": "bullish",
                "sweep_level": swing_low,
                "age": int((n - 1) - sw["bullish"]["idx"]),
            }

    if sw["bearish"] is not None:
        swing_high = sw["bearish"]["level"]
        max_c = float(np.max(close[sw["bearish"]["idx"] :]))
        if max_c > swing_high and current_close < swing_high:
            return {
                "signal": "bearish",
                "sweep_level": swing_high,
                "age": int((n - 1) - sw["bearish"]["idx"]),
            }

    return {"signal": "none", "sweep_level": 0, "age": -1}


# ── Premium / Discount ─────────────────────────────────────────────────


def detect_premium_discount(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 96
) -> dict:
    """Detect if price is in premium or discount zone of swing range."""
    n = len(high)
    sw = _recent_swings(high, low, n_bars=3, max_age=lookback)
    if sw["bullish"] is None or sw["bearish"] is None:
        return {"zone": "none", "range_low": 0, "range_high": 0, "pct_pos": 0.5, "age": -1}

    swing_low = sw["bullish"]["level"]
    swing_high = sw["bearish"]["level"]
    rng = swing_high - swing_low
    if rng <= 0:
        return {
            "zone": "none",
            "range_low": swing_low,
            "range_high": swing_high,
            "pct_pos": 0.5,
            "age": -1,
        }

    pct_pos = max(0.0, min(1.0, (float(close[-1]) - swing_low) / rng))
    zone = "discount" if pct_pos < 0.5 else "premium" if pct_pos > 0.5 else "none"
    return {
        "zone": zone,
        "range_low": swing_low,
        "range_high": swing_high,
        "pct_pos": pct_pos,
        "age": max(sw["bullish"]["age"], sw["bearish"]["age"]),
    }


# ── Market Structure Shift ─────────────────────────────────────────────


def detect_mss(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 48,
    fractal_max_age: int = 24,
    n_bars: int = 3,
) -> dict:
    """Market Structure Shift: close above/below a fractal level."""
    n = len(high)
    if n < 5:
        return {"signal": "none", "level": 0, "age": -1}

    current_close = float(close[-1])
    sw = _recent_swings(high, low, n_bars, fractal_max_age)

    if sw["bearish"] is not None and current_close > sw["bearish"]["level"]:
        return {"signal": "bullish", "level": sw["bearish"]["level"], "age": sw["bearish"]["age"]}
    if sw["bullish"] is not None and current_close < sw["bullish"]["level"]:
        return {"signal": "bearish", "level": sw["bullish"]["level"], "age": sw["bullish"]["age"]}

    return {"signal": "none", "level": 0, "age": -1}


# ── Breaker Block ───────────────────────────────────────────────────────


def detect_breaker_block(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 48,
    move_threshold: float = 0.002,
    max_age: int = 24,
) -> dict:
    """Breaker Block: a broken OB zone becomes a reversed signal.

    Bullish OB broken below → Bearish Breaker (sell retest).
    Bearish OB broken above → Bullish Breaker (buy retest).
    """
    n = len(close)
    if n < 3:
        return {"signal": "none", "level": 0, "age": -1}

    ob = detect_order_block(open, high, low, close, lookback, move_threshold, max_age)
    if ob["signal"] == "none":
        return {"signal": "none", "level": 0, "age": -1}

    idx = ob["idx"]
    ob_low, ob_high = ob["ob_low"], ob["ob_high"]
    current_close = float(close[-1])
    close_since_ob = close[idx:]
    zone_height = ob_high - ob_low

    if ob["signal"] == "bullish":
        min_c = float(np.min(close_since_ob))
        if min_c < ob_low:
            ba = int((n - 1) - (idx + int(np.argmin(close_since_ob))))
            within = abs(current_close - ob_low) <= zone_height
            if within and ba <= max_age * 2:
                return {
                    "signal": "bearish",
                    "level": float(ob_low),
                    "age": ob["age"],
                    "breaker_age": ba,
                }
    else:
        max_c = float(np.max(close_since_ob))
        if max_c > ob_high:
            ba = int((n - 1) - (idx + int(np.argmax(close_since_ob))))
            within = abs(ob_high - current_close) <= zone_height
            if within and ba <= max_age * 2:
                return {
                    "signal": "bullish",
                    "level": float(ob_high),
                    "age": ob["age"],
                    "breaker_age": ba,
                }

    return {"signal": "none", "level": 0, "age": -1, "breaker_age": -1}


# ── Volume-filtered OB ──────────────────────────────────────────────────


def detect_volume_ob(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 48,
    move_threshold: float = 0.002,
    max_age: int = 24,
    volume_mult: float = 1.5,
) -> dict:
    """Order Block filtered by volume — only OB with volume > avg × mult."""
    ob = detect_order_block(open, high, low, close, lookback, move_threshold, max_age)
    if ob["signal"] == "none" or ob["idx"] < 0:
        return {"signal": "none", "ob_high": 0, "ob_low": 0, "age": -1, "idx": -1}

    start = max(0, len(volume) - lookback)
    avg_vol = float(np.mean(volume[start:]))
    if avg_vol > 0 and volume[ob["idx"]] < avg_vol * volume_mult:
        return {"signal": "none", "ob_high": 0, "ob_low": 0, "age": -1, "idx": -1}

    return ob


# ── Equal Highs / Equal Lows ───────────────────────────────────────────


def detect_eqh_eql(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 96,
    n_bars: int = 3,
    tolerance: float = 0.001,
    max_age: int = 48,
) -> dict:
    """Equal Highs/Lows: 2+ swings at same level formed resistance/support."""
    n = len(high)
    if n < 5:
        return {"signal": "none", "level": 0, "age": -1}

    sw = detect_swings(high, low, n_bars)
    current_close = float(close[-1])
    start = max(0, n - lookback)

    mask_bear = sw["bearish_idx"] >= start
    for lvl, _ in reversed(_group_by_level(sw["bearish_levels"][mask_bear], tolerance)):
        if current_close > lvl:
            return {"signal": "bullish", "level": float(lvl), "age": 0}

    mask_bull = sw["bullish_idx"] >= start
    for lvl, _ in reversed(_group_by_level(sw["bullish_levels"][mask_bull], tolerance)):
        if current_close < lvl:
            return {"signal": "bearish", "level": float(lvl), "age": 0}

    return {"signal": "none", "level": 0, "age": -1}
