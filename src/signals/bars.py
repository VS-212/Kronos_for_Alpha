"""
M-SIGNAL-BARS: Japanese candlestick pattern classifier
Contract: OHLCV arrays → bar type probabilities + signal
Status: ✅ ready
"""

"""Bar type classification and pattern probability engine for OHLCV data.

Classifies each bar into one of 15 types, then computes:
  1. P(next_up | bar_type) — probability of up move after this bar type
  2. Avg return after this bar type
  3. Signal strength based on probability edge vs 0.5

Used as additional signal/confirmation for OB/BB strategies.
"""

import numpy as np

# Bar types
DOJI = "doji"
MARUBOZU = "marubozu"
HAMMER = "hammer"
SHOOTING_STAR = "shooting_star"
BULL_ENGULF = "bullish_engulf"
BEAR_ENGULF = "bearish_engulf"
INSIDE = "inside"
OUTSIDE = "outside"
BULL_HARAMI = "bullish_harami"
BEAR_HARAMI = "bearish_harami"
LONG_LOWER = "long_lower"
LONG_UPPER = "long_upper"
BIG_BULL = "big_bull"
BIG_BEAR = "big_bear"
NEUTRAL = "neutral"

ALL_TYPES = [
    DOJI,
    MARUBOZU,
    HAMMER,
    SHOOTING_STAR,
    BULL_ENGULF,
    BEAR_ENGULF,
    INSIDE,
    OUTSIDE,
    BULL_HARAMI,
    BEAR_HARAMI,
    LONG_LOWER,
    LONG_UPPER,
    BIG_BULL,
    BIG_BEAR,
    NEUTRAL,
]


def classify_bar(o, h, l, c, po, ph, pl, pc) -> str:
    """Classify a single bar into a type based on OHLCV.

    Args:
        o, h, l, c: current bar OHLC
        po, ph, pl, pc: previous bar OHLC

    Returns:
        bar type string
    """
    body = abs(c - o)
    rng = max(h - l, 1e-10)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_ratio = body / rng

    # Doji: tiny body
    if body_ratio < 0.05:
        return DOJI

    # Marubozu: full body, no significant wicks
    if body_ratio > 0.9 and upper_wick / rng < 0.05 and lower_wick / rng < 0.05:
        return MARUBOZU

    prev_body = abs(pc - po)
    prev_rng = max(ph - pl, 1e-10)

    # Engulfing: current body > prev body + opposite direction
    if body > prev_body * 1.1:
        if c > o and o < pc and c > po:
            return BULL_ENGULF
        if o > c and o > pc and c < po:
            return BEAR_ENGULF

    # Bullish Harami: body inside prev body, bullish
    if prev_body > 0 and body < prev_body * 0.8:
        if c > o and o > po and c < pc:
            return BULL_HARAMI
        if o > c and po > o and pc > c:
            return BEAR_HARAMI

    # Inside Bar: range inside prev range
    if h < ph and l > pl:
        return INSIDE

    # Outside Bar: range engulfs prev range
    if h > ph and l < pl:
        return OUTSIDE

    # Long lower shadow: lower wick > 2 * body
    if lower_wick > 2 * body and upper_wick < body:
        return HAMMER

    # Long upper shadow: upper wick > 2 * body
    if upper_wick > 2 * body and lower_wick < body:
        return SHOOTING_STAR

    # Long lower shadow (generic)
    if lower_wick > 2 * body:
        return LONG_LOWER

    # Long upper shadow (generic)
    if upper_wick > 2 * body:
        return LONG_UPPER

    # Big bars: body > 2 * average... can't compute avg here, skip
    if body > 3 * prev_body and c > o:
        return BIG_BULL
    if body > 3 * prev_body and o > c:
        return BIG_BEAR

    return NEUTRAL


def compute_bar_probs(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    lookback: int = 1000,
    min_samples: int = 15,
) -> dict:
    """Compute forward probability and avg return for each bar type.

    For each bar (except last), classifies it, then records the next
    candle's return. Aggregates over lookback window.

    Returns:
        {bar_type: {"p_up": float, "p_down": float, "avg_ret": float,
                     "count": int}, ...}
    """
    n = len(close_arr)
    if n < 10:
        return {t: {"p_up": 0.5, "p_down": 0.5, "avg_ret": 0.0, "count": 0} for t in ALL_TYPES}

    start = max(2, n - lookback)
    counts = {t: {"total": 0, "up": 0, "sum_ret": 0.0} for t in ALL_TYPES}

    for i in range(start + 1, n - 1):
        o, h, l, c = (
            float(open_arr[i]),
            float(high_arr[i]),
            float(low_arr[i]),
            float(close_arr[i]),
        )
        po, ph, pl, pc = (
            float(open_arr[i - 1]),
            float(high_arr[i - 1]),
            float(low_arr[i - 1]),
            float(close_arr[i - 1]),
        )
        next_ret = float((close_arr[i + 1] - c) / max(c, 1e-10))

        bt = classify_bar(o, h, l, c, po, ph, pl, pc)
        counts[bt]["total"] += 1
        if next_ret > 0:
            counts[bt]["up"] += 1
        counts[bt]["sum_ret"] += next_ret

    probs = {}
    for t in ALL_TYPES:
        c = counts[t]
        if c["total"] >= min_samples:
            p_up = c["up"] / c["total"]
            probs[t] = {
                "p_up": float(p_up),
                "p_down": 1.0 - float(p_up),
                "avg_ret": float(c["sum_ret"] / c["total"]),
                "count": int(c["total"]),
            }
        else:
            probs[t] = {"p_up": 0.5, "p_down": 0.5, "avg_ret": 0.0, "count": int(c["total"])}
    return probs


def bar_signal(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    probs: dict,
    min_edge: float = 0.05,
) -> dict:
    """Get signal from bar type classification + pre-computed probabilities.

    Args:
        open_arr, high_arr, low_arr, close_arr: must include current + prev bar
        probs: from compute_bar_props()
        min_edge: minimum |p_up - 0.5| to generate a signal

    Returns:
        signal: "bullish" | "bearish" | "none"
        bar_type: classified bar type
        p_up: probability of up move
        avg_ret: average next-candle return for this bar type
    """
    n = len(close_arr)
    if n < 2:
        return {"signal": "none", "bar_type": NEUTRAL, "p_up": 0.5, "avg_ret": 0.0}

    o, h, l, c = (
        float(open_arr[-1]),
        float(high_arr[-1]),
        float(low_arr[-1]),
        float(close_arr[-1]),
    )
    po, ph, pl, pc = (
        float(open_arr[-2]),
        float(high_arr[-2]),
        float(low_arr[-2]),
        float(close_arr[-2]),
    )

    bt = classify_bar(o, h, l, c, po, ph, pl, pc)
    prob = probs.get(bt, {"p_up": 0.5, "avg_ret": 0.0})
    p_up = prob["p_up"]
    edge = p_up - 0.5

    if edge > min_edge:
        return {"signal": "bullish", "bar_type": bt, "p_up": p_up, "avg_ret": prob["avg_ret"]}
    if edge < -min_edge:
        return {"signal": "bearish", "bar_type": bt, "p_up": p_up, "avg_ret": prob["avg_ret"]}

    return {"signal": "none", "bar_type": bt, "p_up": p_up, "avg_ret": prob["avg_ret"]}


def print_bar_summary(probs: dict):
    """Pretty-print bar type probabilities."""
    print(f"{'Bar Type':<20} {'Count':>6} {'P(up)':>7} {'AvgRet':>9}")
    print("-" * 44)
    for t in ALL_TYPES:
        p = probs[t]
        if p["count"] > 0:
            marker = "  ←" if abs(p["p_up"] - 0.5) > 0.03 else ""
            print(f"{t:<20} {p['count']:>6} {p['p_up']:>6.3f} {p['avg_ret']:>+7.4f}{marker}")
    print()
