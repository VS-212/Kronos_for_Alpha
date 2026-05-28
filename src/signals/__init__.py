"""
M-SIGNALS: Signal family catalog
Contract: read-only reference — lists all available signal families
Status: ✅ ready

Usage:
  import src.signals  # triggers catalog registration
  from src.signals.atoms import consensus, boundaries
  from src.signals.bollinger import bb_signal
  from src.signals.ict import detect_order_block
"""

# ── Signal Families (8 total) ──────────────────────────────────────────
#
# 1. ATOMS        — Composable signal primitives from Kronos prediction samples
#    Functions: direction, consensus, boundaries, dispersion, trend_strength,
#               linearity, asymmetry, expectancy
#    File: src/signals/atoms.py
#
# 2. BOLLINGER    — Bollinger Bands (+/-k*std around SMA)
#    Functions: compute_bb, bb_position, bb_squeeze, bb_signal
#    File: src/signals/bollinger.py
#    Note: Most effective filter — BB extremes + consensus direction (S01)
#
# 3. ICT          — Inner Circle Trader / Smart Money Concepts toolkit
#    Functions: detect_swings, detect_order_block, detect_fvg, detect_fvg_multi,
#               detect_liquidity_sweep, detect_premium_discount, detect_mss,
#               detect_breaker_block, detect_volume_ob, detect_eqh_eql
#    File: src/signals/ict.py
#
# 4. VOLATILITY   — Volatility regime and risk signals
#    Functions: compute_atr, compute_adr, volatility_regime
#    File: src/signals/volatility.py
#
# 5. VWAP         — Volume-Weighted Average Price signals
#    Functions: compute_vwap, anchored_vwap, vwap_cross
#    File: src/signals/vwap.py
#
# 6. FRACTAL      — Williams Fractal pattern detection
#    Functions: find_fractals, fractal_signal, breakout_signal,
#               cluster_signal, compute_ao, ao_fractal_signal
#    File: src/signals/fractal.py
#
# 7. DIVERGENCE   — RSI/OBV/MFI divergence detection
#    Functions: rsi, obv_data, mfi, detect_divergence, compute_all
#    File: src/signals/divergence.py
#
# 8. BARS         — Japanese candlestick pattern classifier (15 types)
#    Functions: classify_bar, compute_bar_probs, bar_signal
#    File: src/signals/bars.py
