# Strategy Catalog — MOEX Alpha Discovery

## Strategy Descriptions

### S1: BB + Consensus Filter
**File:** `alpha/strategies/s01_bb_consensus.py`
**Filter:** Bollinger Bands (20,2)
**Logic:** Enter when price is OUTSIDE BB AND Kronos consensus agrees on direction.
- Price above upper BB + consensus short → short
- Price below lower BB + consensus long → long
**Best ct:** 0.8 (4/5 consensus)

### S2: BB Mean Reversion
**File:** `alpha/strategies/s02_bb_mean_rev.py`
**Filter:** Bollinger Bands (20,2)
**Logic:** Trade mean reversion from BB extremes with Kronos confirmation.
**Best ct:** 1.0 (unanimous 5/5)

### S5: BB Narrow (Low Vol Breakout)
**File:** `alpha/strategies/s05_bb_narrow.py`
**Filter:** Narrow Bollinger Bands (low volatility regime)
**Logic:** Enter when BB narrow + Kronos predicts directional move.
**Best ct:** 0.8

### S20: Order Block
**File:** `alpha/strategies/s20_order_block.py`
**Filter:** ICT Order Block detection
**Logic:** Detect institutional supply/demand zones via swing points.
**Best ct:** 1.0
**Note:** Very few trades (19 in 2026), but high Sharpe.

### S28: Volume-Filtered Order Block
**File:** `alpha/strategies/s28_vol_ob.py`
**Filter:** Volume-confirmed Order Block
**Logic:** OB only when volume spike confirms level significance.
**⚠️ Failed May 2026 retro check** — negative Sharpe in May.

### S34: VWAP + Order Block
**File:** `alpha/strategies/s34_vwap_ob.py`
**Filter:** VWAP confirmation + OB
**Logic:** OB at VWAP reaction zones.
**⚠️ Failed May 2026 retro check** — negative Sharpe in May.

### S38: Low-Vol Order Block
**File:** `alpha/strategies/s38_low_vol_ob.py`
**Filter:** Low volatility regime + OB
**Logic:** Enter OB only when volatility is compressed.
**Best ct:** 1.0

### Vanilla: Pure Consensus
**File:** `alpha/strategies/vanilla.py`
**Filter:** None (Kronos only)
**Logic:** Enter on unanimous consensus direction, TP/SL at q90/q10.
**Best ct:** 0.8 (more trades), 1.0 (higher quality)

## Calibration Results

| Parameter | Setting | Notes |
|---|---|---|
| consensus_threshold | 1.0 | Optimal for 6/7 strategies. S1 exception: ct=0.8 |
| TP/SL | q90/q10 from Kronos | Beats ATR in 5/6 strategies |
| save_first_n | 1 | Defers TP/SL by 1 candle, improves BB strategies |
| pred_len | 6 | 6 × 10min = 60min forecast horizon |
| sample_count | 5 | Limited by GPU. 20+ would improve quantile adaptation |

## Leakage Prevention

- BB uses only data BEFORE pred_ts (mask: `df.index < pred_ts`)
- Samples are stochastic predictions, not actuals (5 samples × 6 steps)
- Windows are non-overlapping (step == pred_len == 6)
- Train/Test temporal split: 2025 train, 2026 test
- TP/SL sim iterates step-by-step, no peeking at future actuals
- consensus() uses only sample predictions, never actuals
- All 8 leakage tests passed.

---

*Ported from kronos-artifact/STRATEGIES.md. Revision: 1.1.*
