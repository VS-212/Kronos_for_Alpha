# Leakage Audit Report

## Test Results: ✅ 8/8 PASSED

| # | Test | Result | Detail |
|---|------|--------|--------|
| 1 | BB lookback | ✅ | `lookup_bb()` filters `mamba.index < pred_ts` — no future data |
| 2 | Sample predictions | ✅ | Samples are stochastic (std ~0.9), not actuals — no leakage from ground truth |
| 3 | Window non-overlap | ✅ | step=pred_len=6 → no overlapping data between windows |
| 4 | Mamba lookups | ✅ | All detector functions receive only pre-pred_ts historical data |
| 5 | Train/Test split | ✅ | Strict temporal split: 2025 train (max 2025-12-30), 2026 test (min 2026-01-05) |
| 6 | Walk-forward input | ✅ | x_df = candles [i:i+2048], y_df = candles [i+2048:i+2054] — no overlap |
| 7 | Consensus | ✅ | Uses only sample predictions, never actuals |
| 8 | TP/SL simulation | ✅ | Step-by-step iteration, each step only checks its own high/low |

## Strategy-Level Integrity

- **No hardcoded parameters** — all via `global.yaml` and strategy-specific configs
- **No circular imports** — data → models → signals → strategies → experiments
- **No absolute paths in .py files** — paths only in config/global.yaml
- **Main session filter applied before Kronos** — evening session never contaminates predictions

## Config Hierarchy

```
global.yaml (model, universe, session, data)
  └── calibrate/config.yaml (inference params: T, top_p, sample_count)
        └── experiments/*/config.yaml (strategy-specific overrides)
```

## Data Integrity

- Mamba dataset: 2023-01-03 through 2026-05-28 (46,078 rows)
- All 8 MOEX tickers + IMOEX index
- Main session: 10:00-18:40 MSK (56 candles/day typical)
- Samples: 3,117 windows (each with 5 samples × 6 steps × 6 features)

---

*Ported from kronos-artifact/AUDIT.md. Revision: 1.1.*
