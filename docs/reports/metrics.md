# Performance Metrics

## Full Year 2026 (Hold-Out Test)

| Strategy | n | Sharpe | PF | WinRate | AvgRet% | TotRet% | MDD% |
|---|---|---|---|---|---|---|---|
| Vanilla ct=0.8 | 614 | 2.02 | 1.45 | 48.9% | +0.0165 | +10.15% | -2.03% |
| Vanilla ct=1.0 | 365 | 0.84 | 1.17 | 49.9% | +0.0073 | +2.68% | -2.50% |
| S1 BB+Cons ct=0.8 | 63 | 4.14 | 2.26 | 41.3% | +0.0356 | +2.24% | -0.37% |
| S1 BB+Cons ct=1.0 | 39 | 3.39 | 1.90 | 43.6% | +0.0308 | +1.20% | -0.53% |
| S2 BB MR ct=1.0 | 220 | 1.83 | 1.39 | 53.6% | +0.0154 | +3.38% | -1.61% |
| S5 BB Narr ct=0.8 | 157 | 4.14 | 2.14 | 47.1% | +0.0317 | +4.97% | -0.80% |
| S20 OB ct=1.0 | 19 | 6.89 | 2.87 | 63.2% | +0.0450 | +0.86% | -0.15% |
| S28 VolOB ct=1.0 | 41 | 5.82 | 2.65 | 63.4% | +0.0433 | +1.77% | -0.33% |
| S34 VWAP+OB ct=1.0 | 95 | 5.37 | 2.58 | 56.8% | +0.0397 | +3.77% | -0.60% |
| S38 LoVolOB ct=1.0 | 73 | 4.90 | 2.43 | 60.3% | +0.0352 | +2.57% | -0.49% |

## Monthly Sharpe (2026)

| Strategy | Jan | Feb | Mar | Apr | May | Stable% |
|---|---|---|---|---|---|---|
| Vanilla ct=0.8 | 2.76 | 3.26 | 0.08 | 3.78 | 1.75 | 100% |
| Vanilla ct=1.0 | 1.72 | 2.70 | -1.82 | 2.05 | 2.07 | 80% |
| S1 BB+Cons ct=0.8 | 3.36 | 1.14 | 3.23 | 9.08 | 4.05 | 100% |
| S1 BB+Cons ct=1.0 | 3.57 | -10.49 | 2.13 | 9.04 | 8.54 | 80% |
| S2 BB MR ct=1.0 | 2.13 | 2.74 | 0.14 | 0.68 | 8.43 | 100% |
| S5 BB Narr ct=0.8 | 5.76 | 5.54 | 2.74 | 4.73 | 3.15 | 100% |
| S20 OB ct=1.0 | -0.45 | 20.42 | 15.21 | 4.43 | 86.45 | 80% |
| S28 VolOB ct=1.0 | 4.28 | 0.83 | 8.25 | 17.99 | -4.39 | 80% |
| S34 VWAP+OB ct=1.0 | 6.02 | 3.34 | 7.37 | 12.49 | -1.73 | 80% |
| S38 LoVolOB ct=1.0 | 3.45 | 5.04 | 3.56 | 16.57 | 1.29 | 100% |

## May 25-27 2026 Detail

| Strategy | n | Sharpe | PF | WinRate | AvgRet% | TotRet% | TP | SL | Close |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla ct=0.8 | 33 | -1.14 | 0.82 | 48.5% | -0.0092 | -0.30% | 13 | 13 | 7 |
| Vanilla ct=1.0 | 15 | -1.17 | 0.80 | 40.0% | -0.0113 | -0.17% | 5 | 5 | 5 |
| S1 BB+Cons ct=0.8 | 5 | 20.70 | 61.89 | 80.0% | +0.1084 | +0.54% | 3 | 1 | 1 |
| S1 BB+Cons ct=1.0 | 4 | 20.08 | 53.87 | 75.0% | +0.1177 | +0.47% | 2 | 1 | 1 |
| S2 BB MR ct=1.0 | 7 | 16.50 | 35.91 | 71.4% | +0.0848 | +0.59% | 3 | 2 | 2 |
| S5 BB Narr ct=0.8 | 5 | 8.35 | 3.46 | 60.0% | +0.0427 | +0.21% | 2 | 2 | 1 |
| S20 OB ct=1.0 | 1 | — | ∞ | 100% | +0.0780 | +0.08% | 1 | 0 | 0 |
| S28 VolOB ct=1.0 | 0 | — | — | — | — | — | — | — | — |
| S34 VWAP+OB ct=1.0 | 4 | -6.79 | 0.17 | 25.0% | -0.0980 | -0.39% | 2 | 1 | 1 |
| S38 LoVolOB ct=1.0 | 3 | -7.05 | 0.18 | 33.3% | -0.1219 | -0.37% | 1 | 1 | 1 |

## Retro-Validation (May 2026 as filter)

| Verdict | Strategies |
|---|---|
| ✅ WORKING (9) | Vanilla 0.8/1.0, S1 0.8/1.0, S2 1.0, S5 0.8, S20 1.0, S38 1.0, S1 sfn1 |
| ❌ FAILED (3) | S28 VolOB, S34 VWAP+OB (both variants) |

## Key Findings

1. **S2 BB MR ct=1.0** and **S5 BB Narr ct=0.8** are the most stable — positive Sharpe every month of 2026
2. **S1 BB+Cons ct=0.8** has best Sharpe × trade count balance (Sh 4.14, 63 trades in 2026)
3. **Vanilla ct=0.8** has most trades (614) but lowest Sharpe among working strategies
4. **S28/S34 fail retro check** — strong on old data, negative in recent 30 days
5. **sample_count=5 is the bottleneck** — 5 samples provide ~20% quantile resolution, limiting all sophisticated extraction
6. **Kronos q90/q10 > ATR for TP/SL** in 5/6 strategies
7. **ct=1.0 (unanimous) beats ct=0.8 (4/5)** in 6/7 strategies
8. **save_first_n=1** improves BB strategies by deferring TP/SL

---

*Ported from kronos-artifact/METRICS.md. Revision: 1.1.*
