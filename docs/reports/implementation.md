# Implementation Report — Phase 2: Inference

> **Date**: 2026-05-28
> **Scope**: Modal GPU deployment, belief extraction, multi-TF inference pipeline
> **Status**: M-PREDICT enhanced ✅, M-INFRA deployed ✅

---

## 1. What Works

### 1.1 Kronos-mini Inference on T4

| Parameter | Value |
|-----------|-------|
| Model | `NeoQuasar/Kronos-mini` + `Kronos-Tokenizer-2k` |
| Lookback | 500 bars (~10 trading days) |
| Pred_len | 12 bars (2 hours) |
| MC paths | 5 |
| Sub-batch | 8 windows per call |
| GPU | Modal T4 (16 GB) |
| Windows | 6,428 |
| Time | ~30 min |
| Output | `SBER_preds_pl12_sc5.npy` (9.3 MB), `SBER_belief_pl12_sc5.npy` (6.2 MB) |

Close predictions range: [296, 328] — reasonable for SBER (~300 RUB/share).

### 1.2 Belief Extraction

Extracted per MC path, per autoregressive step:

| Metric | Description | Range |
|--------|------------|-------|
| `confidence` | max softmax probability of s1 token | [0.035, 0.747] |
| `entropy_s1` | token distribution entropy | [0.034, 7.369] |
| `top3_mass` | cumulative prob of top-3 tokens | [0.039, 0.959] |
| `entropy_ratio` | H(s1) / H(s2) — direction vs volatility | computed |

Shape: `(6428, 5, 12, 4)` — windows × MC paths × pred_len × metrics.

### 1.3 Data Pipeline

| Data | Source | Bars | Status |
|------|--------|------|--------|
| 10-min SBER | `combined_dataset.parquet` (raw) | 6,939 test | ✅ |
| 1-hour SBER | MOEX ISS API (`--interval 60`) | 903 test | ✅ |
| 1-hour 9 tickers | MOEX ISS API | 15,800+ each | ✅ |

All 1-hour data fetched with 0 HTTP errors across 396 API requests.

### 1.4 Seed Determinism

Fixed via **per-call seed reset**:

```python
torch.manual_seed(seed + step * 1000 + 0)  # before s1 multinomial
torch.manual_seed(seed + step * 1000 + 1)  # before s2 multinomial
```

Advantages:
- Same (step, token_type) → same random numbers → same tokens regardless of batch size
- Different MC paths within same batch → different tokens (sequential RNG consumption)
- bf16/fp32 safe: `.float()` before softmax prevents rounding differences

### 1.5 Modal Deployment

| Resource | Name | Purpose |
|----------|------|---------|
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | GPU runtime |
| Volume | `kronos-hf-cache` | HF model weights |
| Volume | `kronos-predictions` | Output predictions + beliefs |
| App | `kronos-inference` | Container orchestration |

Functions: `seed()`, `infer_10min()`, `infer_10min_small()`, `infer_1hour()`, `infer_10min_a100()`

---

## 2. What Didn't Work & Fixes

### 2.1 `modal volume get` hangs indefinitely

**Symptom**: `modal volume get kronos-predictions /path/file.npy ./` hangs with no output.

**Root cause**: Modal CLI v1.4.3 may have connectivity issues with volume downloads on some networks.

**Workaround**: Use Python SDK:
```python
import modal
vol = modal.Volume.from_name("kronos-predictions")
data = b''.join(list(vol.read_file(remote_path)))
with open(local_path, "wb") as f:
    f.write(data)
```

**Alternative**: Download from Modal web dashboard or from Modal app function via `modal.Function.from_name()`.

### 2.2 Global z-score in feats*.npy causes double normalization

**Observation**: `data/feats_train.npy` etc. contain globally z-score normalized data, not raw OHLCV. The predictor's `predict_samples_batch()` normalizes again per-window, causing double normalization.

**Fix**: Use raw OHLCV from `combined_dataset.parquet` (not `feats_*.npy`). The predictor extracts per-window context and normalizes it once.

### 2.3 Config split dates mismatch

**Observation**: `config/global.yaml` dates (train_end=2025-02-01) didn't match actual data splits (train_end=2025-05-22). Off by 80-110 days.

**Fix**: Updated config to match actual splits: 2023-01-03 → 2025-05-22 (train), 2025-05-22 → 2025-11-20 (val), 2025-11-20 → 2026-05-28 (test).

### 2.4 Timezone mismatch between API and existing data

**Observation**: MOEX ISS API returns timestamps in `Europe/Moscow` (timezone-aware). Existing 10-min data is timezone-naive.

**Fix**: `preprocess_1h.py` converts to MSK and strips timezone → both TFs use naive local timestamps.

### 2.5 CLI processed one window at a time → extremely slow

**Observation**: Original CLI iterated `for t_idx in range(start_idx, n_total - pred_len + 1)` — one call per window.

**Fix**: Batch mode with `--sub-batch 8` (T4) / 16 (A100). 805 batches instead of 6428 individual calls.

---

## 3. Model Comparisons (In Progress)

| Scenario | Model | Lookback | GPU | Status |
|----------|-------|----------|-----|--------|
| mini(l500) | Kronos-mini | 500 | T4 | ✅ Done |
| small(l500) | Kronos-small | 500 | T4 | 🔄 Running |
| mini(l2036) | Kronos-mini | 2036 | A100 | ⏳ Pending |
| small(l500)_A100 | Kronos-small | 500 | A100 | ⏳ Pending |
| 1h(l510) | Kronos-mini | 510 | T4 | ⏳ Pending |

### Expected differences

- **mini(l500) vs small(l500)**: Same context length, different architectures. Mini has larger d_model (832 vs 256) but fewer params than small (24.7M).
- **mini(l500) vs mini(l2036)**: Same model, different context length. 2036 sees ~40 trading days vs ~10 days.
- **mini(l2036) vs small(l500)**: Max capacity of each model. Mini sees 4× more history.
- **10-min vs 1-hour**: Different temporal resolution. 10-min: 12 AR steps, 1-hour: 2 AR steps, same 2-hour horizon.

---

## 4. Key Learning: Context Window Equation

```
Kronos-mini:  lookback = 2048 - pred_len  (= 2036 for pred_len=12)
Kronos-small: lookback = 512 - pred_len   (= 500 for pred_len=12)
```

The model's `max_context` sets the upper bound. The predictor's sliding buffer uses this to determine how much history to keep. For mini, using only 500/2048 tokens (25%) wastes 75% of capacity. Full context (2036) enables ~40 days of market memory vs ~10 days.

---

## 5. Data Format (for future inference runs)

```python
# Required format for --feats:
# shape: (N_bars, N_tickers * 5) float32
# columns per ticker: [open, high, low, close, volume]
# raw OHLCV prices, NOT normalized
# predictor normalizes per-window internally

# Required format for --timestamps:
# shape: (N_bars,) string
# format: "YYYY-MM-DDTHH:MM:SS.000000"
# timezone-naive MSK local time
```
