# Kronos Skeleton & Knowledge Base

> Generated: 2026-05-28
> Purpose: Reference for inference and fine-tune tasks

---

## 1. MODEL ARCHITECTURE

### Kronos class (`src/core/kronos/model.py:21-203`)

**Signature**: `Kronos(nn.Module, PyTorchModelHubMixin)` — HuggingFace Hub download.

**Constructor** (lines 39-52):
- `s1_bits`, `s2_bits` — codebook bits per hierarchy level
- `n_layers`, `d_model`, `n_heads`, `ff_dim` — transformer spec
- `ffn_dropout_p`, `attn_dropout_p`, `resid_dropout_p`, `token_dropout_p`
- `learn_te` — learnable temporal embeddings

**Components** (lines 66-86):
```
HierarchicalEmbedding(s1_bits, s2_bits, d_model)
  ├── emb_s1: Embedding(2^s1 → d_model)
  └── emb_s2: Embedding(2^s2 → d_model)
  └── fusion_proj: Linear(2*d_model → d_model)

TemporalEmbedding(d_model, learn_te)
  └── 5 embeddings: minute(60), hour(24), weekday(7), day(32), month(13) → summed

n_layers × TransformerBlock:
  └── RMSNorm → MultiHeadAttentionWithRoPE → residual
  └── RMSNorm → FeedForward(SwiGLU) → residual

DependencyAwareLayer(d_model):
  └── MultiHeadCrossAttentionWithRoPE (query=s1_emb, key/value=hidden_states)
  └── residual + RMSNorm

DualHead(s1_bits, s2_bits, d_model):
  ├── proj_s1: Linear(d_model → 2^s1)  # unconditional s1
  └── proj_s2: Linear(d_model → 2^s2)  # conditional s2 (s2 conditioned on s1)
```

**Forward** (lines 102-151):
```
Input: s1_ids [B,T], s2_ids [B,T], stamp [B,T,5], padding_mask, teacher_forcing

1. x = HierarchicalEmbedding([s1_ids, s2_ids])         → [B,T,d_model]
2. x += TemporalEmbedding(stamp)                        → [B,T,d_model]
3. x = token_dropout(x)
4. For each block: x = block(x, causal_mask)           → [B,T,d_model]
5. x = RMSNorm(x)
6. s1_logits = head.proj_s1(x)                          → [B,T,2^s1]

7. If teacher_forcing: sibling = emb_s1(s1_targets)
   Else: sample s1 from s1_logits → sibling = emb_s1(sample)

8. x2 = DependencyAwareLayer(hidden=x, sibling_embed=sibling) → [B,T,d_model]
9. s2_logits = head.proj_s2(x2)                                 → [B,T,2^s2]

Return: (s1_logits, s2_logits)
```

**Key insight**: s2 tokens conditioned on s1 via cross-attention (`DependencyAwareLayer`). Coarse-to-fine hierarchy.

### Model Variants (official from HuggingFace model zoo)

| Model | Tokenizer | Context length | Params | Vocab |
|-------|-----------|---------------|--------|-------|
| **Kronos-mini** | `NeoQuasar/Kronos-Tokenizer-2k` | 2048 | 4.1M | ~2048 |
| **Kronos-small** | `NeoQuasar/Kronos-Tokenizer-base` | 512 | 24.7M | ~base |
| Kronos-base | `NeoQuasar/Kronos-Tokenizer-base` | 512 | 102.3M | ~base |
| Kronos-large | `NeoQuasar/Kronos-Tokenizer-base` | 512 | 499.2M | ~base |

**⚠️ Критично**: Tokenizer для mini и small — **РАЗНЫЕ**:
- **mini** → `Kronos-Tokenizer-2k` (2^11 ≈ 2048 vocab, 2048 context)
- **small** → `Kronos-Tokenizer-base` (512 context, другой codebook)
- `config/global.yaml` line 74: `source: "NeoQuasar/Kronos-Tokenizer-2k"` — верно только для mini
- Для fine-tune small потребуется `NeoQuasar/Kronos-Tokenizer-base`

### CE Loss (`modules.py:544-557`, `DualHead.compute_loss`)
```python
compute_loss(s1_logits, s2_logits, s1_targets, s2_targets, padding_mask):
    ce_s1 = CE(s1_logits[mask], s1_targets[mask])
    ce_s2 = CE(s2_logits[mask], s2_targets[mask])
    return (ce_s1 + ce_s2) / 2
```

---

## 2. TOKENIZER — VQ-VAE with BSQ (`src/core/kronos/tokenizer.py`)

**Class**: `KronosTokenizer(nn.Module, PyTorchModelHubMixin)` (line 16)

**Architecture** (lines 121-162):
```
Input: x [B, T, 6] (OHLCV + amount)

1. embed: Linear(6 → d_model)
2. Encoder: (n_enc_layers-1) × TransformerBlock
3. quant_embed: Linear(d_model → codebook_dim=s1_bits+s2_bits) → [B,T,20]
4. F.normalize(z, dim=-1)
5. BSQuantizer: sign(z) → {-1,+1}^20 → z_indices (split into s1_ids + s2_ids if half=True)
6. post_quant_embed: Linear(20 → d_model)
7. Decoder: (n_dec_layers-1) × TransformerBlock
8. head: Linear(d_model → 6) → x̂

Returns: ((z_pre, z), bsq_loss, quantized, z_indices)
```

### s1_ids and s2_ids
- Total: `s1_bits + s2_bits = 10 + 10 = 20` bits (2^20 = 1,048,576 codebook)
- With `half=True`: split into s1 (first 10 bits, vocab 1024), s2 (last 10 bits, vocab 1024)
- `encode(x, half=True)` → `(s1_ids, s2_ids)` each `[B, T]` int64
- `decode(indices, half=True)` → reconstructed OHLCV `[B, T, 6]`

### BSQ Loss
```
bsq_loss = beta * MSE(zq.detach(), z) + zeta * entropy_penalty / inv_temperature
entropy_penalty = gamma0 * H_sample - gamma * H_codebook
```

### Tokenizer config (`config/global.yaml:73-88`)
```yaml
tokenizer:
  source: "NeoQuasar/Kronos-Tokenizer-2k"   # ТОЛЬКО для mini! Для small: Kronos-Tokenizer-base
  codebook_vocab: 1024
  input_dim: 6
  codebook_heads: 8
  s1_bits: 10
  s2_bits: 10
  d_model: 256
  encoder_layers: 4
  frozen: true
```

---

## 3. PREDICTOR / INFERENCE (`src/core/kronos/predictor.py`)

### Sampling (`sample_from_logits`, line 57)
```python
sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits /= temperature
    # top_k_top_p filtering (lines 13-54)
    probs = softmax(logits)
    return multinomial(probs) if sample_logits else argmax(probs)
```

### `auto_regressive_inference` (line 73-188)
- **Input**: `x [B,T,6], x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5`
- **Output**: `np.ndarray [B, pred_len, 6]` — **averaged** over MC paths

**Algorithm**:
1. Clip input to [-5, 5]; replicate `B → B × sample_count` for MC
2. `tokenizer.encode(x, half=True)` → `(s1_ids, s2_ids)` each `[B*sample_count, T_context]`
3. Ring buffer `max_context` initialized with last tokens
4. For each pred_len step:
   - `s1_logits, context = model.decode_s1(s1_buffer, s2_buffer, stamp)`
   - `s1_next = sample_from_logits(s1_logits[:, -1, :], T, top_k, top_p)`
   - `s2_logits = model.decode_s2(context, s1_next)`
   - `s2_next = sample_from_logits(s2_logits[:, -1, :], T, top_k, top_p)`
   - Append to ring buffer
5. `tokenizer.decode([full_pre, full_post], half=True)` → OHLCV
6. Average over `sample_count` paths → `[B, pred_len, 6]`

### `KronosPredictor` (line 311-553)
- `generate()` — calls `auto_regressive_inference`, returns last `pred_len`
- `predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count)` — single series:
  1. Validate OHLCV columns, fill volume/amount
  2. `calc_time_stamps()` — minute, hour, weekday, day, month
  3. Normalize: `(x - mean) / (std + 1e-5)`, clip to [-clip, clip]
  4. `generate()` → denormalize
  5. Return `pd.DataFrame` with OHLCV
- `predict_batch()` — batched (same lengths required)

### `KronosModel` (line 563-873) — High-level wrapper
- Default: `NeoQuasar/Kronos-mini` + `NeoQuasar/Kronos-Tokenizer-2k`
- `load()` → `from_pretrained()` via HuggingFace Hub
- `_filter_session()` — 10:00-18:40 MSK
- `predict_samples()` — returns `[sample_count, pred_len, 6]` (raw paths)
- `predict_batch()` — averaged, batched

### Inference config (`config/global.yaml:149-161`)
```yaml
inference:
  lookback: 506
  pred_len: 6
  temperature: 0.6
  top_p: 0.9
  top_k: 50
  sample_count: 5
  batch_size: 16
  sub_batch_size: 4
  session_filter: "10:00-18:45"
  pred_lens: [1, 3, 6, 12, 24]
  gpu: T4
```

---

## 4. FINE-TUNE (❌ future — `src/core/kronos/fine_tune.py` не существует)

### Что специфицировано:
- **Purpose**: Fine-tune Kronos-small on MOEX tokens (tokenizer frozen), CE loss
- **Input**: DataLoader from M-DATASET + `NeoQuasar/Kronos-small` from HuggingFace
- **Tokenizer**: `NeoQuasar/Kronos-Tokenizer-base` (не Tokenizer-2k!)
- **Output**: `checkpoints/kronos_moex_best.pt` + `history.json`
- **Execution**: `modal run src/core/kronos/fine_tune.py` (A100 40GB, timeout 8h)
- **VRAM**: ~10-12 GB (batch=12, bf16, grad_accum=2 → effective batch=24)

### Training config (`config/global.yaml:132-147`)
```yaml
training:
  batch_size: 12
  grad_accum: 2
  lr: 1e-5
  weight_decay: 0.1
  grad_clip: 3.0
  scheduler: cosine
  epochs: 30
  precision: bf16
  gpu: A100
  timeout: 28800
  checkpoint:
    every_epochs: 3
    heartbeat_steps: 100
    name: "kronos_moex"
```

### Teacher forcing scheme (from `architecture.md`):
```
Tokenize full window: L = lookback + pred_len bars
token_in  = [s1[:, :-1], s2[:, :-1]]   # 0..L-2
token_out = [s1[:, 1:],  s2[:, 1:]]    # 1..L-1
loss = DualHead.compute_loss(s1_logits, s2_logits, s1_targets, s2_targets, padding_mask)
```

### Checkpoint strategy:
- Every 3 epochs: `kronos_moex_epoch_N.pt`
- Best val CE: `kronos_moex_best.pt`
- Heartbeat (100 steps): `kronos_moex_latest.pt`
- History: `history.json`

### Verification gates (P1):

| ID | Gate | Threshold |
|----|------|-----------|
| V-M-FINETUNE-01 | Val CE convergence | Monotonically decreasing over 30 epochs |
| V-M-FINETUNE-02 | Train/Val CE gap | < 0.5 |
| V-M-FINETUNE-03 | vs random baseline | Val CE < 0.9 × random CE (~10.0) |
| V-M-FINETUNE-04 | No NaN | All params finite |
| V-M-FINETUNE-05 | Gradient norm | 0.01 < norm < 50.0 |
| V-M-FINETUNE-06 | Resume integrity | Same loss trajectory |
| V-M-FINETUNE-07 | Per-asset CE | Std dev < 2.0 |

---

## 5. CONFIGURATION (`config/global.yaml`, 271 lines)

| Section | Key Parameters |
|---------|---------------|
| **Universe** (10-57) | 20 stocks + IMOEX, lot sizes, board=TQBR |
| **Model** (60-70) | name=NeoQuasar/Kronos-mini, alt=Kronos-small, d_model=832, n_layers=12, n_heads=16, 24.7M params |
| **Tokenizer** (73-88) | s1_bits=10, s2_bits=10, d_model=256, encoder_layers=4, frozen=true |
| **Data** (91-122) | interval=10min, lookback=506, pred_len=6, session=10:00-18:40, splits: train 2023→2025-02, val 2025-02→2025-09, test 2025-09→2026-05, per-window z-score, clip=5.0 |
| **Dataset** (125-129) | L=512, stride=8, shift=1 |
| **Training** (132-147) | batch=12, lr=1e-5, bf16, cosine, 30 epochs, A100 |
| **Inference** (149-161) | lookback=506, pred_len=6, T=0.6, top_p=0.9, top_k=50, sample_count=5, T4 |
| **Strategies** (163-198) | consensus threshold=0.6, TP/SL quantile, cross-sectional top-3/bot-2, BB, OB, FVG, VWAP |
| **Backtest** (200-227) | cross_sectional_top3_bot2, walk-forward step=6, metrics gates |
| **Calibration** (231-242) | pred_len sweep [1,3,6,12,24], T sweep [0.2-1.0], top_p sweep [0.7-0.95] |
| **Execution** (245-261) | timezone=Europe/Moscow, GPU mapping |
| **Monitoring** (264-270) | log_level=INFO, wandb disabled |

---

## 6. DATA PIPELINE

```
M-FETCH  (✅) → M-PREPROCESS (❌) → M-TOKENIZE (✅) → M-DATASET (❌)
   ↓                                                      ↓
parquet + manifest                                   DataLoader
   ↓                                                      ↓
 src/data/fetcher.py                               src/data/dataset.py

M-DATASET → M-FINE-TUNE (❌) → M-PREDICT (✅) → M-BACKTEST (❌)
               ↓                      ↓
         kronos_moex_best.pt    predictions .npy
               ↓                      ↓
         src/core/kronos/       src/evaluation/
         fine_tune.py           backtest.py
```

### Data files:
- `data/v3/raw/{TICKER}.parquet` — from M-FETCH
- `data/v3/processed/{ticker}_{split}.npy` — planned from M-PREPROCESS
- `data/v3/tokens/{ticker}_{split}.npy` — planned from M-TOKENIZE
- `data/v3/predictions/{ticker}_{split}.npy` — from M-PREDICT

### Time stamps (`calc_time_stamps`, predictor.py:301-308):
```python
# 5 features: minute(60), hour(24), weekday(7), day(32), month(13)
# Maps to TemporalEmbedding with 5 summed embeddings
```

---

## 7. EXISTING CHECKPOINTS / MODELS

**No local checkpoint files** (no `.pt`, `.pth`, `.bin`, `.safetensors` on disk).

**Model loading**: via `PyTorchModelHubMixin.from_pretrained()` from HuggingFace Hub.

| Model | HF ID | Requires |
|-------|-------|----------|
| Tokenizer (mini) | `NeoQuasar/Kronos-Tokenizer-2k` | `HF_TOKEN` (gated) |
| Tokenizer (small) | `NeoQuasar/Kronos-Tokenizer-base` | `HF_TOKEN` (gated) |
| Kronos-mini | `NeoQuasar/Kronos-mini` | `HF_TOKEN` (gated) |
| Kronos-small | `NeoQuasar/Kronos-small` | `HF_TOKEN` (gated) |

### Model registry (`src/core/registry.py:81-92`)
```python
register_model("kronos_mini", KronosModel)
register_model("kronos_mini_2048", lambda **kw: KronosModel(
    model_name="NeoQuasar/Kronos-mini",
    tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k",
    max_context=2048, **kw))
```

---

## 8. SIGNALS (8 families, `src/signals/__init__.py`)

| # | Family | Key Functions |
|---|--------|--------------|
| 1 | **ATOMS** | direction, consensus(threshold=0.8), boundaries(tp_q=0.80, sl_q=0.20), dispersion, trend_strength, linearity, asymmetry, expectancy |
| 2 | **BOLLINGER** | compute_bb(period=20, std=2), bb_position, bb_squeeze, bb_signal |
| 3 | **ICT** | detect_swings, detect_order_block, detect_fvg, detect_liquidity_sweep, detect_premium_discount, detect_mss, detect_breaker_block, detect_eqh_eql |
| 4 | **VOLATILITY** | compute_atr, compute_adr, volatility_regime |
| 5 | **VWAP** | compute_vwap, anchored_vwap, vwap_cross |
| 6 | **FRACTAL** | find_fractals, fractal_signal, breakout_signal, cluster_signal, compute_ao |
| 7 | **DIVERGENCE** | rsi, obv_data, mfi, detect_divergence |
| 8 | **BARS** | classify_bar (15 types: doji, marubozu, hammer, shooting star, engulfing, harami, inside/outside) |

---

## 9. STRATEGIES (`src/strategies/`)

| Strategy | File | Logic |
|----------|------|-------|
| **VANILLA** | vanilla.py (58L) | Pure Kronos consensus, q90/q10 TP/SL |
| **S01_BB** | s01_bb.py (69L) | BB extreme + consensus direction match |
| **S02_BB_MR** | s02_bb_mr.py (56L) | BB mean reversion |
| **S05_BB_BREAKOUT** | s05_bb_breakout.py (72L) | Narrow BB + MR |
| **S20_OB** | s20_ob.py (48L) | Order Block + consensus |
| **S28_VOL_OB** | s28_vol_ob.py (85L) | Volume-filtered OB |
| **S34_VWAP_OB** | s34_vwap_ob.py (88L) | VWAP-confirmed OB |
| **S38_LOWVOL_OB** | s38_lowvol_ob.py (94L) | Low-vol OB |

### Core engine (`src/strategies/core.py:109-204`)
```python
_simulate_trade():
  1. Check consensus (≥threshold at step 0)
  2. Optional dispersion cap filter
  3. Compute TP/SL from sample quantiles (or ATR-based)
  4. Walk candle by candle: exit on TP, SL, or close at pred_len
  5. Optional exit on consensus flip
```

---

## 10. EVALUATION (`src/evaluation/`)

### metrics.py (✅ ready)
- `sharpe_ratio(rf=0, periods=13104)` — annualized (52×252)
- `max_drawdown`, `profit_factor`, `win_rate`, `calmar_ratio`
- `direction_accuracy`, `direction_sharpe`, `return_correlation`
- `ic_rank` — Spearman
- `bias`, `mae`, `prediction_volatility`
- `psr` (Probabilistic Sharpe Ratio), `dsharpe_ratio` (Deflated)
- `sortino_ratio`, `avg_return`, `n_trades`, `trade_pct`
- `StrategyMetrics` class — named container
- `evaluate_model()` — 7+ metrics in one call

### calibrate.py (✅ ready)
- Pass 1: pred_len sweep [1,3,6,12,24]
- Pass 2: T × top_p × sample_count sweep
- Output: `results.json`

### walk_forward.py (✅ ready)
- Per-ticker OHLCV, main session filter
- Non-overlapping windows (step=pred_len)
- Grouped by calendar month
- Batch inference (sub-batched for GPU)
- Per-window samples + monthly aggregate metrics

### output.py (✅ ready)
- `save_samples()` — compact parquet with `samples_blob` + `actual_blob`
- `reconstruct(blob, *shape)` — deserialize
- `compute_monthly_metrics()`, `trade_summary()`, `save_summary()`

---

## 11. MODULE STATUS SUMMARY

| ID | File | Status |
|----|------|--------|
| M-FETCH | `src/data/fetcher.py` | ✅ ready |
| M-PREPROCESS | `src/data/preprocess.py` | ❌ future |
| M-TOKENIZE | `src/core/kronos/tokenizer.py` | ✅ ready |
| M-DATASET | `src/data/dataset.py` | ❌ future |
| M-FINE-TUNE | `src/core/kronos/fine_tune.py` | ❌ future |
| M-PREDICT | `src/core/kronos/predictor.py` | ✅ ready |
| M-BACKTEST | `src/evaluation/backtest.py` | ❌ future |
| M-METRICS | `src/evaluation/metrics.py` | ✅ ready |
| M-CALIBRATE | `src/evaluation/calibrate.py` | ✅ ready |
| M-CONFIG | `config/global.yaml` | ✅ ready |

---

## 12. KEY CORRECTIONS TO AGENTS.md & CONFIG

1. **Tokenizer mismatch**: `config/global.yaml:74` specifies `Kronos-Tokenizer-2k`, но:
   - Для **Kronos-mini** (инференс сейчас) → ✅ корректно
   - Для **Kronos-small** (fine-tune) → нужен `Kronos-Tokenizer-base`
   - У них разный vocab size, context length, и параметры codebook

2. **max_context разный**: mini=2048, small=512 — критично для fine-tune (lookback должен влезать)

3. **Registry**: `src/core/registry.py` регистрирует только mini — при добавлении small потребуется расширение

4. **HF auth**: все модели gated — необходим `HF_TOKEN`
