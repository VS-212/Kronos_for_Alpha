# PR_Mamba → Kronos Pivot: Cognitive Engineering Artifact

> **Target**: Cross-sectional alpha model on MOEX top-21 liquid assets.  
> **Purpose**: Single source of truth for the pivot from FinMamba v2 to Kronos fine-tune.  
> **Readers**: Developer implementing the plan.  
> **Notation**: `(B,T,6)→(B,T)→2×int64` = shape flow. `[§1.2]` = cross-reference to Layer.  

---

## L0: SITUATION & DIAGNOSIS

> **Greenfield note**: This plan maps to ZERO existing code in this repository. All current code (FinMamba, MSE loss, engineered features) is superseded. The Kronos fine-tune path is a new codebase within `src/`. Existing `backtest_cs.py`(portfolio PnL), `metrics.py`, `moex_fetcher.py` are reusable. All model code is new.

### What We Tried

| Version | Approach | Best Result | Root Failure |
|---------|----------|-------------|--------------|
| **v1** | Mamba-2, 4 OHLCV features, no TF, `h[:,-1,:]` | Sharpe −163 | Untrained position, too few features |
| **v2** | Mamba-2, 31 engineered features, teacher forcing, MSE on log return | **Sharpe −14, WR 43%, MaxDD −53%** (after all bug fixes) | **MSE on noisy continuous target → mean regression collapse** |

### Root Cause Hierarchy

```
┌─── ROOT CAUSE ───────────────────────────────────────────────────────┐
│ MSE on noisy log return (μ≈0, σ≈0.006) → optimal prediction = E[y]≈0 │
│ The model learns to predict the MEAN, which is zero.                  │
│ CE on discrete tokens would learn the DISTRIBUTION.                  │
└──────────────────────────────────────────────────────────────────────┘
         │
         ├── ENABLING BUGS (inflated/deflated backtest metrics) ──────
         │   [a] Position 255 masked (offset=6): backtest used untrained output
         │   [b] ret_1/5/20 were FORWARD returns (ln(C[t+N]/C[t])) — lookahead
         │   [c] ADR used future bars (max(h[i:i+50]) at bar 0) — lookahead
         │   [d] PnL compounding (equity × (1+pnl) per position) — 5× multiplication
         │   [e] z-score μ,σ computed on entire dataset — test stats leak
         │   ↳ All [a-e] FIXED. Remaining negative Sharpe is REAL.
         │
         └── DESIGN FLAWS (why architecture was wrong even after fixes) ──────
              [f] 31 engineered features ≈ noisy transforms of raw OHLCV, no new info
              [g] Single scalar output → model has no way to express uncertainty
              [h] MSE = convex loss on non-convex problem (market can go up OR down)
```

### Why Kronos Is The Answer

Kronos solves [f] [g] [h] simultaneously:
- **[f]**: Raw OHLCV → VQ-VAE tokenizer learns latent representation automatically
- **[g]**: Discrete 1024-class token prediction → distributional output (30% up, 25% down, 45% flat)
- **[h]**: Cross-Entropy = proper scoring rule for categorical distribution → no mean collapse

---

## L1: KRONOS ARCHITECTURE & WHY IT WINS

### 1.1 Tokenizer (VQ-VAE Autoencoder)

```
Raw OHLCV+Amount (B,T,6)
       │
       ├── KronosTokenizer.embed (Linear 6→256)
       ├── Encoder: 4×Transformer (d=256, head=4, ff=512)
       │      → z.shape = (B,T,256)
       ├── quant_embed (Linear 256→20)     # codebook_dim = s1_bits + s2_bits = 20
       ├── F.normalize(z, dim=-1)          # L2 normalize
       ├── BSQuantizer: z → sign(z) in {−1,+1}^20  (straight-through estimator)
       │      → quantized: (B,T,20) binary ±1
       │      → z_indices: (B,T) int64, vocab 0..2^20-1  (or 2×(B,T) for half=True)
       ├── post_quant_embed (Linear 20→256)
       ├── Decoder: 4×Transformer (d=256)
       └── head (Linear 256→6)
              → x̂.shape = (B,T,6)  reconstructed OHLCV+Amount
```

**Training loss**: `L_recon(MSE) + β·L_commit + ζ·(γ₀·H_sample − γ·H_codebook)`

| Param | Value | Meaning |
|-------|-------|---------|
| β | 0.05 | Commit loss weight — embeddings stay near codes |
| γ₀ | 1.0 | Per-sample entropy weight — prevents few codes dominating |
| γ | 1.1 | Codebook entropy weight — encourages diverse code usage |
| ζ | 0.05 | Overall entropy weight |
| group_size | 5 | BSQ group approximation (entropy computation) |

**BSQ core quantize** (module.py:82-88):
```python
zhat = where(z > 0, +1, -1)           # hard binarize
return z + (zhat - z).detach()         # straight-through gradient
```

### 1.2 Predictor (Autoregressive Transformer)

```
s1_ids (B,L) ∈ [0,1023]        s2_ids (B,L) ∈ [0,1023]
        │                               │
        ├── emb_s1 (1024×256)            ├── emb_s2 (1024×256)
        │   Embedding lookup             │   Embedding lookup
        └───────────┬───────────────────┘
                    │ concat → Linear(512→256) → fusion
                    │  +
                    │ TemporalEmbedding (minute, hour, weekday, day, month)
                    │  — 5 separate embeddings, summed
                    ↓
          N×TransformerBlock (causal, RoPE, RMSNorm, SwiGLU)
          │   d_model=256/832, heads=4/16, layers=4/12
          ↓
          DualHead:
            ├── s1_logits: Linear(256→1024)            → CE_s1
            └── s2_logits: Linear(256→1024)            → CE_s2
                 (conditioned on s1 via DependencyAwareLayer:
                  cross-attn where query=s1_emb, key/value=hidden_states)
          ↓
          Loss = (CE_s1 + CE_s2) / 2
```

**Teacher forcing** (ALL positions, no masking):
```python
# Tokenize entire window: L = lookback + pred_len bars
token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)  # 2×(B,L) int64

# Shift right by 1
token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]   # 0..L-2
token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]     # 1..L-1

# Forward pass — causal attention, all positions supervised
logits = model(token_in[0], token_in[1], stamp[:, :-1, :])
loss = (CE(s1_logits, s1_targets) + CE(s2_logits, s2_targets)) / 2
```

### 1.3 Inference (Autoregressive Generation)

```python
# Phase 1: Encode context
context_bars = data[i-lookback : i]   # (lookback, 6)
s1_ctx, s2_ctx = tokenizer.encode(context_bars, half=True)  # each (1, lookback)

# Phase 2: Generate pred_len bars
for step in range(pred_len):
    # Decode s1 token
    s1_logits, hidden = model.decode_s1(s1_buffer, s2_buffer, stamp)
    s1_next = sample(s1_logits[:, -1, :], T=0.6, top_p=0.9)  # (1,)

    # Decode s2 token (conditioned on s1)
    s2_logits = model.decode_s2(hidden, s1_next)
    s2_next = sample(s2_logits[:, -1, :], T=0.6, top_p=0.9)

    # Append to context buffer
    s1_buffer = cat([s1_buffer, s1_next], dim=1)
    s2_buffer = cat([s2_buffer, s2_next], dim=1)

    # Slide context if exceeding max_context
    if s1_buffer.shape[1] > max_context:
        s1_buffer = s1_buffer[:, -max_context:]
        s2_buffer = s2_buffer[:, -max_context:]

# Phase 3: Decode tokens → OHLCV
all_tokens = [s1_buffer, s2_buffer]   # each (1, lookback+pred_len)
pred_bars = tokenizer.decode(all_tokens, half=True)  # (1, lookback+pred_len, 6)

# Phase 4: MC averaging
pred_bars = pred_bars.reshape(-1, sample_count, pred_len, 6)
pred_bars = pred_bars.mean(axis=1)  # average over MC samples
```

**Sampling**: `sample_from_logits` with T=0.6 (softening), top_p=0.9 (nucleus), top_k=0 (disabled). Non-deterministic → each MC path is a possible market scenario.

### 1.4 Comparison: Why Kronos Succeeds Where FinMamba Failed

| Dimension | FinMamba v2 | Kronos | Impact |
|-----------|------------|--------|--------|
| **Input** | 31 engineered features (RSI, MFI, BB, etc.) | Raw OHLCV+Amount (6 dims) → VQ tokens | Kronos learns representation; FinMamba given noisy transforms |
| **Target** | Single scalar (log return, continuous) | 2 discrete tokens per bar (s1∈1024, s2∈1024) | Kronos predicts bar SHAPE, not just direction |
| **Loss** | MSE = E[y] → 0 collapse | CE(s1)+CE(s2) = proper scoring for distribution | CE preserves multi-modality (up ⊕ down ⊕ flat) |
| **Output** | Deterministic point estimate | Probabilistic: 1024-way classification per token | Can express "30% chance of up, 25% down" |
| **Teacher Forcing** | 256 positions, 6 masked (offset=6) | All L positions, shift by 1, no masking | Full utilization of training signal |
| **Inference** | Single forward pass, no sampling | Autoregressive + temperature sampling + MC paths | Robustness via ensemble of scenarios |
| **Uncertainty** | None (single number) | Distribution over OHLCV bars | Can detect regime change (high entropy in predictions) |
| **Params** | 3.6M (8 Mamba2 layers) | 4.1M (mini) / 24.7M (small) | Comparable scale |

**Core insight**: Financial returns are fundamentally multi-modal (market can go up OR down from the same pattern). MSE averages these modes into zero. CE preserves the distribution, allowing the model to be useful even when uncertain — it tells you HOW uncertain it is.

### 1.5 Input Data: OHLCV → 6 Dimensions

> **Resolution note**: Kronos tokenizer was trained on ALI09988 **5-min** bars. MOEX minimum is **10-min** bars. This means each bar encodes 2× the duration — potentially LESS noise (stronger trends, fewer false reversals) but coarser microstructure. Empirically this should HELP prediction quality (higher SNR per bar). If performance is poor, test with 5-min data via MOEX `marketdata` engine (supports `interval=5`).

Kronos expects 6 columns: `open, high, low, close, volume, amount`.

| Col | MOEX Source | Computation |
|-----|------------|-------------|
| open | ISS API `candles.columns=open` | Raw |
| high | ISS API `candles.columns=high` | Raw |
| low | ISS API `candles.columns=low` | Raw |
| close | ISS API `candles.columns=close` | Raw |
| volume | ISS API `candles.columns=volume` | Raw (lots for stocks, RUB for IMOEX) |
| **amount** | Computed | **`close × volume × lot_size`** |

**IMPORTANT**: `lot_size` varies per ticker:

| Ticker | Lot size | Ticker | Lot size |
|--------|----------|--------|----------|
| SBER | 10 | CHMF | 1 |
| GAZP | 10 | NLMK | 10 |
| LKOH | 1 | MAGN | 10 |
| ALRS | 10 | AFLT | 10 |
| ROSN | 10 | FIVE | 1 |
| NVTK | 10 | MOEX | 10 |
| PLZL | 1 | TCST | 1 |
| GMKN | 1 | YNDX | 1 |
| TATN | 1 | SNGS | 100 |
| VTBR | 1000 | SNGSP | 100 |

**Normalization**: Kronos normalizes per-window (μ,σ of lookback bars), clip to [−5, 5]. This removes price-level differences between assets automatically. Multi-asset training is safe.

---

## L2: EXECUTION PLAN

### 2.1 Single Strategic Path

```
┌────────────────────────────────────────────────────────────────┐
│ FINE-TUNE KRONOS-SMALL ON MOEX 21 ASSETS                       │
│                                                                │
│ Reason: Proven VQ+CE architecture. 24.7M params (fits A100).   │
│         No new code beyond dataloader + fine-tune loop.         │
│         Fastest path to a working alpha model.                  │
│                                                                │
│ Fallback only if Phase 0 tokenizer validation fails:            │
│   → Retrain tokenizer on MOEX (see §2.2)                        │
│                                                                │
│ Research appendix (NOT active path):                            │
│   → Replace Transformer with Mamba2 (Mamba-VQ)                  │
│   → Ensemble with regime-specific heads                         │
│   → Multi-scale prediction (1h + 1d + 1w)                       │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Contingency: Tokenizer Failure

```
Phase 0 Tokenizer Validation
         │
         ├── PASS (utilisation>200/1024, recon_MSE<0.01, entropy>5)
         │       → Proceed to Phase 1 (fine-tune predictor)
         │
         └── FAIL
                 │
                 ├── Option A: Retrain tokenizer on MOEX (8+8 bit, ~10h A100)
                 │       └── Success → fine-tune predictor
                 │
                 └── Option B: Abandon VQ → use raw OHLCV (normalized) + Kronos predictor
                         (tokenizer.encode replaced by linear projection to codebook_dim)
```

### 2.3 Data Pipeline

```
MOEX ISS API (2023-01-01 → 2026-05-01)
        │
        ├── 21 tickers (stocks) + IMOEX (index)
        │   Selection: top-21 by MOEX liquidity, all sectors
        │   Full list: see §1.5 lot_size table
        │
        ├── Fetch: 10-min candles, columns=begin,open,high,low,close,volume
        │   IMOEX endpoint: /iss/engines/stock/markets/index/securities/IMOEX/candles
        │   IMOEX volume = total constituent volume in RUB (API returns it)
        │
        ├── Filter: main session 10:00-18:40 MSK (existing preprocess.py)
        │   └── CRITICAL: no overnight bars, no pre-market, no post-market
        │
        ├── Compute: amount = close × volume × lot_size (per ticker)
        │   └── IMOEX: amount = close × volume  (volume already in RUB)
        │
        ├── Normalize: per-window z-score (μ,σ of lookback=506 bars), clip [-5,5]
        │   └── This REMOVES price level → multi-asset training safe
        │
        ├── Dividend adjustment: subtract dividend from open/close on ex-div date
        │   └── MOEX ISS endpoint: /securities/{ticker}/dividends.json
        │   └── Prevents fake "alpha" from mechanical price drops (−2% to −10%)
        │   └── LOW priority: affects ~1-2 bars/year/ticker, skip for Phase 0-1
        │
        ├── Split (walk-forward):
        │   ├── Train: 2023-01 → 2025-02  (~26 months, ~57k bars/asset)
        │   ├── Val:   2025-02 → 2025-09  (~7 months, ~15k bars/asset)
        │   └── Test:  2025-09 → 2026-05  (~7 months, ~15k bars/asset)
        │   └── WHY walk-forward: single split overfitted in v2, OOS = more honest
        │
        └── Tokenize: KronosTokenizer.encode(ohlcv_6d, half=True) → (s1_ids, s2_ids)
            └── Save: data/v3/{ticker}/tokens_{split}.npy (2 × T int64)
```

**Data volume** (52 bars/day × 252 trading days/year = 13,104 bars/year):
```
Period: 2023-01 → 2026-05 = 3.33 years
Total:  21 assets × 43,636 bars ≈ 916,356 bars

Train: 2023-01 → 2025-02 (2.08y)  = 21 × 27,256 ≈ 572,376 bars
Val:   2025-02 → 2025-09 (0.58y)  = 21 ×  7,633 ≈ 160,293 bars
Test:  2025-09 → 2026-05 (0.67y)  = 21 ×  8,747 ≈ 183,687 bars
```

**Data quality assertions** (run before tokenization):
- No NaN in any OHLCV column
- Min 500 bars per asset (not dead/delisted)
- Timestamps monotonically increasing
- No gaps > 1 bar between consecutive bars (check for missing API data)
- Session boundaries: all bars between 10:00-18:40 MSK
- Volume > 0 for all bars (zero-volume bars indicate holiday or API error)

### 2.4 Multi-Asset Strategy

**Design decision**: Train ONE model on ALL 21 assets without explicit asset identification.

**Rationale**:
- Kronos was trained on single asset (ALI09988) — has no asset-ID embedding built in
- Per-window z-score normalization removes price-level differences
- Candlestick patterns are universal (hammer is a hammer whether SBER or YNDX)
- 21× more data than single-asset → better generalization
- Adding asset embedding requires modifying Kronos architecture — risk of breaking pre-trained weights

**Risk**: If assets have SYSTEMATICALLY different bar shape distributions, tokenizer utilization may vary per asset. **Mitigation**: Phase 0 validation checks per-asset codebook usage.

### 2.5 Cross-Sectional Signal Pipeline

```
For each bar t in test set (6765 bars):
   ┌─────────────────────────────────────────────────────┐
   │ For each asset a in 1..21:                          │
   │   1. Context = bars[t-506 : t]  (506 bars OHLCV)    │
   │   2. Tokenize context → s1_ctx, s2_ctx (506 tokens) │
   │   3. Autoregressive predict 6 bars → s1_pred, s2_pred│
   │   4. Decode tokens → OHLCV_pred (6, 6)              │
   │   5. pred_return[a] = ln(close_pred[5] / close[t])   │
   │      (log return from now to end of pred horizon)    │
   └─────────────────────────────────────────────────────┘
    
    6. z_score[a] = (pred_return[a] - mean(pred_returns)) / std(pred_returns)
       → scores across 21 assets, mean=0, std=1

    7. Rank by z_score descending:
       long  top-3  (weight +1/3 each)
       short bot-2  (weight -1/2 each)    ← optional, may remove if short costly
       rest   flat  (weight 0)

    8. Portfolio PnL per bar:
       for each active position:
          if entry:  entry_price = open[bar]
          if continue: pnl += weight × (close[bar] - close[bar-1]) / close[prev]
          if exit (SL=-1.5% or TP=+2.5% or horizon_end=t+6):
             pnl -= cost × |weight|

    9. Session enforcement:
       ├── Enter positions ONLY at bar 10:00-18:10 (to allow full horizon)
       └── FORCE exit all positions by 18:35 (main session ends 18:40)
```

### 2.6 Phase-by-Phase Execution

#### Phase 0: Tokenizer Quality Gate (2h, CPU + T4)

| Step | Action | File | Time | Gate |
|------|--------|------|------|------|
| P0.1 | Download KronosTokenizer-2k from HuggingFace | — | 0.1h | Model loaded |
| P0.2 | Tokenize MOEX val set (231k bars) | `tokenize_v3.py` | 0.5h | No crash |
| P0.3 | **Check codebook utilization**: count unique s1 IDs across all bars | script | 0.1h | **≥200 unique codes (20%)** |
| P0.4 | **Check token entropy**: H = −Σ p(i)·log₂(p(i)) | script | 0.1h | **≥5 bits (out of 10 max)** |
| P0.5 | **Check reconstruction**: MSE(tokenizer.decode(encode(x)) - x) | script | 0.3h | **≤ 0.01** |
| P0.6 | **Per-asset utilization**: verify no asset <50 unique codes | script | 0.2h | All assets ≥50 codes |
| P0.7 | **Cross-correlation matrix**: 21-asset return correlations | script | 0.1h | RMS < 0.5 (assets are differentiable) |

**If P0.3-P0.6 FAIL**: Go to §2.2 contingency (retrain tokenizer on MOEX with 8+8 bit).

**If P0.7 FAIL**: Consider sector-neutral z-scoring or reduce to fewer, more diverse assets.

#### Phase 1: Fine-tune Kronos-small (~2-4h, A100)

| Step | Action | File | Time | Gate |
|------|--------|------|------|------|
| P1.1 | Download Kronos-small from HuggingFace | — | 0.1h | Model loaded |
| P1.2 | Create `PredictorDataset`: L=512, shift=1, stride=8, batch=12 | `predictor_dataset.py` | — | Shapes verified |
| P1.3 | Train session 1 (epoch 1-15, ~7.5h) | `fine_tune.py` | 7.5h | — |
| P1.4 | Save checkpoint `kronos_moex_epoch_15.pt` to Modal volume | — | — | Saved |
| P1.5 | Resume: load best, continue to epoch 16-30 | `fine_tune.py --resume` | 7.5h | — |
| P1.6 | Save `kronos_moex_best.pt` (val CE minimum) | — | — | Saved |

**Fine-tune config**:
```
Model:          Kronos-small.from_pretrained("NeoQuasar/Kronos-small")
Frozen:         Tokenizer (eval mode + no_grad + excluded from optimizer)
Trainable:      Predictor only (HierarchicalEmbedding, Transformer, DualHead, TemporalEmbedding)
Batch:          12 (A100 40GB limit — attention KV cache dominant)
Accumulate:     2 (effective batch=24)
Lr:             1e-5 (fine-tune, low to preserve pre-trained knowledge)
Scheduler:      CosineAnnealing (T_max = steps_per_epoch × 30)
Weight decay:   0.1
Grad clip:      3.0
Precision:      bf16 (mixed, autocast)
Device:         A100 (40GB) — T4 does NOT fit KV cache for L=512, batch=12
```

**VRAM breakdown for fine-tune**:
```
Model weights (bf16):          25M × 2     =   50 MB
Optimizer (AdamW, fp32):       25M × 12    =  300 MB
Attention scores (peak):       12 layers × batch × heads × L² × 2B
  = 12 × 12 × 13 × 512² × 2   ≈ 980 MB per layer attention matrix (!)
KV cache (during TF):          12 × 12 × 512 × 64 × 2 × 2 ≈ 19 MB
FFN activations:              ~1.5 GB
Total:                         ~10-12 GB (fits A100 comfortably)
T4 would need batch=2 → 6× slower, not recommended
```

**Overfitting prevention**:
```python
# Val CE monitoring every 250 steps
if val_ce > best_val_ce for 5 consecutive eval rounds:
    early_stop()
    
# Per-asset CE tracking
# If ONE asset dominates CE improvement → possible data quality issue
```

**Checkpoint resume protocol**:
```
Volume:  finmamba-models  (Modal persistent volume)
Path:    /checkpoints/kronos_moex/
Files:
  kronos_moex_epoch_N.pt     — every 3 epochs (optimizer state included)
  kronos_moex_best.pt        — whenever val_CE improves
  kronos_moex_latest.pt      — heartbeat, every 100 steps (survive spot preemption)
  history.json                — train/val loss per epoch

Resume:  python fine_tune.py --resume kronos_moex_best.pt
         → loads optimizer state, continues from epoch best+1
```

**Time estimates** (corrected):
```
Train windows per asset (stride=8, L=512):   (27,256 - 512) / 8 ≈ 3,343
Train windows total (21 assets):              3,343 × 21        = 70,203
Batches per epoch (eff batch=24):             70,203 / 24       ≈ 2,925
A100 throughput (512-seq Transformer):        ~40 batches/sec
Time per epoch:                               2,925 / 40 / 60  ≈ 1.2 min
30 epochs:                                    1.2 × 30         ≈ 36 min
```

Fine-tune should complete in **~1 hour** on A100. Bottleneck: I/O (token loading), not compute.

#### Phase 2: Inference + Backtest (2h, T4)

| Step | Action | File | Time | Gate |
|------|--------|------|------|------|
| P2.1 | Load `kronos_moex_best.pt` | `predict_v3.py` | 0.1h | Loaded |
| P2.2 | Autoregressive inference on test (21 × 6,765 bars) | `predict_v3.py` | 1.5h | Per-asset OHLCV saved |
| P2.3 | Cross-sectional backtest (see §2.5 pipeline) | `backtest_v3.py` | 0.5h | — |
| P2.4 | Compute metrics: Sharpe, MaxDD, WR, PSR, Calmar | `metrics.py` | 0.1h | — |
| P2.5 | Walk-forward validation: re-run on val split, compare | `backtest_v3.py --split val` | 0.5h | Val ≈ Test (no overfit) |

**Live inference feasibility** (Phase 3+):
```
Per-bar latency:  tokenize(506 bars) + autoregressive_predict(6 steps) + decode
                  ≈ 0.1s + 6 × 0.05s + 0.05s ≈ 0.45s per asset (T4)
21 assets serial: ≈ 10s per bar
21 assets batched: ≈ 1.5s per bar (batch=21 on single GPU)
MOEX bar interval: 600s (10 min)

→ Inference latency << bar interval. Live deployment feasible on T4.
   Key: run tokenization ONCE, share across assets. Predict batched.
```

**Inference config**:
```
Model:          kronos_moex_best.pt
Tokenizer:      KronosTokenizer-2k (frozen)
Lookback:       506 bars (~10 trading days)
Pred_len:       6 bars (1 hour)
Sample_count:   4 (MC paths)
Temperature:    0.6 (moderate uncertainty)
Top_p:          0.9 (standard nucleus sampling)
Top_k:          0 (disabled — top_p sufficient)
Device:         T4 (16GB) — inference batch=1, fits easily
```

**VRAM for inference**:
```
Model (bf16):         25 MB
Context buffer:       batch × L × 256 × 2 = 1 × 512 × 256 × 2 = 0.26 MB
Total:                < 0.5 GB — T4 is overkill for inference
```

#### Phase 3: Iteration (if Sharpe < 0.3)

| Symptom | Likely Cause | Fix | Priority |
|---------|-------------|-----|----------|
| Sharpe < 0 | Tokenizer mismatch | Retrain tokenizer on MOEX (8+8 bit, §2.2) | P0 |
| Sharpe 0-0.3 | Weak signal | Increase sample_count to 8, ensemble per-bar | P1 |
| High MaxDD | Stop-loss too wide | Reduce SL to 1.0%, add trailing stop | P1 |
| WR < 50% | Random predictions | Check token entropy — if low → tokenizer bad | P1 |
| Predictions too similar across assets | High correlation | Sector-neutral z-score: subtract sector mean | P1 |
| Overtraded | Pred_len too short | Increase to 12 bars (2h), reduce turnover | P2 |
| Regime shift (val Sharpe >> test Sharpe) | Overfit to bull period | Add dropout 0.2, reduce epochs to 15 | P2 |
| Low cross-sectional IC | Ranking noise | Add ranking loss (ListMLE or pairwise) to fine-tune | P2 |

### 2.7 Hardware & Timing Summary

| Phase | Hardware | Time | VRAM | Key file |
|-------|----------|------|------|----------|
| P0: Tokenizer Gate | CPU + T4 | 2h | < 2 GB | `tokenize_v3.py` |
| P1: Fine-tune | **A100** (40GB) | 2-4h | ~12 GB | `fine_tune.py` |
| P2: Inference | **T4** (16GB) | 1.5h | < 1 GB | `predict_v3.py` |
| P2: Backtest | T4 (16GB) | 0.5h | < 1 GB | `backtest_v3.py` |
| **Total (A100)** | | **3-5h** | 1 session | |
| **Total (T4)** | | **4h** | 1 session | |
| **Calendar** | | **1-2 days** | parallelizable | |

### 2.8 Metrics Gates

| Phase | Metric | Minimum | Stop If | Action |
|-------|--------|---------|---------|--------|
| P0 | Codebook utilization | ≥ 200/1024 (20%) | < 50 | Retrain tokenizer |
| P0 | Token entropy | ≥ 5 bits | < 3 | Check data quality |
| P0 | Recon MSE | ≤ 0.01 | > 0.05 | Reduce codebook bits |
| P0 | Per-asset min unique codes | ≥ 50 | < 20 | Remove outlier asset |
| P0 | Cross-sectional RMS correlation | < 0.5 | > 0.7 | Sector-neutralize |
| P1 | Train CE | — (monitor) | Train CE < Val CE - 0.5 | Overfit → add dropout |
| P1 | Val CE (relative to baseline) | < 0.9 × random_CE | > random_CE | Model not learning → check tokenizer |
| P1 | Val CE plateau | Δ ≤ 0.001 for 5 evals | — | Early stop |
| P2 | Sharpe (test, walk-forward) | **≥ 0.5** | < 0.0 | Go to §2.3 Phase 3 |
| P2 | Win Rate | ≥ 52% | < 45% | Check prediction bias |
| P2 | Max Drawdown | ≤ 25% | > 50% | Reduce position size |
| P2 | PSR (Probabilistic Sharpe) | ≥ 0.7 | < 0.3 | Sharpe not robust → more MC |
| P2 | Cross-sectional IC mean | ≥ 0.02 | < 0.0 | Weak signal → Phase 3 |

---

## APPENDIX A: FinMamba v1 → v2 Post-Mortem

| # | Bug | Root Cause | Symptom | Fix | Honest Result |
|---|-----|------------|---------|-----|---------------|
| 1 | `h[:, -1, :]` in v1 | Only last hidden state used | 1 prediction per window | Teacher forcing on all positions | 3.9M examples/epoch |
| 2 | Position 255 masked | offset=6; last 6 pos untrained | Backtest used garbage | Fixed offset alignment [L0 row a] | Position 249 still ≠ forward |
| 3 | ret_1/5/20 forward | `ln(C[t+N]/C[t])` lookahead | Model saw future as feature | Backward `ln(C[t]/C[t-N])` | Fixed |
| 4 | ADR progressive | `max(h[i:i+50])` at bar 0 | Bar 0 "knew" future bars | `np.maximum.accumulate` | Fixed |
| 5 | PnL compounding | `equity*(1+pnl)` per position | 5× multiplication | Portfolio mark-to-market | Fixed |
| 6 | z-score global | μ,σ computed on full array | Test stats leaked to train | Per-window normalization | Fixed |
| 7 | 31 features weak | RSI,MFI,BB ≈ transforms of OHLCV | No new information, noise | Raw OHLCV via VQ tokenizer | Kronos path |
| **R** | **MSE on log return** | **Convex loss → E[y]≈0** | **Sharpe −14 after all fixes** | **CE on discrete tokens** | **Kronos path** |

Bugs 1-6 FIXED. Bug 7 + Root Cause R → architectural pivot required. All honest backtests after fixing bugs 1-6 still show negative Sharpe.

---

## APPENDIX B: Future Research (NOT active strategy)

### B.1 Mamba-VQ

Replace Transformer backbone in Kronos predictor with Mamba-2 SSM. Theoretical advantages:
- Faster inference (linear vs quadratic attention)
- Better long-range dependencies (2048+ context)

**Prerequisites**: Kronos baseline MUST show alpha (Sharpe ≥ 0.5) before investing in Mamba-VQ. If even Transformer (proven) fails to find signal on MOEX, Mamba won't either.

### B.2 Multi-Scale Prediction

Train model to predict simultaneously:
- 6 bars (1 hour) — short-term alpha
- 24 bars (4 hours) — intraday swing
- 260 bars (~1 week) — trend positioning

Multi-head output, weighted CE loss by horizon.

### B.3 Ensemble with Regime Detection

Online regime classifier (volatility, trend, correlation) → condition prediction on regime. Fine-tune regime-specific prediction heads (NOT separate models — single model with conditional heads).

---

*Ported from kronos-alpha/docs/kronos_knowledge.md. Revision: 1.1.*
