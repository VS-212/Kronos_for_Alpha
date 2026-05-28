# Module Contracts (M-XXX)

Контракты для всех 7 модулей пайплайна Kronos fine-tune. Используется как grep-first reference: AI agent ищет `grep "M-TOKENIZE" docs/module-contracts.md` → находит I/O контракт, гарантии, известные отказы.

Смежные документы: `docs/conventions/cli.md` (CLI стандарт), `docs/operations/failures.md` (каталог ошибок Modal), `docs/architecture.md` (архитектура, фазы, gates), `config/global.yaml` (все параметры).

---

### M-FETCH: MOEX Data Fetcher

File: `src/data/fetcher.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Download OHLCV+Amount for 21 tickers from MOEX ISS API |
| Input | `--start {date} --end {date}` |
| Input placeholders | `{date}` = YYYY-MM-DD |
| Output | `data/v3/raw/{ticker}.parquet` × 21 + `manifest.json` |
| Manifest keys | `ticker → {rows, start, end}` |
| Guarantees | Idempotent, Resume (progress file per month), Rate-limit backoff (429), Retry (5, exponential `2^n × 2s`) |
| Exit codes | 0=success, 1=partial missing |
| Preflight | `--dry-run` (plan: N tickers × M months), `--status` (check manifest rows) |
| Known failures | HTTP 429 rate-limit, ConnectionError, empty holidays (expected — bars < 50% of trading days), Timeout |
| CLI | `python -m src.data.fetcher --start 2023-01-01 --end 2026-05-01` |
| Columns | `timestamp, open, high, low, close, volume, amount` |
| Tickers | 20 stocks (SBER, GAZP, LKOH, ALRS, ROSN, NVTK, PLZL, GMKN, TATN, VTBR, CHMF, NLMK, MAGN, AFLT, FIVE, MOEX, TCST, YNDX, SNGS, SNGSP) + 1 index (IMOEX) |

---

### M-PREPROCESS: Session Filter + Split

File: `src/data/preprocess.py`
Status: ❌ future

| Поле | Значение |
|------|----------|
| Purpose | Filter main session (10:00–18:40 MSK), compute amount (close × vol × lot), per-window z-score normalize (lookback=506, clip [−5, 5]), split train/val/test |
| Input | `data/v3/raw/{ticker}.parquet` + `manifest.json` |
| Input validation | Check manifest: all 21 tickers present, rows > 0 per ticker |
| Output | `data/v3/processed/{ticker}_{split}.npy` × (21 × 3) + `manifest.json` |
| Manifest keys | `ticker → {split: {rows, shape=(T, 6), start, end}}` |
| Guarantees | Idempotent, Validates input manifest (refuses partial data), Per-window μ,σ (no global stats leak) |
| Exit codes | 0=success, 1=missing ticker or empty data |
| CLI | `python -m src.data.preprocess --input data/v3/raw --output data/v3/processed` |
| Split ranges | train: 2023-01-01 → 2025-02-01, val: 2025-02-01 → 2025-09-01, test: 2025-09-01 → 2026-05-01 |
| Features | 6 columns: open, high, low, close, volume, amount (normalized via per-window z-score) |
| Known failures | Missing ticker in manifest (→ abort), NaN after normalization (→ check data), Empty split (→ check date range) |

---

### M-TOKENIZE: Kronos VQ-VAE Tokenizer

File: `src/core/kronos/tokenizer.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Encode normalized OHLCV → (s1_ids, s2_ids) via KronosTokenizer-2k |
| Input | `data/v3/processed/{ticker}_{split}.npy` |
| Input validation | Check manifest: shapes (T, 6), all 3 splits present per ticker |
| Output | `data/v3/tokens/{ticker}_{split}.npy` × (21 × 3) + `manifest.json` |
| Manifest keys | `ticker → {split: {tokens, codebook_util, entropy, recon_mse, per_asset_unique_codes}}` |
| Quality gates | codebook_util ≥ 200, entropy ≥ 5.0, recon_mse ≤ 0.01, per_asset_codes ≥ 50 (see `config/global.yaml §tokenizer.quality_gates`) |
| Guarantees | Idempotent, Quality gate validation (Phase 0 — gates MUST pass before Phase 1) |
| Known failures | Low codebook util (domain shift → retrain tokenizer on MOEX, `docs/architecture.md §2.2`), Tokenizer download (HF auth — gated model `NeoQuasar/KronosTokenizer-2k`), CUDA OOM on T4 (use CPU fallback) |
| CLI | `python -m src.core.kronos.tokenizer --input data/v3/processed --output data/v3/tokens --quality-check` |
| Tokenizer config | s1_bits=10, s2_bits=10, vocab_size=1024, d_model=256, encoder_layers=4, frozen=true |
| Phase 0 gates | If ANY gate fails → stop. Do NOT proceed to M-FINE-TUNE. Contingency: retrain tokenizer on MOEX data (8+8 bit). |

---

### M-DATASET: Predictor Dataset

File: `src/data/dataset.py`
Status: ❌ future

| Поле | Значение |
|------|----------|
| Purpose | Create sliding windows (L=512, stride=8, shift=1) over tokenized data → torch DataLoader |
| Input | `data/v3/tokens/{ticker}_{split}.npy` |
| Output | `torch.utils.data.DataLoader` yielding `(input_ids, target_ids)` both shape `(B, 512, 2)` int64 |
| Guarantees | No lookahead (shift=1 ensures `x[t] → y[t+1]`), No token shuffle across assets (each window within one ticker) |
| Note | NOT a CLI — importable library. Called by M-FINE-TUNE. |
| Config | L=512 (context), stride=8 (training stride), shift=1 (next-token prediction), pred_len=6 (future bars, teacher forcing off during inference) |
| Train windows | ~3,343 per asset (stride=8 over ~27k bars) × 21 assets ≈ 70,203 windows total |
| Known failures | DataLoader OOM (reduce batch_size), Token file missing (→ run M-TOKENIZE first) |

---

### M-FINE-TUNE: Kronos Predictor Training

File: `src/core/kronos/fine_tune.py`
Status: ❌ future

| Поле | Значение |
|------|----------|
| Purpose | Fine-tune Kronos-small predictor on MOEX tokens (tokenizer frozen), CE loss, A100 GPU |
| Input | DataLoader from M-DATASET + `NeoQuasar/Kronos-small` from HuggingFace |
| Output | `checkpoints/kronos_moex_best.pt` + `checkpoints/kronos_moex/history.json` manifest |
| Manifest keys | `{best_epoch, train_ce, val_ce, steps, epochs_completed}` |
| Guarantees | Checkpoint resume (every 3 epochs + heartbeat every 100 steps), Early stop (val CE plateau: Δ ≤ 0.001 for 5 evals) |
| Execution | `modal run src/core/kronos/fine_tune.py` (A100 40GB, timeout 8h) |
| VRAM | ~10–12 GB (batch=12, bf16, grad_accum=2 → effective batch=24) |
| Known failures | See `docs/operations/failures.md` (13 entries): CUDNN not compiled, OOM, timeout, HF auth, stale image, checkpoint collision, ephemeral disk full |
| CLI | `python src/core/kronos/fine_tune.py --resume kronos_moex_best.pt` |
| Train config | lr=1e-5, scheduler=cosine, weight_decay=0.1, grad_clip=3.0, epochs=30, precision=bf16 |
| Checkpoint files | `kronos_moex_epoch_N.pt` (every 3 epochs), `kronos_moex_best.pt` (best val CE), `kronos_moex_latest.pt` (heartbeat), `history.json` (loss log) |

---

### M-PREDICT: Autoregressive Inference

File: `src/core/kronos/predictor.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Generate pred_len=6 bars autoregressively (T=0.6, top_p=0.9, MC=4) → decode to OHLCV |
| Input | `data/v3/processed/{ticker}_{split}.npy` (context: last 506 bars) + `kronos_moex_best.pt` + `KronosTokenizer-2k` |
| Output | `data/v3/predictions/{ticker}_{split}.npy` × (21 × 3) + `manifest.json` |
| Manifest keys | `ticker → {split: {shape=(T, 6), sample_count}}` |
| Guarantees | Deterministic seed per MC path (reproducible), T4 GPU (inference < 0.5 GB VRAM) |
| Known failures | CUDA OOM (model + tokenizer on small GPU → use T4 or CPU), Tokenizer not loaded (HF auth), Checkpoint not found (run M-FINE-TUNE first) |
| CLI | `python src/core/kronos/predictor.py --checkpoint kronos_moex_best.pt --split test` |
| Inference config | lookback=506 bars, pred_len=6 bars, temperature=0.6, top_p=0.9, top_k=0, sample_count=4, device=T4 |
| Latency | ~0.45s per asset serial, ~1.5s batched (batch=21) — well within 10-min bar interval |

---

### M-BACKTEST: Cross-Sectional Backtest

File: `src/evaluation/backtest.py`
Status: ❌ future

| Поле | Значение |
|------|----------|
| Purpose | Cross-sectional z-score ranking (top-3 long, bot-2 short), session-enforced entry/exit, compute metrics via `src/evaluation/metrics.py` |
| Input | `data/v3/predictions/{ticker}_{split}.npy` + `data/v3/raw/{ticker}.parquet` (for actual prices) |
| Output | `data/v3/results/report.json` + `equity_curve.npy` |
| Manifest keys | `{sharpe, max_dd, win_rate, profit_factor, calmar, psr, dsr, ic_mean, total_return, total_trades, long_trades, short_trades}` |
| Guarantees | Walk-forward validation (val vs test comparison), Session enforcement (enter 10:00–18:10, force-exit 18:35), No lookahead bias, Metrics via reusable `src/evaluation/metrics.py` |
| Metrics gates | See `config/global.yaml §backtest.metrics_gates`: sharpe ≥ 0.5, win_rate ≥ 52%, max_dd ≤ 25%, psr ≥ 0.7, ic_mean ≥ 0.02 |
| CLI | `python -m src.evaluation.backtest --predictions data/v3/predictions --split test` |
| Strategy config | long=3, short=2, stop_loss=-1.5%, take_profit=+2.5%, commission=0.03%, slippage=0.01% |
| Known failures | Not enough tickers for top-3/bot-2 (→ reduce N), All predictions correlated → no spread (→ check M-TOKENIZE per-asset codes), Metrics gates fail → go to Phase 3 iteration (`docs/architecture.md §2.3`) |

---

## Inter-Step Manifest Chain

Each module validates the previous module's manifest before executing. This is the chain of trust:

```
M-FETCH manifest.json
  │  keys: ticker → {rows, start, end}
  │
  ▼ M-PREPROCESS validates: all 21 tickers present, rows > 0
M-PREPROCESS manifest.json
  │  keys: ticker → {split: {rows, shape=(T,6), start, end}}
  │
  ▼ M-TOKENIZE validates: shapes correct (T,6), all 3 splits per ticker
M-TOKENIZE manifest.json
  │  keys: ticker → {split: {tokens, codebook_util, entropy, recon_mse}}
  │
  ▼ M-DATASET validates: tokens present, quality gates passed
M-DATASET (no manifest — returns DataLoader in memory)
  │
  ▼ M-FINE-TUNE (next step in Python process)
M-FINE-TUNE history.json + kronos_moex_best.pt
  │  keys: {best_epoch, train_ce, val_ce}
  │
  ▼ M-PREDICT validates: checkpoint file exists, val_ce < random_CE
M-PREDICT manifest.json
  │  keys: ticker → {split: {shape=(T,6), sample_count}}
  │
  ▼ M-BACKTEST validates: predictions exist for all 21 tickers per split
M-BACKTEST report.json
     keys: {sharpe, max_dd, win_rate, psr, ic_mean, total_return}
```

**Rule**: if a module cannot validate the predecessor's manifest, it MUST exit with code 1 and print the missing/broken key. Do NOT skip validation.

---

*Ported from kronos-alpha/docs/module-contracts.md. Revision: 1.1.*
