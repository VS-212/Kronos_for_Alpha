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
| Known failures | See `docs/operations/failures.md` (11 entries): CUDNN not compiled, OOM, timeout, HF auth, stale image, checkpoint collision, ephemeral disk full |
| CLI | `python src/core/kronos/fine_tune.py --resume kronos_moex_best.pt` |
| Train config | lr=1e-5, scheduler=cosine, weight_decay=0.1, grad_clip=3.0, epochs=30, precision=bf16 |
| Checkpoint files | `kronos_moex_epoch_N.pt` (every 3 epochs), `kronos_moex_best.pt` (best val CE), `kronos_moex_latest.pt` (heartbeat), `history.json` (loss log) |

---

### M-PREDICT: Autoregressive Inference

File: `src/core/kronos/predictor.py`
Status: ✅ ready (enhanced)

| Поле | Значение |
|------|----------|
| Purpose | Generate pred_len bars autoregressively (T=0.6, top_p=0.9, MC=5) → decode to OHLCV + belief state |
| Input | `data/v3/processed/{ticker}_{split}.npy` (context: last N bars) + `kronos_moex_best.pt` + `KronosTokenizer` |
| Output | `data/v3/predictions/{ticker}_{split}.npy` + `belief/{ticker}_belief_{split}.npy` + `manifest.json` |
| Manifest keys | `ticker → {split: {shape=(T, 6), sample_count}}`; belief: `shape=(T, sample_count, pred_len, 4)` |
| Guarantees | Deterministic seed per MC path (reproducible), T4 GPU (inference < 0.5 GB VRAM for batch=1) |
| Known failures | CUDA OOM (model + tokenizer on small GPU → use T4 or CPU), Tokenizer not loaded (HF auth), Checkpoint not found (run M-FINE-TUNE first) |
| CLI | `python -m src.core.kronos.predictor --feats X --timestamps Y --output Z --pred-len 12 --lookback 500 --seed 42 --bf16 --belief` |
| CLI flags | `--feats`, `--timestamps`, `--ticker-names`, `--pred-len`, `--sample-count`, `--temperature`, `--top-p`, `--top-k`, `--lookback`, `--sub-batch` (T4: 8, A100: 16), `--model`, `--tokenizer`, `--seed`, `--bf16`, `--belief`, `--device` |
| Inference config | lookback=500 (small) / 2036 (mini), pred_len=12, temperature=0.6, top_p=0.9, top_k=50, sample_count=5, device=T4/A100 |
| Latency | ~0.45s per asset serial, ~1.5s batched (batch=21) — well within 10-min bar interval |
| Modal deployment | `modal run scripts/modal_inference.py::{seed, infer_10min, infer_10min_small, infer_10min_a100}` |
| Belief metrics | confidence, entropy_s1, top3_mass, entropy_ratio — shape `(n_windows, sample_count, pred_len, 4)` |
| Seed determinism | `torch.manual_seed(seed + step*1000 + token_type)` — batch-size-independent MC paths |
| Multi-TF | 10-min (lookback=500/2036, pred_len=12) and 1-hour (lookback=510, pred_len=2) |

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

### M-SIM: Trade Simulation

File: `src/evaluation/simulation.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Simulate single trade with TP/SL, SL priority on same bar |
| Input | `i, sig, tp_level, sl_level, entry_open, raw, LK, PL` |
| Output | `(bar_idx, trade_return, holding_period)` |
| Functions | `simulate_trade`, `get_tp_sl`, `get_tp_sl_no_sl`, `get_tp_sl_no_tp`, `_weighted_q`, `get_tp_sl_w` |
| Guarantees | SL priority over TP on same bar; horizon exit at PL; no lookahead |

---

### M-ENGINE: Backtest Execution Engine

File: `src/evaluation/engine.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Run backtest on signal array, compute all metrics |
| Input | signals array + data dict (from M-LOAD-SBER) |
| Output | metrics dict + per_bar PnL array |
| Functions | `run_backtest`, `run_backtest_custom`, `compute_all_metrics` |
| Constants | `STRAT_METRICS` (12), `PRED_METRICS` (6), `TRADE_METRICS` (4), `ANNUAL_BARS` |

---

### M-QUARTERLY: Quarterly Performance Breakdown

File: `src/evaluation/quarterly.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Per-quarter AverRet/Sharpe tables + CSV output |
| Input | labels, per_bar arrays, timestamps |
| Output | quarterly tables (printed) + CSV files |
| Functions | `compute_quarterly_tables`, `save_results_csv`, `save_quarterly_csv`, `compute_h1_2026_metrics`, `save_all_results` |

---

### M-REGIME: BB Regime Classification

File: `src/evaluation/regime.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | BB regime classification + temporal (hour/day-of-week) analysis |
| Input | data dict + per_bar arrays |
| Output | regime tables, filtered signals |
| Functions | `compute_bar_metadata`, `compute_raw_bb_mapping`, `classify_regimes`, `build_regime_masks`, `compute_quarterly_breakdown`, `champion_breakdown`, `top5_15h_16h` |

---

### M-FILTERS: Signal Filter Computation

File: `src/signals/filters.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Compute all signal filters (BB, LR, ATR, Volume, RSI, MC, rollWR) from data dict |
| Input | data dict (from M-LOAD-SBER) |
| Output | numpy arrays: filter masks, bool arrays |
| Functions (19) | `compute_bb`, `bb_width_ok`, `bb_pct_ok`, `bb_touch_ok`, `compute_lr`, `compute_atr_filter`, `compute_volume_filter`, `compute_bb_momentum`, `compute_conf_trend`, `compute_mc_breadth`, `compute_rsi14`, `compute_pred_z`, `compute_roll_wr`, `compute_mc_agreement`, `compute_weighted_quantiles`, `compute_best_mc`, `compute_drop_low_conf`, `compute_asymmetry_ratio`, `apply_filters` |

---

### M-LOAD-SBER: SBER Numpy Data Loader

File: `src/data/loader_sber.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Load SBER numpy predictions + raw features → standardized data dict |
| Input | `data/v3/predictions/10min_sber_mini/SBER_preds_pl12_sc5.npy` + `feats_test_raw.npy` |
| Output | dict with 16 keys: preds, belief, raw, ts, N, conf, entry_close, entry_open, pred_ret, actual_ret, g_tp, g_sl, wf_q90, wf_q10, close_arr, conf_per_mc, LK, PL, COMM, Q_LONG, Q_SHORT, TP_Q, SL_Q |
| Guarantees | Walk-forward quantiles (no lookahead), per-window TP/SL from MC distribution |

---

### M-REGISTRY: Unified Strategy Registry

File: `src/strategies/registry.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Unified strategy registry — lookup, import, export metrics per strategy/asset |
| Input | `registry.json` (definitions + instances), CSV (for import) |
| Output | Python API (list, lookup, import_csv, export, discover, add_strategy) + CLI |
| Functions | `list_strategies()`, `lookup_strategy()`, `import_from_csv()`, `export_registry()`, `discover_verified()`, `add_strategy()` |
| CLI | `python -m src.strategies.registry --top N --asset X`, `--lookup NAME`, `--import-csv PATH`, `--stats` |
| Guarantees | Idempotent import, single-file truth, machine-readable JSON output |

### M-CLI-BT: SBER Backtest Runner

File: `src/cli/backtest.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Run a single SBER backtest with custom parameters (PL, LK, TP/SL, commission) |
| Input | Signal array (.npy) or built-in WF strategy; `load_sber_data()` |
| Output | Metrics dict (JSON stdout) + optional append to registry.json |
| CLI | `python -m src.cli.backtest --strategy wf --name my-test --pl 12 --tp-sl default` |
| Guarantees | Reproducible (same params → same metrics), JSON machine-readable, --register appends to registry |

### M-CLI-CMP: Strategy Compare

File: `src/cli/compare.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Side-by-side comparison of two strategies (registry name or signals.npy) |
| Input | Two strategy references (registry names or .npy paths), optional PL/TP/SL override |
| Output | Table (stdout) or JSON with both metrics dicts |
| CLI | `python -m src.cli.compare --ref "WF+BB%B+BBmom+rollWR noTP" --test signals.npy` |
| Guarantees | Delta column for every metric, works with registry names and raw .npy |

---

### M-KRONOS-REGISTRY: Model Registry and Base Model Interface

File: `src/core/registry.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Model registry and base model ABC for Kronos inference (register/get/list models) |
| Input | DataFrame with columns ['open','high','low','close'] and datetime index; model name/class |
| Output | Prediction DataFrame with ['open','high','low','close','volume','amount']; registered model instances |
| Functions | `BaseModel.predict()`, `BaseModel.load()`, `BaseModel.predict_batch()`, `BaseModel.__call__()`, `register_model()`, `get_model()`, `list_models()` |
| Guarantees | Registry is singleton in process; lazy default registrations avoid circular imports; BaseModel ABC enforces `predict()` and `load()` contract; `get_model()` raises `KeyError` for unknown names |

---

### M-SIGNAL-ATOMS: Composable Signal Atoms

File: `src/signals/atoms.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Composable signal atoms from Kronos sample predictions — pure numpy, no pandas |
| Input | `(sample_count, pred_len)` close-price array + `prev_close` scalar |
| Output | dict of derived signals: direction, consensus, boundaries, dispersion, trend_strength, linearity, asymmetry, belief_weight, expectancy |
| Functions | `direction()`, `consensus()`, `boundaries()`, `dispersion()`, `trend_strength()`, `linearity()`, `asymmetry()`, `belief_weight()`, `expectancy()` |
| Guarantees | Pure numpy (no torch/pandas), composable atom interface, per-sample vectorization |

---

### M-SIGNAL-BARS: Candlestick Pattern Classifier

File: `src/signals/bars.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Japanese candlestick pattern classifier — 15 bar types with forward probability estimation |
| Input | OHLCV arrays + optional lookback/min_samples for conditional probability |
| Output | bar type string + dict of {bar_type: {p_up, p_down, avg_ret, count}} |
| Functions | `classify_bar()`, `compute_bar_probs()`, `bar_signal()`, `print_bar_summary()` |
| Guarantees | 15 distinct bar types (Doji, Marubozu, Hammer, Engulfing, Harami, etc.), minimum samples threshold, signal edge filtering |

---

### M-SIGNAL-BOLLINGER: Bollinger Bands Signal

File: `src/signals/bollinger.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Bollinger Bands computation, price position relative to bands, squeeze detection, combined signal |
| Input | `(N,)` close price array |
| Output | dict with sma, upper, lower, width, zone, pct_bandwidth, is_extreme, is_squeeze, mr_long/mr_short |
| Functions | `compute_bb()`, `bb_position()`, `bb_squeeze()`, `bb_signal()` |
| Guarantees | Pure numpy + pandas rolling, no lookahead, configurable period/std_mult, NaN-safe for early bars |

---

### M-SIGNAL-DIV: RSI/OBV/MFI Divergence Detection

File: `src/signals/divergence.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Divergence detection atoms — RSI (daily reset), OBV, MFI with swing-point divergence analysis |
| Input | OHLCV arrays + `is_day_start` boolean array |
| Output | dict with indicator values (rsi, obv, mfi) + divergence flags per indicator |
| Functions | `rsi()`, `obv_data()`, `mfi()`, `detect_divergence()`, `compute_all()` |
| Guarantees | Daily RSI/MFI reset (no gap contamination), swing-point analysis (peaks/valleys with order config), pure numpy |

### M-DATA-BASE: Abstract DataSource Interface

File: `src/data/base.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Abstract base class for financial time-series data sources |
| Input | Subclass implements: ticker, interval, start, end |
| Output | DataFrame with DatetimeIndex (begin) and OHLCV columns |
| Functions | `fetch_candles()`, `fetch_securities()`, `fetch_index_candles()` |
| Guarantees | All methods are abstract — enforces DataSource contract via ABC; no default implementation; subclasses must implement all 3 methods |

---

### M-DATA-CACHE: Parquet-Backed Data Cache

File: `src/data/cache.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Deterministic parquet-backed cache for OHLCV DataFrames |
| Input | `key(str)` — deterministic string; `DataFrame` to cache via `put()` |
| Output | Cached DataFrame via `get(key)` or `None` on miss |
| Functions | `get()`, `put()`, `key()` |
| Guarantees | Idempotent (put same key twice = overwrite same file); deterministic key generation from ticker/interval/start/end; corrupt cache files auto-removed with warning |

---

### M-STRATEGY-WF: WF Baseline Strategy

File: `src/strategies/verified/s01_wf.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Walk-forward q90/q10 signal generation with TP/SL execution |
| Input | data dict with keys: N, wf_q90, wf_q10, pred_ret |
| Output | metrics dict + per_bar PnL via run_backtest |
| Functions | `run()`, `_build_signal()` |
| Guarantees | Signal long when pred_ret > wf_q90, short when pred_ret < wf_q10; no lookahead (uses pre-computed walk-forward quantiles) |

---

### M-STRATEGY-BBPCT: WF+BB%B Strategy (no TP)

File: `src/strategies/verified/s02_bb_pct.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Walk-forward signal filtered by BB %B position, no take-profit |
| Input | data dict with keys: N, wf_q90, wf_q10, pred_ret |
| Output | metrics dict + per_bar PnL via run_backtest_custom |
| Functions | `run()`, `_build_wf_signal()` |
| Guarantees | WF signal filtered by %B position (only enter when %B confirms direction); no take-profit via get_tp_sl_no_tp |

---

### M-STRATEGY-BBMOM: WF+BB%B+BBmom Strategy (no TP)

File: `src/strategies/verified/s03_bb_mom.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Walk-forward signal filtered by BB %B position and BB momentum expansion, no take-profit |
| Input | data dict with keys: N, wf_q90, wf_q10, pred_ret |
| Output | metrics dict + per_bar PnL via run_backtest_custom |
| Functions | `run()`, `_build_wf_signal()` |
| Guarantees | WF signal filtered by %B position AND BB momentum expansion (both must confirm direction); no take-profit via get_tp_sl_no_tp |

---

### M-STRATEGY-BBROLLWR: WF+BB%B+BBmom+rollWR Champion

File: `src/strategies/verified/s04_bb_rollwr.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Champion strategy — WF signal filtered by BB %B position, BB momentum expansion, and rolling win rate |
| Input | data dict with keys: N, wf_q90, wf_q10, pred_ret |
| Output | metrics dict + per_bar PnL via run_backtest_custom |
| Functions | `run()`, `_build_wf_signal()` |
| Guarantees | Triple-filtered signal: %B position + BB momentum + rolling win rate all confirm direction; champion Sharpe 23.51 at PL=12/sc=5; no take-profit via get_tp_sl_no_tp |

### M-CALIBRATE: Hyperparameter Sweep Runner

File: `src/evaluation/calibrate.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Two-pass hyperparameter sweep over Kronos model inference config (pred_len → T/top_p/sample_count) |
| Input | config.yaml + global.yaml; KronosModel; MOEX data for calibration ticker (default SBER) |
| Output | `src/evaluation/results.json` with best config: pred_len, T, top_p, sample_count |
| Functions | `main()`, `load_config()`, `fetch_data()`, `_random_indices()`, `pass_1_pred_len_sweep()`, `pass_2_param_sweep()` |
| Guarantees | Two-pass optimization (pred_len then T/top_p/sc), cache-aware data loading, reproducible via seeded random indices |

---

### M-EVALUATE: Per-Window Evaluation Metrics

File: `src/evaluation/evaluate.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Compute 14 calibration metrics from actual OHLCV and Kronos per-sample predictions per window |
| Input | actual_df (OHLCV DataFrame), samples (ndarray S×N×6), prev_close (float), tp/sl quantiles |
| Output | dict with 14+ metrics: direction_accuracy, consensus_dir_acc, expectancy, consensus_sharpe, return_correlation, ic_rank, bias, mae, prediction_volatility, max_drawdown_simple |
| Functions | `evaluate()` |
| Guarantees | Stateless pure function, handles edge cases (flat predictions, missing scipy), SL priority over TP, consensus filtering at 4/5 threshold |

---

### M-METRICS: Financial Strategy Evaluation Metrics

File: `src/evaluation/metrics.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Standalone financial metrics library for strategy evaluation — used by M-BACKTEST, M-ENGINE, M-EVALUATE |
| Input | Returns array, equity array, pred/actual arrays |
| Output | Individual metric values (sharpe_ratio, max_drawdown, win_rate, psr, dsr, etc.) |
| Functions (18) | `sharpe_ratio()`, `sortino_ratio()`, `max_drawdown()`, `profit_factor()`, `win_rate()`, `calmar_ratio()`, `direction_accuracy()`, `direction_sharpe()`, `return_correlation()`, `ic_rank()`, `bias()`, `mae()`, `prediction_volatility()`, `avg_return()`, `n_trades()`, `trade_pct()`, `psr()`, `dsharpe_ratio()` |
| Class | `StrategyMetrics` — structured container with `from_trades()` factory |
| Guarantees | Pure functions (no side effects), optional scipy with graceful degradation, handles edge cases (empty/single-element arrays), PSR/DSR with skewness/kurtosis correction |

---

### M-OUTPUT: Compact Result Serialization

File: `src/evaluation/output.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Save/load per-window prediction samples, belief metrics, and monthly summaries to compact parquet/npy/JSON |
| Input | Windows list, ticker, config dict; DataFrame or numpy arrays |
| Output | Parquet files (samples, monthly metrics), .npy files (beliefs), JSON (summary, metadata) |
| Functions | `save_samples()`, `load_samples()`, `reconstruct()`, `compute_monthly_metrics()`, `save_monthly_metrics()`, `trade_summary()`, `save_beliefs()`, `load_beliefs()`, `save_summary()` |
| Guarantees | Compact blob storage (~700 KB for 3K windows), lossless roundtrip via np.frombuffer, metadata alongside binary data, lazy import of M-EVALUATE |

---

### M-WALK-FORWARD: Walk-Forward Validation Pipeline

File: `src/evaluation/walk_forward.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Generate non-overlapping prediction windows from Mamba dataset, batch-infer via Kronos model, save per-window samples + monthly aggregate metrics |
| Input | Mamba parquet path (from MAMBA_PATH env or config), ticker, KronosModel, config (pred_len, T, top_p, sample_count) |
| Output | Per-window samples parquet + monthly metrics parquet + summary JSON |
| Functions | `load_mamba_data()`, `extract_ticker()`, `filter_main_session()`, `generate_windows()`, `group_windows_by_month()`, `run_batch_inference()`, `main()` |
| CLI | `python -m src.evaluation.walk_forward --ticker SBER --month 2025-01`, `--full` for all months |
| Guarantees | Non-overlapping windows (step=pred_len), main-session filter (10:00-18:40), batch inference with sub-batch split to avoid GPU OOM, walk-forward date range from config |

### M-KRONOS-MODEL: Kronos Transformer Model

File: `src/core/kronos/model.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Hierarchical dual-token (s1, s2) Transformer — token IDs → logits for pre and post tokens |
| Input | `s1_ids, s2_ids` [B,T] int64, `stamp` [B,T,5] (optional), `padding_mask` (optional), teacher forcing flags |
| Output | `(s1_logits [B,T,V_s1], s2_logits [B,T,V_s2])` — logits for s1 and s2 token predictions |
| Functions | `forward()`, `decode_s1()`, `decode_s2()`, `_init_weights()` |
| Guarantees | Causal masking via RoPE, RMSNorm normalization, hierarchical dual-head with dependency-aware s2 conditioning via cross-attention, teacher-forcing support |

---

### M-KRONOS-MODULES: Kronos NN Building Blocks

File: `src/core/kronos/modules.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Reusable neural network building blocks — quantizer, attention, embeddings, normalization |
| Input | Varies by component (tensors, configuration parameters) |
| Output | Varies by component (tensor outputs, loss values, codebook indices) |
| Functions (16) | `BinarySphericalQuantizer`, `BSQuantizer`, `RMSNorm`, `FeedForward`, `RotaryPositionalEmbedding`, `MultiHeadAttentionWithRoPE`, `MultiHeadCrossAttentionWithRoPE`, `HierarchicalEmbedding`, `DependencyAwareLayer`, `TransformerBlock`, `DualHead`, `FixedEmbedding`, `TemporalEmbedding`, `DifferentiableEntropyFunction`, `codebook_entropy` |
| Guarantees | Modular composability, RoPE-based attention, BSQ with soft/hard entropy, no project-internal dependencies (pure PyTorch + einops) |

---

### M-KRONOS-PREDICTOR: Kronos Inference Runner

File: `src/core/kronos/predictor.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Autoregressive inference — tokenizer + model → OHLCV predictions + belief state |
| Input | OHLCV context [B,T,6], model + tokenizer from HuggingFace |
| Output | MC-averaged OHLCV preds [B,pred_len,6] or raw MC paths [B,sample_count,pred_len,6]; optional belief [B,sample_count,pred_len,4] with confidence, entropy, top3_mass, entropy_ratio |
| Functions | `KronosPredictor.generate()`, `.predict()`, `.predict_batch()`, `KronosModel.predict_samples()`, `.predict_samples_batch()`, `.predict_batch()`, `load_model()`, `auto_regressive_inference()`, `auto_regressive_inference_raw()`, `sample_from_logits()`, `top_k_top_p_filtering()`, `calc_time_stamps()` |
| Guarantees | Deterministic seed per MC path (batch-size-independent), bf16 precision, top-k/top-p nucleus sampling, session filtering, parallel batch inference |
| CLI | `python -m src.core.kronos.predictor --feats X --timestamps Y --output Z --pred-len 12 --lookback 500 --seed 42 --bf16 --belief` |

---

### M-KRONOS-TOKENIZER: Kronos VQ-VAE Tokenizer (Core)

File: `src/core/kronos/tokenizer.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | VQ-VAE tokenizer — encode OHLCV [B,T,6] → discrete token IDs, decode back → reconstruction |
| Input | OHLCV tensor [B,T,6] |
| Output | `((z_pre, z), bsq_loss, quantized, z_indices)` — reconstructed s1/full tensors, quantizer loss and state |
| Functions | `forward()`, `encode()`, `decode()`, `indices_to_bits()` |
| Guarantees | Binary Spherical Quantization (BSQ), HuggingFace Hub publishing (PyTorchModelHubMixin), frozen during fine-tune, supports half-mode (s1/s2 split encoding) |

### M-SIGNAL-FRACTAL: Williams Fractal Pattern Detection

File: `src/signals/fractal.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Detect Williams Fractals (bullish/bearish), fractal breakouts, fractal clusters, AO-confirmed fractal signals |
| Input | OHLCV arrays (high, low, close) + parameters: max_age (default 12), tolerance (default 0.002), min_cluster (default 2) |
| Output | dict with signal ("bullish"|"bearish"|"none"), age, level; AO value for AO variant |
| Functions | `find_fractals()`, `fractal_signal()`, `breakout_signal()`, `cluster_signal()`, `compute_ao()`, `ao_fractal_signal()` |
| Guarantees | Pure numpy, no lookahead (fractal uses 2 bars each side), cluster groups fractals by price proximity, AO uses SMA5/SMA34 median crossover |

---

### M-SIGNAL-ICT: ICT/SMC Signal Generation Toolkit

File: `src/signals/ict.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Generate ICT/Smart Money Concepts signals — order blocks, fair value gaps, liquidity sweeps, market structure shifts, breaker blocks, premium/discount zones, equal highs/lows |
| Input | OHLCV arrays + parameters (lookback, max_age, n_bars, move_threshold, tolerance, volume_mult) |
| Output | dict with signal type, level/zone info, age, confluence count |
| Functions (9) | `detect_swings()`, `detect_order_block()`, `detect_fvg()`, `detect_fvg_multi()`, `detect_liquidity_sweep()`, `detect_premium_discount()`, `detect_mss()`, `detect_breaker_block()`, `detect_volume_ob()`, `detect_eqh_eql()` |
| Guarantees | Pure numpy, configurable swing detection (3/5-bar), FVG multi-confluence detection, volume-filtered OB with configurable multiplier |

---

### M-SIGNAL-VOL: Volatility Signals (ATR/ADR/Regime)

File: `src/signals/volatility.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Compute ATR with percentile ranking, ADR (Average Daily Range) with percentile, classify volatility regime (compression/expansion/neutral) |
| Input | OHLCV arrays + is_day_start boolean array + parameters (atr_period=14, adr_period=20, percentile_threshold=0.3) |
| Output | dict with atr, atr_pct, low_vol/high_vol flags, adr, current_range, range_ratio, percentile, regime string |
| Functions | `compute_atr()`, `compute_adr()`, `volatility_regime()` |
| Guarantees | Pure numpy, ATR via EMA of True Range, ADR uses daily bars from is_day_start markers, regime combines ATR percentile + ADR range percentile |

---

### M-SIGNAL-VWAP: VWAP Signals

File: `src/signals/vwap.py`
Status: ✅ ready

| Поле | Значение |
|------|----------|
| Purpose | Compute Volume-Weighted Average Price with standard deviation bands, cross detection, anchored VWAP |
| Input | OHLCV arrays + volume + is_day_start boolean array + k (std dev multiplier, default 2.0) |
| Output | dict with vwap, upper/lower bands, sigma, zscore, pct_pos, below_vwap/above_vwap flags, near_lower/near_upper, outside_lower/outside_upper, crossed_up/crossed_down |
| Functions | `compute_vwap()`, `anchored_vwap()` |
| Guarantees | Pure numpy, VWAP resets per trading day (is_day_start), cumulative sigma computation, cross detection over last 2 candles, anchored VWAP from arbitrary start index |

---

*Ported from kronos-alpha/docs/module-contracts.md. Revision: 1.1.*
