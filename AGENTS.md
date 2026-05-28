# Kronos for Alpha — Agent Entry Point

## Grep-First Navigation

```
grep "M-FETCH"       → src/data/fetcher.py              (input/output contract)
grep "M-TOKENIZE"    → docs/module-contracts.md          (M-TOKENIZE spec)
grep "M-PREDICT"     → src/core/kronos/predictor.py      (inference contract)
grep "M-ERROR"       → docs/operations/failures.md       (exact-match log string)
grep "M-COMMIT"      → docs/conventions/commit.md        (commit format)
grep "CLI"           → docs/conventions/cli.md           (how to write CLI modules)
grep "CONFIG"        → config/global.yaml                (all parameters)
grep "M-METRICS"     → src/evaluation/metrics.py         (Sharpe, MaxDD, WR, PSR)
grep "M-BACKTEST"    → docs/module-contracts.md          (M-BACKTEST spec)
grep "GRACE"         → docs/grace/                       (GRACE XML artifacts)
```

## M-XXX Module ID Namespace

| ID | File | Status |
|---|---|---|
| M-FETCH | src/data/fetcher.py | ✅ ready |
| M-PREPROCESS | src/data/preprocess.py | ❌ future |
| M-TOKENIZE | src/core/kronos/tokenizer.py | ✅ ready |
| M-DATASET | src/data/dataset.py | ❌ future |
| M-FINE-TUNE | src/core/kronos/fine_tune.py | ❌ future |
| M-PREDICT | src/core/kronos/predictor.py | ✅ ready |
| M-BACKTEST | src/evaluation/backtest.py | ❌ future |
| M-METRICS | src/evaluation/metrics.py | ✅ ready |
| M-CONFIG | config/global.yaml | ✅ ready |
| M-INFRA | — | ❌ future |
| M-DOCS | docs/ | ✅ ready |
| M-CI | .github/workflows/ | ❌ future |

## Directory

```
Kronos_for_Alpha/
├── AGENTS.md                 # Entry point (this file)
├── config/
│   └── global.yaml           # Single source of truth (all params)
│
├── src/
│   ├── core/kronos/          # Tokenizer, model, modules, predictor, fine_tune
│   ├── data/                 # Fetcher, preprocess, dataset, base abstractions
│   ├── signals/              # 7 signal families (atoms, ict, volatility, vwap, fractal, divergence, bars)
│   ├── strategies/           # Engine + 8 strategies
│   └── evaluation/           # Metrics, backtest, walk-forward, calibration
│
├── templates/                # Reference examples (not production)
│   ├── sweeps/
│   ├── strategies/
│   └── scripts/
│
├── docs/
│   ├── grace/                # GRACE XML artifacts (6 files)
│   ├── architecture.md       # Master architecture document
│   ├── module-contracts.md   # M-XXX contracts for all pipeline modules
│   ├── conventions/
│   │   ├── cli.md            # CLI standard for all modules
│   │   └── commit.md         # Commit format specification
│   ├── operations/
│   │   └── failures.md       # Exact-match failure catalog (Modal, CUDA, torch)
│   └── reports/
│       ├── audit.md          # Leakage audit report
│       ├── metrics.md        # Performance metrics
│       └── strategies.md     # Strategy catalog
│
├── pyproject.toml            # Python project config (Ruff, MyPy, deps)
├── pyrightconfig.json        # Pyright LSP config
└── .gitignore
```

## Pipeline (7 modules)

```
M-FETCH        src/data/fetcher.py          MOEX ISS API → 21 parquet + manifest.json
  ↓
M-PREPROCESS   src/data/preprocess.py       Session filter, amount, per-window z-score, split
  ↓
M-TOKENIZE     src/core/kronos/tokenizer.py KronosTokenizer → (s1_ids, s2_ids)
  ↓
M-DATASET      src/data/dataset.py          Sliding windows (L=512, stride=8) → DataLoader
  ↓
M-FINE-TUNE    src/core/kronos/fine_tune.py Kronos-small, freeze tokenizer, CE loss, A100
  ↓
M-PREDICT      src/core/kronos/predictor.py Autoregressive inference (T=0.6, MC=4) → OHLCV
  ↓
M-BACKTEST     src/evaluation/backtest.py   Cross-sectional (top-3 long, bot-2 short) → Sharpe
```

## Agent Commands

| Command | Module | Status |
|---|---|---|
| `python -m src.data.fetcher --start 2023-01-01 --end 2026-05-01` | M-FETCH | ✅ ready |
| `python -m src.data.fetcher --status` | M-FETCH | ✅ ready |
| `python -m src.data.fetcher --dry-run` | M-FETCH | ✅ ready |
| `modal run src/core/kronos/fine_tune.py` | M-FINE-TUNE | ❌ future |
| _rest_ | M-* | ❌ future |

## Docs (read by situation)

| File | When |
|---|---|
| `docs/architecture.md` | Understand architecture, why Kronos, why not Mamba |
| `docs/conventions/cli.md` | Creating or modifying a CLI module |
| `docs/module-contracts.md` | Learn module contract (M-XXX → input/output/guarantees) |
| `docs/conventions/commit.md` | Making a commit — format spec |
| `docs/operations/failures.md` | Modal job crashed — exact-match symptom from log |
| `docs/grace/` | GRACE XML artifacts (requirements, technology, development-plan, verification-plan, knowledge-graph, operational-packets) |
| `config/global.yaml` | All parameters (ticker, split, model, train, backtest) |

## Commit Format Specification

```
<type>(M-XXX): <description>

Contract: <input → output, what contract was fulfilled>
Added:    <new files/interfaces created>
Changed:  <existing code modifications>
Removed:  <deleted files/deprecated>
Deprecated: <soft removal with migration path>
Why:      <architectural decision rationale>
Phase:    <pipeline phase: 0-3>
Verified: <exact reproduction command + result>
Issue:    <linked issue number>
Refs:     <related commits/modules>
BREAKING CHANGE: <description of what breaks>
```

Where `type` ∈ {feat, fix, refactor, docs, test, ci, chore, perf, revert}.

Multi-module: `feat(M-FETCH,M-PREPROCESS): ...`

See `docs/conventions/commit.md` for full field semantics, security rules, and examples.

## Key Decisions

- **Kronos-small** (not FinMamba v2): MSE — mean collapse (Sharpe -14). CE on discrete tokens avoids this
- **21 asset** (20 stocks + IMOEX): more diversity → stronger cross-sectional z-score
- **One model, 21 assets**: per-window z-score removes price level. No asset-ID embedding
- **Tokenizer frozen**: Phase 0 quality gate validates KronosTokenizer on MOEX data first
- **Walk-forward split**: train 2023→2025, val 2025-02→2025-09, test 2025-09→2026-05
- **CLI, not MCP**: batch operations, fire-and-forget. CLI via bash tool
- **GRACE integration**: XML artifacts in `docs/grace/`, semantic markers, grace lint

## Project Goal

Fine-tune Kronos-small (VQ-VAE tokenizer + Transformer predictor, CE loss) on MOEX 21 assets for cross-sectional alpha.
