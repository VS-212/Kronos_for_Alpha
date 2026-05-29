# Kronos for Alpha — Agent Entry Point

## PURPOSE
Maximize price prediction metrics for MOEX financial assets via:
- Kronos transformer fine-tune (CE loss on discrete tokens)
- Multi-signal alpha generation (8 families — see `src/signals/__init__.py` for catalog)
- Strategy backtesting with walk-forward validation
- Cross-sectional z-score ranking for alpha extraction
See: `docs/requirements.xml`

## SCOPE
IN:
  M-FETCH      MOEX ISS data ingestion
  M-PREPROCESS Session filter, per-window z-score, train/val/test split
  M-TOKENIZE   Kronos VQ-VAE tokenizer inference
  M-DATASET    Sliding window dataset (L=512, stride=8)
  M-FINE-TUNE  Kronos predictor training (CE loss, A100)
  M-PREDICT    Autoregressive inference → OHLCV predictions
  M-BACKTEST   Cross-sectional backtest (top-K long/short)
  M-METRICS    Strategy evaluation
  M-CONFIG     Single source of truth (config/global.yaml)
OUT:
  Real-time execution, portfolio optimization, order management
  (These require infrastructure beyond prediction + backtesting)
See: `docs/requirements.xml`, `docs/module-contracts.md`

## KEY METRICS
Strategy evaluation (from `src/evaluation/metrics.py`):
  WinRate, Sharpe, MaxDrawdown, ProfitFactor, Calmar, Sortino,
  IcRank, AvgReturn, DirectionAccuracy, Bias, MAE, PSR, DSR,
  N-trades, trade_pct, PredictionVolatility

Quality gates (thresholds in `config/global.yaml §quality_gates`):
  P0 tokenizer, P1 fine-tune, P2 backtest
See: `config/global.yaml`, `docs/verification-plan.xml`, `docs/requirements.xml §success_criteria`

## MODULE CHANGE CHECKLIST
When adding or modifying a module (new strategy, new signal, new pipeline step),
update ALL of these or `grace lint` will flag the mismatch:

1. **Code** — `src/<layer>/<module>.py` + update `__init__.py`
2. **Config** — `config/global.yaml` (if new parameters needed)
3. **Contract** — `docs/module-contracts.md` (M-XXX: purpose, input, output, guarantees, CLI)
4. **Plan** — `docs/development-plan.xml` `<module-registry>` (add `<M-XXX></M-XXX>`)
5. **Graph** — `docs/knowledge-graph.xml` `<module-registry>` (add `<M-XXX></M-XXX>`)
6. **Verification** — `docs/verification-plan.xml` (add V-M-XXX entry if the module has quality gates)
7. **Signals catalog** — `src/signals/__init__.py` (if adding a new signal family)
8. **Module table** — `AGENTS.md` M-XXX table (update status/file)
9. **Verify** — `ruff check && grace lint --profile standard`

## DEVELOPMENT PRINCIPLES
1. **Contract-first**: define M-XXX I/O in `module-contracts.md` before implementing
2. **One layer — one direction**: imports follow dependency graph (enforced by `.importlinter`)
3. **Reproducible**: every commit has `Verified:` with exact command + result
4. **Template-isolated**: experiments in `templates/`, production in `src/`
5. **Single config source**: all params in `config/global.yaml`, not hardcoded
6. **Grace-gated**: changes pass `grace lint --profile standard`
7. **No invented scope**: if a feature is not in `module-contracts.md`, it is OUT

## WORKFLOW
```
research → plan → review → implement → verify
  ↑         ↑        ↑         ↑            ↑
  │         │        │         │            └── ruff + grace lint + grace-checklist
  │         │        │         └── sub-agents per M-XXX, commit (Verified:)
  │         │        └── ruff + import-linter + grace-reviewer + grace-checklist
  │         └── grace-plan + knowledge-graph.xml cross-check
  └── grep M-XXX + sub-agent explore
```

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
grep "GRACE"         → docs/                                (GRACE XML artifacts)
```

## M-XXX Module ID Namespace

| ID | File | Status |
|---|---|---|---|
| M-FETCH | src/data/fetcher.py | ✅ ready |
| M-PREPROCESS | src/data/preprocess.py | ❌ future |
| M-TOKENIZE | src/core/kronos/tokenizer.py | ✅ ready |
| M-DATASET | src/data/dataset.py | ❌ future |
| M-FINE-TUNE | src/core/kronos/fine_tune.py | ❌ future |
| M-PREDICT | src/core/kronos/predictor.py | ✅ ready (enhanced) |
| M-BACKTEST | src/evaluation/backtest.py | ❌ future |
| M-METRICS | src/evaluation/metrics.py | ✅ ready |
| M-CONFIG | config/global.yaml | ✅ ready |
| M-CALIBRATE | src/evaluation/calibrate.py | ✅ ready |
| M-INFRA | scripts/modal_inference.py | ✅ ready |
| M-DOCS | docs/ | ✅ ready |
| M-CI | .github/workflows/ | ❌ future |
| M-SIM | src/evaluation/simulation.py | ✅ ready |
| M-ENGINE | src/evaluation/engine.py | ✅ ready |
| M-QUARTERLY | src/evaluation/quarterly.py | ✅ ready |
| M-REGIME | src/evaluation/regime.py | ✅ ready |
| M-FILTERS | src/signals/filters.py | ✅ ready |
| M-LOAD-SBER | src/data/loader_sber.py | ✅ ready |
| M-STRATEGY-WF | src/strategies/verified/s01_wf.py | ✅ ready |
| M-STRATEGY-BBPCT | src/strategies/verified/s02_bb_pct.py | ✅ ready |
| M-STRATEGY-BBMOM | src/strategies/verified/s03_bb_mom.py | ✅ ready |
| M-STRATEGY-BBROLLWR | src/strategies/verified/s04_bb_rollwr.py | ✅ ready |
| M-REGISTRY | src/strategies/registry.py | ✅ ready |
| M-CLI-BT | src/cli/backtest.py | ✅ ready |
| M-CLI-CMP | src/cli/compare.py | ✅ ready |
| M-KRONOS-MODEL | src/core/kronos/model.py | ✅ ready |
| M-KRONOS-MODULES | src/core/kronos/modules.py | ✅ ready |
| M-KRONOS-TOKENIZER | src/core/kronos/tokenizer.py | ✅ ready |
| M-KRONOS-PREDICTOR | src/core/kronos/predictor.py | ✅ ready |
| M-KRONOS-REGISTRY | src/core/registry.py | ✅ ready |
| M-DATA-BASE | src/data/base.py | ✅ ready |
| M-DATA-CACHE | src/data/cache.py | ✅ ready |
| M-EVALUATE | src/evaluation/evaluate.py | ✅ ready |
| M-OUTPUT | src/evaluation/output.py | ✅ ready |
| M-WALK-FORWARD | src/evaluation/walk_forward.py | ✅ ready |
| M-SIGNAL-ATOMS | src/signals/atoms.py | ✅ ready |
| M-SIGNAL-BARS | src/signals/bars.py | ✅ ready |
| M-SIGNAL-BOLLINGER | src/signals/bollinger.py | ✅ ready |
| M-SIGNAL-DIV | src/signals/divergence.py | ✅ ready |
| M-SIGNAL-FRACTAL | src/signals/fractal.py | ✅ ready |
| M-SIGNAL-ICT | src/signals/ict.py | ✅ ready |
| M-SIGNAL-VOL | src/signals/volatility.py | ✅ ready |
| M-SIGNAL-VWAP | src/signals/vwap.py | ✅ ready |
| M-STRATEGY-CORE | src/strategies/pending/core.py | ✅ ready |
| M-STRATEGY-VANILLA | src/strategies/pending/vanilla.py | ✅ ready |
| M-STRATEGY-S01 | src/strategies/pending/s01_bb.py | ✅ ready |
| M-STRATEGY-S02 | src/strategies/pending/s02_bb_mr.py | ✅ ready |
| M-STRATEGY-S05 | src/strategies/pending/s05_bb_breakout.py | ✅ ready |
| M-STRATEGY-S20 | src/strategies/pending/s20_ob.py | ✅ ready |
| M-STRATEGY-S28 | src/strategies/pending/s28_vol_ob.py | ⚠️ failed-may2026 |
| M-STRATEGY-S34 | src/strategies/pending/s34_vwap_ob.py | ⚠️ failed-may2026 |
| M-STRATEGY-S38 | src/strategies/pending/s38_lowvol_ob.py | ✅ ready |
| M-DATA-MOEX | src/data/moex.py | ❌ future |

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
│   ├── signals/              # 8 signal families + BB filters + MC filters
│   ├── strategies/           # verified/ (champions) + pending/ (unverified)
│   └── evaluation/           # Metrics, simulation, engine, quarterly, regime, backtest, calibration
│
├── templates/                # Reference examples (not production)
│   ├── sweeps/
│   └── scripts/
│
├── docs/
│   ├── development-plan.xml, knowledge-graph.xml, etc. # GRACE XML artifacts (6 files)
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

## Pipeline (7 modules + infra)

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
M-PREDICT      src/core/kronos/predictor.py Autoregressive inference (T=0.6, MC=5, seed) → OHLCV + belief
  ↓
M-BACKTEST     src/evaluation/backtest.py   Cross-sectional (top-3 long, bot-2 short) → Sharpe
```

**M-PREDICT enhancements**: belief extraction (confidence, entropy, top3_mass, entropy_ratio per MC path), batch mode (`--sub-batch N`), seed determinism (per-call reset), bf16 precision, Modal GPU deployment.

## SBER Single-Asset Evaluation Pipeline

```
run_sber.py  templates/scripts/run_sber.py    (orchestrator, 940 lines)
  ↓
M-LOAD-SBER  src/data/loader_sber.py          Load SBER numpy → data dict
  ↓
M-FILTERS    src/signals/filters.py           BB, LR, Tier1, Tier2, MC filters
  ↓
M-ENGINE     src/evaluation/engine.py         Backtest runner + metric computation
  ↓
M-SIM        src/evaluation/simulation.py     Trade simulation with TP/SL
  ↓
M-QUARTERLY  src/evaluation/quarterly.py      Quarterly breakdown + CSV output
  ↓
M-REGIME     src/evaluation/regime.py         BB regime + temporal analysis
```

## GRACE Artifact Management Pipeline

```
make grace-deep-dry  (or scripts/grace-full-refresh.sh)
  ├─ grace lint --profile autonomous  → Level 1 gate
  └─ python scripts/audit_grace_coverage.py  → manifest.json (gaps report)

@grace-orchestrator  (native opencode agent — type in any session)
  ├─ Audit → manifest.json
  ├─ Work split (max 8 units/agent)
  ├─ Spawn N × @grace-worker (hidden subagent)
  │   └─ Each: scan layer → add KG nodes/edges → add contracts → add V-M entries
  ├─ Post-flight: cross-layer edges + AGENTS.md table + lint + reviewer
  └─ Commit + report (docs/grace-work/sessions/{ts}/report.md)

Agents: .opencode/agents/grace-orchestrator.md, .opencode/agents/grace-worker.md
Config: config/global.yaml → grace.deep_refresh
```

## Agent Commands

| Command | Module | Status |
|---|---|---|
| `python -m src.data.fetcher --start 2023-01-01 --end 2026-05-01` | M-FETCH | ✅ ready |
| `python -m src.data.fetcher --status` | M-FETCH | ✅ ready |
| `python -m src.data.fetcher --dry-run` | M-FETCH | ✅ ready |
| `python -m src.core.kronos.predictor --feats X --timestamps Y --output Z --belief` | M-PREDICT | ✅ ready |
| `modal run scripts/modal_inference.py::seed` | M-INFRA | ✅ ready |
| `modal run scripts/modal_inference.py::infer_10min` | M-INFRA | ✅ ready |
| `modal run scripts/modal_inference.py::infer_10min_small` | M-INFRA | ✅ ready |
| `modal run --detach scripts/modal_inference.py::infer_10min_a100` | M-INFRA | ✅ ready |
| `modal run src/core/kronos/fine_tune.py` | M-FINE-TUNE | ❌ future |
| `python scripts/fetch_1h.py` | M-FETCH | ✅ ready |
| `python scripts/extract_ticker.py --ticker SBER` | M-FETCH | ✅ ready |
| `python scripts/align_predictions.py --tf10-preds ... --tf1h-preds ...` | M-PREDICT | ✅ ready |
| `python templates/scripts/run_sber.py` | M-ENGINE | ✅ ready |
| `python templates/scripts/backtest_sber_v2.py` | M-ENGINE | ✅ ready |
| `python -m src.strategies.registry --top 10 --asset SBER` | M-REGISTRY | ✅ ready |
| `python -m src.strategies.registry --lookup "rollWR"` | M-REGISTRY | ✅ ready |
| `python -m src.strategies.registry --stats` | M-REGISTRY | ✅ ready |
| `python -m src.strategies.registry --discover` | M-REGISTRY | ✅ ready |
| `python -m src.strategies.registry --import-csv PATH` | M-REGISTRY | ✅ ready |
| `python -m src.cli.backtest --strategy wf --name my-test --pl 12` | M-CLI-BT | ✅ ready |
| `python -m src.cli.compare --ref "WF+BB%B+BBmom+rollWR noTP" --test signals.npy` | M-CLI-CMP | ✅ ready |
| `make grace-deep-dry` | M-GRACE | ✅ ready |
| `@grace-orchestrator` (in opencode session) | M-GRACE | ✅ ready |
| `python scripts/audit_grace_coverage.py --summary` | M-GRACE | ✅ ready |
| _rest_ | M-* | ❌ future |

## Docs (read by situation)

| File | When |
|---|---|
| `docs/architecture.md` | Understand Kronos architecture and pipeline design |
| `docs/conventions/cli.md` | Creating or modifying a CLI module |
| `docs/module-contracts.md` | Learn module contract (M-XXX → input/output/guarantees) |
| `docs/conventions/commit.md` | Making a commit — format spec |
| `docs/operations/failures.md` | Modal job crashed — exact-match symptom from log |
| `docs/` | GRACE XML artifacts (requirements, technology, development-plan, verification-plan, knowledge-graph, operational-packets) |
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

- **Kronos-small**: CE on discrete tokens avoids mean collapse and produces distributional predictions. ⚠ Tokenizer differs from mini: mini → `Kronos-Tokenizer-2k`, small → `Kronos-Tokenizer-base`
- **21 asset** (20 stocks + IMOEX): more diversity → stronger cross-sectional z-score
- **One model, 21 assets**: per-window z-score removes price level. No asset-ID embedding
- **Tokenizer frozen**: Phase 0 quality gate validates KronosTokenizer on MOEX data first
- **Walk-forward split**: train 2023→2025, val 2025-02→2025-09, test 2025-09→2026-05
- **CLI, not MCP**: batch operations, fire-and-forget. CLI via bash tool
- **GRACE integration**: XML artifacts in `docs/`, semantic markers, grace lint
- **Seed determinism**: per-call `torch.manual_seed(seed + step*1000 + token_type)` ensures batch-size-independent reproducibility
- **bf16 stability**: `.float()` before softmax prevents bf16 rounding from changing token selection between T4 and A100
- **Belief state**: entropy, confidence, top3_mass extracted per MC path per autoregressive step — zero-cost from existing logits
- **Sub-batch mode**: `sub_batch=8` (T4) / `sub_batch=16` (A100) — balances VRAM vs throughput
- **Modal volumes**: `kronos-hf-cache` (model cache), `kronos-predictions` (output) — persist across runs
- **DVR refactoring**: monolith `backtest_sber_v2.py` (1556 lines) decomposed into 7 modules + orchestrator using DVR (Decompose→Verify→Replace) + Strangler Fig + Contract-First. Verified: 107 strategies bit-identical after refactoring.
- **Pending/verified split**: `src/strategies/verified/` = validated on SBER numpy PL=12/sc=5. `src/strategies/pending/` = from Mamba pipeline, NOT validated.
