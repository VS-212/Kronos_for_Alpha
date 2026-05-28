# Commit Conventions (AI-friendly)

Каждый коммит в этом репозитории — **запись в контрактном логе**. Написан так, чтобы следующий AI-агент мог за секунду понять: что сделано, к какому модулю, зачем, проверено ли.

## Формат

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

Multi-module convention: `feat(M-FETCH,M-PREPROCESS): ...`

Only `<type>(M-XXX):` and `Contract:` are mandatory. Other fields are optional — include them when applicable.

## Field Semantics

| Field | Required | Definition |
|---|---|---|
| Contract: | Always | Module I/O contract from module-contracts.md |
| Added: | When new files created | New artifact not existing before |
| Changed: | When existing code modified | Modifications to existing behavior/APIs |
| Removed: | When code deleted | Deleted files/functions |
| Deprecated: | When marking for removal | Soft removal with migration path |
| Why: | For architectural decisions | Why this approach vs alternatives |
| Phase: | Recommended | Pipeline phase 0-3 from architecture.md |
| Verified: | Strongly recommended | Exact command + expected/actual result |

## Security Rule

**Never include actual secret values.** Use secret IDs only (e.g., `'HF_TOKEN (stored in Modal Secrets)'`). Verified: commands must not include credentials.

## Примеры

### Создание модуля

```
feat(M-PREPROCESS): session filter + per-window z-score normalization

Contract: data/v3/raw/{ticker}.parquet → data/v3/processed/{ticker}_{split}.npy + manifest.json
Added:    src/data/preprocess.py (CLI), --start/--end/--lookback flags, session filter 10:00-18:40
Phase:    Phase 0
Verified: python -m src.data.preprocess --start 2023-01-01 --end 2023-03-01 output OK
```

### Исправление бага

```
fix(M-FETCH): fix --resume flag inversion (was --no-resume)

Contract: --resume (default: true) per docs/conventions/cli.md §2
Changed:  --no-resume → --resume, default=True
Verified: python -m src.data.fetcher --help shows --resume, exit 0
```

### Инфраструктура

```
feat(M-INFRA): Kronos Modal image (CUDA 12.4, torch 2.4, Kronos-small)

Contract: modal.Image.from_registry(nvidia/cuda:12.4.0) + pip install → deployable
Added:    src/core/kronos/image.py, HF_TOKEN secret
Verified: modal image build ok, modal run test on A100
```

### Multi-module commit

```
feat(M-FETCH,M-PREPROCESS): add amount computation and lot_size table

Contract: raw OHLCV → OHLCV + amount (close × volume × lot_size)
Added:    src/data/lot_sizes.py (per-ticker lot sizes), amount column in M-FETCH output
Changed:  M-FETCH output schema: 6 cols → 7 cols
Phase:    Phase 0
Verified: python -m src.data.fetcher --dry-run confirms 7 cols output
```

### Architectural decision

```
refactor(M-PREDICT): switch from single MC path to ensemble of 4

Contract: (context_bars) → 4 MC paths × 6 predicted bars → averaged OHLCV
Why:      Single path shows Sharpe 0.2; ensemble of 4 improves to 0.5.
          MC averaging reduces variance without adding bias.
          Alternative considered: 8 paths — diminishing returns beyond 4.
Changed:  predictor.py inference loop: sample_count 1 → 4, added reshape → mean
Phase:    Phase 2
Verified: python src/core/kronos/predictor.py --split test --mc 4 → Sharpe 0.52 vs 0.21
```

### Breaking change

```
feat(M-BACKTEST): replace long-only with cross-sectional top-3/bot-2

Contract: asset predictions → cross-sectional z-score ranking → portfolio weights
Removed:  legacy long-only weight allocation, equal_weight.py
Deprecated: --strategy long-only (use --strategy cs with --long 3 --short 2)
Why:      Long-only Sharpe 0.3 vs cross-sectional Sharpe 0.8. CS captures spread.
Phase:    Phase 2
BREAKING CHANGE: --strategy long-only removed. Use --strategy cs --long 3 --short 2.
```

## Как AI-агент использует коммиты

| Задача | Команда | Зачем |
|--------|---------|-------|
| Быстрый обзор | `git log --oneline` | 3 секунды — какие модули менялись |
| История модуля | `git log --grep M-FETCH` | Все коммиты по одному модулю |
| Детали изменений | `git show <hash>` | Что/зачем/проверено — читать только 1 коммит |
| Откат | `git revert <hash>` | Создаёт revert с контрактом в сообщении |
| Поиск решения | `git log --grep "Verified:"` | Что уже проверяли и как |

## Почему это важно для AI

1. **Контекст экономится**: `git show` даёт контрактную запись (5 строк), а не `git diff` (200 строк кода)
2. **Grep-first**: `git log --grep M-TOKENIZE` находит все изменения токенизатора без чтения кода
3. **Воспроизводимость**: поле `Verified:` содержит точную команду проверки — AI может её повторить
4. **Причина решения**: поле `Contract:` объясняет ЗАЧЕМ, не только ЧТО
5. **Safety**: Security rule prevents credential leaks in commit history

---

*Ported from kronos-alpha/docs/commit-conventions.md. Revision: 1.1.*
