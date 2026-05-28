# Kronos for Alpha — Project Plan

## Purpose
Maximize financial asset price prediction metrics for MOEX instruments.
Core tools: Kronos transformer (inference + fine-tune), MOEX ISS API, Modal GPU cloud, backtesting framework.

## Architecture
```
Kronos_for_Alpha/
├── .vscode/                 # LSP + editor configs
├── .github/                 # CI/CD workflows
├── .opencode/               # Agent config + permissions + plans
│   ├── config.json          # Agent rules, drift protection
│   ├── permissions.json     # File access control
│   └── contracts/           # Audit + module contracts
├── docs/
│   ├── grace/               # GRACE XML artifacts (6 files)
│   ├── conventions/         # Commit + CLI standards
│   ├── operations/          # Failure catalogs
│   └── reports/             # Audit + metrics + strategies
├── AGENTS.md                # Agent navigation hub (grep-first)
├── src/
│   ├── core/kronos/         # Tokenizer, model, modules, predictor
│   ├── data/                # Fetcher, cache, base abstractions
│   ├── signals/             # 7 signal families
│   ├── strategies/          # Engine + 8 strategies
│   └── evaluation/          # Metrics, walk-forward, calibration
├── templates/               # Reference examples (not production)
│   ├── sweeps/
│   ├── strategies/
│   └── scripts/
├── config/global.yaml       # Single source of truth
├── pyproject.toml           # Python project config (Ruff, MyPy, deps)
├── pyrightconfig.json       # Pyright LSP config
└── .gitignore
```

## Key Design Decisions
1. Modular: core abstractions + instrument-specific adapters
2. GRACE full integration: XML artifacts, semantic markers, grace lint
3. SSH agent-protected: separate ed25519 key, deny-read permissions
4. Commit format: Conventional Commits superset with M-XXX module IDs
5. Phase 0 quality gates from kronos-alpha docs (tokenizer → Sharpe ≥ 0.5 gate)

## Phase Summary
- A: Foundation (git, SSH, LSP, .opencode/) — DONE
- B: Documentation layer — port all docs from both artifact sets
- C: Unified global.yaml — merge alpha + artifact configs
- D: Data layer — fetcher + cache + base
- E: Core Kronos model — tokenizer, model, modules, predictor, registry
- F: Signals — 7 signal families (atoms, ict, volatility, vwap, fractal, divergence, bars)
- G: Strategies — engine + 8 strategies
- H: Evaluation — metrics, walk-forward, calibration, output
- I: Templates — experiments + scripts as reference examples
- J: GRACE skills — installed in ~/.config/opencode/skills/
- K: Raw repo analysis — audit PR_Kronos, PR_Mamba, AI_research, Antigravity
- L: GRACE XML — 6 artifacts (requirements, technology, development-plan, verification-plan, knowledge-graph, operational-packets)
- M: Verification — lint, grace lint, tests, pre-commit hooks

## Merged Sources
- /home/aaa/kronos-artifact — Complete production pipeline (8,428 lines, 61 .py)
- /home/aaa/kronos-alpha — Skeleton with agent-aware architecture (2,386 lines, 17 files)
- /home/aaa/kronos-alpha/docs/ — 5-layer knowledge architecture (1,177 lines)

## SSH Key
- Key: ~/.ssh/kronos_alpha_ed25519 (ed25519)
- Public: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDyWMzMMQHX0XsKkjG2IVgacapdzs6lIyGu2DCGHC10q kronos-alpha-agent
- Protected: chmod 600, deny-read in .opencode/permissions.json
- GitHub: add this public key to https://github.com/settings/keys
- Repository: create on GitHub, then: git remote add origin git@github.com:<user>/Kronos_for_Alpha.git
