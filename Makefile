.PHONY: grace-deep-dry audit lint

# Quick audit — no changes, just report
grace-deep-dry:
	python3 scripts/audit_grace_coverage.py --summary

audit: grace-deep-dry

lint:
	ruff check src/

# Full deep refresh — invoke native opencode agent
# Type this in any opencode session:
#   @grace-orchestrator
#
# The orchestrator will:
#   1. Run coverage audit → manifest.json
#   2. Split work by layer (max 8 modules/agent)
#   3. Spawn grace-worker subagents in parallel
#   4. Merge results + cross-layer edges + AGENTS.md
#   5. Commit with session report
