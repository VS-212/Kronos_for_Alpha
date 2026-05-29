#!/usr/bin/env bash
# GRACE Full Refresh — audit only (Level 1 gate)
# For full deep refresh, use the native opencode agent:
#   @grace-orchestrator
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"

DRY_RUN=false
AUTO_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run | --auto) shift ;;  # kept for backward compat
        -h|--help)
            echo "GRACE Audit & Artifact Refresh"
            echo ""
            echo "  Quick audit:"
            echo "    make grace-deep-dry"
            echo "    python3 scripts/audit_grace_coverage.py --summary"
            echo ""
            echo "  Full deep refresh (native opencode agent):"
            echo "    @grace-orchestrator   ← type this in opencode"
            echo ""
            echo "  The @grace-orchestrator agent:"
            echo "    - Runs coverage audit"
            echo "    - Splits work by layer"
            echo "    - Spawns grace-worker subagents in parallel"
            echo "    - Merges results, adds cross-layer edges"
            echo "    - Updates AGENTS.md, commits"
            exit 0 ;;
        *) shift ;;
    esac
done

cd "$PROJECT_ROOT"

echo "=== GRACE Audit ==="
echo ""

if command -v grace &>/dev/null; then
    grace lint --profile autonomous --path . 2>&1 || echo "  lint: issues found"
fi

ruff check src/ 2>&1 || true
echo ""

python3 scripts/audit_grace_coverage.py --summary
echo ""

# Generate manifest for deep refresh
mkdir -p "$PROJECT_ROOT/docs/grace-work/sessions/$TIMESTAMP"
python3 scripts/audit_grace_coverage.py -o "$PROJECT_ROOT/docs/grace-work/sessions/$TIMESTAMP/manifest.json"

echo "For deep refresh: @grace-orchestrator"
echo "Manifest: docs/grace-work/sessions/$TIMESTAMP/manifest.json"
