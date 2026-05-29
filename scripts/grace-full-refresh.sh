#!/usr/bin/env bash
# GRACE Full Refresh — Level 2 deep sync
# Usage:
#   make grace-deep          → full run (interactive)
#   scripts/grace-full-refresh.sh --dry-run  → audit only, no changes
#   scripts/grace-full-refresh.sh --auto     → non-interactive opencode run
#   scripts/grace-full-refresh.sh --layer evaluation  → single layer
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
SESSION_DIR="$PROJECT_ROOT/docs/grace-work/sessions/$TIMESTAMP"

# ── Flags ──
DRY_RUN=false
AUTO_MODE=false
LAYER_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --auto)     AUTO_MODE=true; shift ;;
        --layer)    LAYER_FILTER="$2"; shift 2 ;;
        -h|--help)
            echo "GRACE Full Refresh (Level 2)"
            echo "  --dry-run    Audit only, report gaps (no changes)"
            echo "  --auto       Non-interactive run with --approve-all"
            echo "  --layer X    Only process one layer directory"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

# ── Level 1 Gate ──
echo "=== Level 1: grace lint ==="
if command -v grace &>/dev/null; then
    grace lint --profile autonomous --path . 2>&1 || {
        echo "WARNING: grace lint found issues. Continuing anyway..."
    }
else
    echo "SKIP: grace CLI not installed (pip install grace-cli?)"
fi

echo ""
echo "=== Level 1: ruff check ==="
ruff check src/ 2>&1 || true

# ── Audit ──
mkdir -p "$SESSION_DIR"
echo ""
echo "=== Level 2: Audit ==="
python3 scripts/audit_grace_coverage.py -o "$SESSION_DIR/manifest.json"
echo ""

CRITICAL=$(python3 -c "
import json
m = json.load(open('$SESSION_DIR/manifest.json'))
print(m['summary']['critical_issues'])
")

# ── Summary ──
python3 scripts/audit_grace_coverage.py --summary

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN COMPLETE ==="
    echo "Manifest: $SESSION_DIR/manifest.json"
    echo "No changes made. Run without --dry-run to fix."
    exit 0
fi

if [[ "$CRITICAL" == "0" ]] && [[ -z "$LAYER_FILTER" ]]; then
    echo "No critical issues. Skipping Level 2 deep sync."
    rm -rf "$SESSION_DIR"
    exit 0
fi

# ── Level 2: Deep Sync ──
echo ""
echo "=== Level 2: Deep Refresh ==="
echo "Session: $TIMESTAMP"

ORCHESTRATOR_PROMPT=".opencode/agents/grace-orchestrator.md"
LAYER_ARG=""
if [[ -n "$LAYER_FILTER" ]]; then
    LAYER_ARG="Only process layer: $LAYER_FILTER. Skip all other layers."
    echo "Layer filter: $LAYER_FILTER"
fi

FULL_PROMPT="$(cat "$ORCHESTRATOR_PROMPT")

## Session Configuration
- TIMESTAMP: $TIMESTAMP
- SESSION_DIR: $SESSION_DIR
- MANIFEST: $SESSION_DIR/manifest.json
- $LAYER_ARG

Proceed with Step 1 (health check) now. The manifest is already generated at $SESSION_DIR/manifest.json."

if [[ "$AUTO_MODE" == "true" ]]; then
    echo "Running non-interactive..."
    opencode -p "$FULL_PROMPT" --approve-all
else
    echo ""
    echo "=== Starting opencode with GRACE Orchestrator ==="
    echo "Session dir: $SESSION_DIR"
    echo "Prompt file: $ORCHESTRATOR_PROMPT"
    echo ""
    echo "Run manually:"
    echo "  opencode"
    echo "  Then paste the prompt from: $SESSION_DIR/prompt.txt"
    echo ""
    echo "$FULL_PROMPT" > "$SESSION_DIR/prompt.txt"
    echo "Prompt saved to $SESSION_DIR/prompt.txt"
fi
