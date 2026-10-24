#!/usr/bin/env bash
# Rebuild the Graphify code graph and every view built from it.
#   bash scripts/graph-refresh.sh          # incremental (AST only, no LLM)
#   bash scripts/graph-refresh.sh --relabel  # also rename all communities via the claude CLI
# The git hooks (`graphify hook status`) already run the incremental update after
# each commit/checkout/merge; run this when you also want fresh names and views.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
cd "${ROOT}"

graphify update .
if [[ "${1:-}" == "--relabel" ]]; then
  graphify label . --backend=claude-cli
else
  # Name only communities that are new since the last labeling run.
  graphify label . --missing-only --backend=claude-cli || echo "labeling skipped (claude CLI unavailable)" >&2
fi
python3 scripts/graph-views.py
