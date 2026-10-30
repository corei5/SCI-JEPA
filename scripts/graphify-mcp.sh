#!/usr/bin/env bash
# Cursor MCP entrypoint. Serves the local Graphify graph over stdio.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.local/bin:/opt/anaconda3/bin:${PATH}"
GRAPH="${ROOT}/graphify-out/graph.json"
if [[ ! -f "${GRAPH}" ]]; then
  echo "Missing ${GRAPH}. From the repo root run: graphify extract . --code-only" >&2
  exit 1
fi
exec graphify-mcp "${GRAPH}"
