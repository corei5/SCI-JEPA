#!/usr/bin/env bash
# Workstation launcher. Run from anywhere:
#   bash Analysis/run_local.sh papers device 0
#   bash Analysis/run_local.sh dashboard
set -euo pipefail

ANALYSIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${ANALYSIS}/.." && pwd)"
export PATH="${HOME}/.local/bin:/opt/anaconda3/bin:${PATH}"
export SCIJEPA_DATA_ROOT="${SCIJEPA_DATA_ROOT:-${ANALYSIS}/dataset}"
export PYTHONUNBUFFERED=1

cmd="${1:-papers}"
shift || true

if [[ "${cmd}" == "dashboard" ]]; then
  exec uv run --project "${ROOT}" graph-jepa dashboard "$@"
fi

cd "${ANALYSIS}"
exec uv run --project "${ROOT}" python -m "train.${cmd}" "$@"
