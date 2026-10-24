#!/bin/bash
# Present a finished (or running) SCI-JEPA dashboard on localhost. Needs only python3: no GPU, no
# sci_jepa install, no internet (the JS libraries are in the folder's vendor/).
#
#   bash scripts/present_dashboard.sh                       # ./runs/paper_a_results/dashboard
#   bash scripts/present_dashboard.sh /path/to/dashboard    # any copied dashboard folder
#   PORT=9000 bash scripts/present_dashboard.sh ...         # first port to try (default 8765)
#
# The folder is the results' dashboard/ (index.html, runs.json, runs/, graphs/, vendor/). Copy it to a
# laptop with rsync to present offline (docs/cluster_grete.md). Ctrl-C stops the server.
set -uo pipefail

DIR="${1:-runs/paper_a_results/dashboard}"
[[ -f "${DIR}/index.html" && -f "${DIR}/runs.json" ]] || {
  echo "not a dashboard folder: ${DIR} (needs index.html and runs.json)"; exit 1; }
PY="$(command -v python3 || command -v python)" || { echo "python3 not found"; exit 1; }

# first free port from PORT (login nodes are shared: someone may hold 8765)
PORT="$("${PY}" - "${PORT:-8765}" <<'EOF'
import socket, sys
p = int(sys.argv[1])
for port in range(p, p + 50):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port)); print(port); break
        except OSError:
            pass
EOF
)"
[[ -n "${PORT}" ]] || { echo "no free port"; exit 1; }

RUNS="$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${DIR}/runs.json" 2>/dev/null || echo "?")"
URL="http://localhost:${PORT}/"
echo "================================================================"
echo " SCI-JEPA dashboard: ${URL}"
echo " folder : $(cd "${DIR}" && pwd)  (${RUNS} runs)"
echo " on a cluster through VS Code: the Ports tab forwards ${PORT} and opens it"
echo " Ctrl-C to stop"
echo "================================================================"
case "$(uname -s)" in Darwin) (sleep 1; open "${URL}") & ;; esac      # macOS: open the browser
cd "${DIR}" && exec "${PY}" -m http.server "${PORT}" --bind 127.0.0.1
