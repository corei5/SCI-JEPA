#!/bin/bash
# SCI-JEPA: the 60k BENCH run (model size S) on Grete, with the live dashboard. Run from a login-node terminal:
#
#   bash ~/Alex/SCI-JEPA/run_60k.sh all       # the whole experiment: seed 0 (~3 days), then seeds 1-2 start by
#                                             #   themselves (~5 more days); opens the dashboard tunnel
#   bash ~/Alex/SCI-JEPA/run_60k.sh phase1    # only seed 0 of every system (~3 days)
#   bash ~/Alex/SCI-JEPA/run_60k.sh phase2    # seeds 1 and 2 by hand (seed 0 is loaded, not retrained)
#   bash ~/Alex/SCI-JEPA/run_60k.sh watch     # (re)open the dashboard tunnel, e.g. after the laptop slept
#   bash ~/Alex/SCI-JEPA/run_60k.sh status    # jobs, progress of every run, where the results are
#   bash ~/Alex/SCI-JEPA/run_60k.sh stop      # cancel the chain (finished runs and checkpoints are kept)
#
# The jobs run under Slurm (12-hour jobs that continue each other from checkpoints), so the laptop can sleep.
# The dashboard tunnel runs in tmux on THIS login node; open http://localhost:8765/ (VS Code forwards the port).
# Settings and the expected timeline: the technical report, section "Training" (T1-T7), and scaling/README.md.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}" || exit 1
source cluster/grete/site.env

NAME=arxiv60k_S
OUT="${STORE}/SCI-JEPA/${NAME}"                          # results, PDF, figures, dashboard, infrastructure
SPLIT_DIR="${STORE}/Scientific-Summaries/splits/arxiv_60k"
COMMON=(PAPER_A_OUT="${OUT}" SPLIT="${SPLIT_DIR}" N_PAPERS=0 SIZE=S EXPS=BENCH EPOCHS=10 PATIENCE=0
        LLM_QUERIES=300 CKPT_EVERY=2000)
CHAINS="${OUT}/chains.txt"                                # every chain file of this run, newest last

latest_chain() { tail -1 "${CHAINS}" 2>/dev/null; }
chain_running() {                                        # job ids of this run's chains still in the queue
  local c ids=""
  [[ -f "${CHAINS}" ]] || return 0
  while read -r c; do [[ -f "${c}" ]] && ids+=" $(cat "${c}")"; done < "${CHAINS}"
  [[ -n "${ids// /}" ]] && squeue -h -u "${USER}" -j "$(tr ' ' ',' <<< "${ids# }")" -o %i 2>/dev/null
}

start() {   # $1 = number of 12-hour jobs, $2 = seeds, $3... = extra KEY=VALUE (e.g. the next phase)
  [[ -d "${SPLIT_DIR}/train" ]] || { echo "split not found: ${SPLIT_DIR}"; exit 1; }
  if [[ -n "$(chain_running)" ]]; then
    echo "a chain of this run is already in the queue: $(chain_running | tr '\n' ' ')"
    echo "watch it: bash ${REPO}/run_60k.sh watch   ·   stop it: bash ${REPO}/run_60k.sh stop"; exit 1
  fi
  mkdir -p "${OUT}" "${LOGS}"
  bash cluster/grete/submit_chain.sh "$1" -- "${COMMON[@]}" "SEEDS=$2" "${@:3}" || exit 1
  ls -t "${LOGS}"/chain_*.txt | head -1 >> "${CHAINS}"
  echo
  watch
}

watch() {
  local c; c="$(latest_chain)"
  [[ -f "${c}" ]] || { echo "no chain started yet: bash ${REPO}/run_60k.sh phase1"; exit 1; }
  tmux kill-session -t sci-jepa-dashboard 2>/dev/null    # a fresh tunnel for the newest chain
  FOLLOW_CHAINS="${CHAINS}" bash cluster/grete/run_paper_a.sh watch "${c}"   # follows the next phase too
}

status() {
  echo "== jobs";    squeue -u "${USER}" -o "%.10i %.10j %.9T %.10M %.10l %R"
  echo "== runs finished: $(ls "${OUT}"/BENCH/*_seed*.json 2>/dev/null | wc -l) of 18 per seed"
  ls "${OUT}"/BENCH/*_seed*.json 2>/dev/null | xargs -r -n1 basename | sed 's/\.json$//' | paste -sd' ' | fold -s -w 110
  echo "== runs in progress"
  for f in "${OUT}"/infra/progress/*.json; do
    [[ -f "${f}" ]] && "${VENV}/bin/python" -c "
import json,sys; p=json.load(open(sys.argv[1]))
if p.get('status') != 'done':
    eta = p.get('eta_max_s'); print(f\"  {p.get('name', sys.argv[1])[:46]:46s} {p.get('status','')[:9]:9s} epoch {p.get('epoch',0):.2f}/{p.get('max_epochs','?')}\"
          f\"  {p.get('steps_per_s') or 0:.2f} steps/s\" + (f'  ~{eta/3600:.1f} h left' if eta else ''))" "${f}" 2>/dev/null
  done
  echo "== latest log: $(ls -t "${LOGS}"/paper_a_*.out 2>/dev/null | head -1)"
  echo "== results: ${OUT}   models: ${STORE}/SCI-JEPA/models/${NAME}"
  tmux has-session -t sci-jepa-dashboard 2>/dev/null \
    && echo "== dashboard tunnel: running on $(hostname -s) -> http://localhost:8765/" \
    || echo "== dashboard tunnel: not running here ($(hostname -s)); start it: bash ${REPO}/run_60k.sh watch"
}

stop() {
  local ids; ids="$(chain_running)"
  [[ -n "${ids}" ]] && { scancel ${ids}; echo "cancelled: $(tr '\n' ' ' <<< "${ids}")"; } || echo "no job of this run in the queue"
  tmux kill-session -t sci-jepa-dashboard 2>/dev/null && echo "dashboard tunnel closed"
  echo "finished runs and checkpoints are kept in ${OUT}; phase1 / phase2 continues where it stopped"
}

case "${1:-}" in
  all)    rm -f "${OUT}/next_phase_started"; start 8 "0" "NEXT_SEEDS=0 1 2" "NEXT_CHAIN_LEN=14" ;;
  phase1) start 8 "0" ;;
  phase2) start 14 "0 1 2" ;;
  watch)  watch ;;
  status) status ;;
  stop)   stop ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
