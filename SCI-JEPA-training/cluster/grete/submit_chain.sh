#!/bin/bash
# Long experiments in jobs of at most 12 h (grete:interactive): a CHAIN of up to N jobs of paper_a.sbatch. Every
# job continues each unfinished run from its last checkpoint (RESUME=1: every 2,000 steps and at every epoch end).
# grete:interactive allows only 2 jobs per user, queued ones included, so the chain grows one job at a time: when a
# job starts it submits its successor (which waits until it ends). The job that finishes the last run writes the
# reports and cancels its successor. Run on the login node, from the repository root:
#
#   bash cluster/grete/submit_chain.sh 8 -- PAPER_A_OUT=/path/results_d60k SPLIT=/path/splits/d60k N_PAPERS=0 EXPS=BENCH
#   bash cluster/grete/submit_chain.sh 4 --gres=gpu:3g.40gb:2 --time=12:00:00 -- PAPER_A_OUT=... KEY=VALUE ...
#
# Arguments: N (most jobs), then sbatch options (optional, used for every job), then -- and the KEY=VALUE
# parameters of paper_a.sbatch. Added unless given: RESUME=1, KEEP_SERVING=0 (no idle dashboard time between jobs),
# SKIP_SINGLE=1 (the notebook's extra reference run duplicates BENCH's "reference" arm). PAPER_A_OUT is required:
# a fresh results folder per experiment, so finished runs of other experiments are never mixed in.
# Watch: squeue -u $USER   ·   dashboard: bash cluster/grete/run_paper_a.sh tunnel <running job id>
# Stop the chain: scancel $(cat <chain file>)
set -uo pipefail

N="${1:?usage: submit_chain.sh N [sbatch options] -- KEY=VALUE ...}"; shift
SB=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SB+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
PARAMS=("$@")
has() { local k; for k in "${PARAMS[@]}"; do [[ "${k%%=*}" == "$1" ]] && return 0; done; return 1; }
has PAPER_A_OUT || { echo "give PAPER_A_OUT=<results folder> (a fresh folder for this experiment)"; exit 1; }
has RESUME || PARAMS+=("RESUME=1")
has KEEP_SERVING || PARAMS+=("KEEP_SERVING=0")
has SKIP_SINGLE || PARAMS+=("SKIP_SINGLE=1")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}" || exit 1
source cluster/grete/site.env
CHAIN="${LOGS}/chain_$(date +%Y%m%d_%H%M%S).txt"
: > "${CHAIN}"
PARAMS+=("CHAIN_FILE=${CHAIN}" "CHAIN_LEFT=${N}" "CHAIN_SBATCH=${SB[*]}")

jid=$(sbatch --parsable "${SB[@]}" cluster/grete/paper_a.sbatch "${PARAMS[@]}") || { echo "sbatch failed"; exit 1; }
echo "${jid}" >> "${CHAIN}"
echo "chain started: job ${jid} (up to ${N} jobs; each submits the next when it starts)"
echo "chain file: ${CHAIN}   ·   logs: ${LOGS}/paper_a_<job>.out   ·   stop: scancel \$(cat ${CHAIN})"
echo "dashboard through the whole chain (survives a sleeping laptop; open http://localhost:8765/):"
echo "  bash cluster/grete/run_paper_a.sh watch ${CHAIN}"
