#!/bin/bash
#SBATCH --job-name=paper_a
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --partition=grete:interactive
#SBATCH --gres=gpu:3g.40gb:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/logs/paper_a_%j.out
#
# SCI-JEPA Paper A as a plain batch job: submit, log out, close the laptop. From the repository root:
#
#   sbatch cluster/grete/paper_a.sbatch                                   # notebook defaults, 2 GPUs (dist)
#   sbatch cluster/grete/paper_a.sbatch N_PAPERS=5000 SIZE=S SEEDS="0 1 2 3 4" EXPS="BENCH E10"
#   sbatch --gres=gpu:1g.10gb:1 --time=01:00:00 cluster/grete/paper_a.sbatch MODE=smoke
#   sbatch --partition=kisski --gres=gpu:A100:4 --cpus-per-task=64 --time=2-00:00:00 cluster/grete/paper_a.sbatch
#
# Parameters go AFTER the script name as KEY=VALUE. (Grete sets SBATCH_EXPORT=none, so variables set in
# front of `sbatch` never reach the job.) Keys: MODE = dist (default) | run | smoke | check, and the notebook
# parameters N_PAPERS SIZE EPOCHS PATIENCE SEEDS EXPS SPEED LLM_MODEL LLM_QUERIES FIELD_BALANCE FIELD_ALPHA
# DATASET, plus PAPER_A_OUT (results folder) and KEEP_SERVING (minutes the dashboard stays up after the run),
# SPLIT (a stored split folder), LLM_MODELS (several LLM-Reasoning arms, comma-separated Hugging Face ids),
# RESUME=1 (checkpoint every CKPT_EVERY steps; a later job continues), SKIP_SINGLE=1. Chains of 12 h jobs that
# continue each other: cluster/grete/submit_chain.sh (sets CHAIN_FILE / CHAIN_LEFT / CHAIN_SBATCH).
# sbatch options before the script name override the #SBATCH lines above.
# Python: the conda env set in site.env (VENV, "Alex"); nothing needs to be activated.
# Dashboard later, from a login-node terminal:  bash cluster/grete/run_paper_a.sh tunnel <jobid>
set -uo pipefail
for arg in "$@"; do
  case "${arg}" in
    [A-Z_]*=*) export "${arg}"; echo "parameter: ${arg}" ;;
    *) echo "ignored argument (expected KEY=VALUE): ${arg}" ;;
  esac
done

export REPO="${SLURM_SUBMIT_DIR}"                  # the repository root you submitted from
cd "${REPO}" || exit 1
[[ -f cluster/grete/run_paper_a.sh ]] || { echo "submit from the SCI-JEPA repository root"; exit 1; }

# a chain (submit_chain.sh): submit the successor now; it waits until this job ends and resumes the runs
if [[ "${CHAIN_LEFT:-1}" -gt 1 ]]; then
  NEXT_ARGS=()
  for arg in "$@"; do
    case "${arg}" in CHAIN_LEFT=*) NEXT_ARGS+=("CHAIN_LEFT=$(( CHAIN_LEFT - 1 ))") ;; *) NEXT_ARGS+=("${arg}") ;; esac
  done
  # shellcheck disable=SC2086  (CHAIN_SBATCH holds several sbatch options)
  if NEXT=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" ${CHAIN_SBATCH:-} cluster/grete/paper_a.sbatch "${NEXT_ARGS[@]}"); then
    echo "chain: next job ${NEXT} (${CHAIN_LEFT} jobs left including this one)"
    [[ -n "${CHAIN_FILE:-}" ]] && echo "${NEXT}" >> "${CHAIN_FILE}"
  else
    echo "chain: could not submit the next job; continue by hand: bash cluster/grete/submit_chain.sh ..."
  fi
fi

# run a private copy: editing the repository's scripts while this job runs cannot break it
COPY="${TMPDIR:-/tmp}/run_paper_a_${SLURM_JOB_ID}.sh"
cp cluster/grete/run_paper_a.sh "${COPY}"
bash "${COPY}" "${MODE:-dist}"
RC=$?
rm -f "${COPY}"
exit ${RC}
