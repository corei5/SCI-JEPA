#!/bin/bash
# Paired benchmark, protocol v2 (evaluation audit fixes): every saved model and comparison system re-scored on one
# shared question set, then the summary for the dashboard. No retraining. Resumes: finished runs are kept, so a
# rerun after the 60 k run's last seed only scores the new models.
#   sbatch cluster/grete/paired_eval.sbatch
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --partition=grete:preemptible
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --job-name=paired_eval
#SBATCH --output=scaling/logs/paired_eval_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}" || exit 1
source cluster/grete/site.env
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT="${OUT:-${STORE}/SCI-JEPA/arxiv60k_S}"
echo "paired eval v2 | job ${SLURM_JOB_ID} on $(hostname) | $(date)"; nvidia-smi -L
"${VENV}/bin/python" -m sci_jepa.evaluation.paired_eval "${OUT}" --n "${N:-300}" --n-llm "${N_LLM:-150}" --pairs "${PAIRS:-3000}"
"${VENV}/bin/python" -m sci_jepa.evaluation.compliance "${OUT}"      # regulation assessment uses the new results
cp src/sci_jepa/dashboard/dashboard.html "${OUT}/dashboard/index.html"
echo "done rc=$? $(date)"
