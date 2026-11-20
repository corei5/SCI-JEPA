#!/bin/bash
# Science-specific safety checks (KU faithfulness with Bespoke-MiniCheck-7B, dual-use scan with WMDP, retractions,
# and fabrication / novelty / popularity / overclaiming / hazard questions for every system on one shared question
# set), then the safety report and the dashboard. Resumes: systems already scored are kept.
#   sbatch cluster/grete/science_safety.sbatch
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --partition=grete:preemptible
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --job-name=science_safety
#SBATCH --output=scaling/logs/science_safety_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}" || exit 1
source cluster/grete/site.env
# .pylib: einops + sentencepiece for the MiniCheck judge's own model code (kept out of the shared environment)
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}/.pylib" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT="${OUT:-${STORE}/SCI-JEPA/arxiv60k_S}"
echo "science safety | job ${SLURM_JOB_ID} on $(hostname) | $(date)"; nvidia-smi -L
"${VENV}/bin/python" -m sci_jepa.evaluation.science_safety "${OUT}" --n "${N:-300}" --n-llm "${N_LLM:-150}"
"${VENV}/bin/python" -m sci_jepa.evaluation.safety "${OUT}"
"${VENV}/bin/python" -m sci_jepa.evaluation.compliance "${OUT}"      # regulation assessment uses the new results
cp src/sci_jepa/dashboard/dashboard.html "${OUT}/dashboard/index.html"
echo "done rc=$? $(date)"
