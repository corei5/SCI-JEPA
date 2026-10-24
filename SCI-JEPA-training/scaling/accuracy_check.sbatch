#!/bin/bash
# Before / after accuracy check of the scaling changes (scaling/accuracy_check.py), on one A100 slice.
#   sbatch scaling/accuracy_check.sbatch before|after [extra accuracy_check.py options]   (from ~/Alex/SCI-JEPA)
# Writes scaling/logs/accuracy_<job>.out and scaling/runs/accuracy_<variant>/ (index, results, models, check.json).
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --partition=grete:interactive
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=sj_accuracy
#SBATCH --output=scaling/logs/accuracy_%j.out
set -uo pipefail
VARIANT="${1:?before or after}"; shift
REPO="${SLURM_SUBMIT_DIR}"
source "${REPO}/cluster/grete/site.env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO}" || exit 1
OUT="${REPO}/scaling/runs/accuracy_${VARIANT}"
mkdir -p "${OUT}"
export PAPER_A_INDEX_DIR="${OUT}/index" PAPER_A_OUT="${OUT}" PAPER_A_MODEL_DIR="${OUT}/models"   # everything stays in scaling/
[[ "${VARIANT}" == before ]] && export PYTHONPATH="${REPO}/scaling/code_before"   # the code before the changes
echo "accuracy check ${VARIANT} | job ${SLURM_JOB_ID} on $(hostname) | $(date)"
nvidia-smi -L
"${VENV}/bin/python" scaling/accuracy_check.py --variant "${VARIANT}" \
  --split "${STORE}/Scientific-Summaries/splits/arxiv_60k" --out "${OUT}" --workers $(( SLURM_CPUS_PER_TASK - 1 )) "$@"
RC=$?
cp "scaling/logs/accuracy_${SLURM_JOB_ID}.out" "${OUT}/accuracy.log" 2>/dev/null
echo "done rc=${RC}"
