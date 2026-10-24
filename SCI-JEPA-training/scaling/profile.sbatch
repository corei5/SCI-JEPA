#!/bin/bash
# Speed profile of SCI-JEPA training on real data (scaling/profile_training.py), on one A100 slice.
#   sbatch scaling/profile.sbatch [TAG] [N_PAPERS]        (run from ~/Alex/SCI-JEPA)
# Writes scaling/logs/profile_<TAG>_<job>.out and scaling/runs/profile_<TAG>/ (index cache + profile.json).
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --partition=grete:interactive
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=sj_profile
#SBATCH --output=scaling/logs/profile_%j.out
set -uo pipefail
TAG="${1:-before}"; N="${2:-2000}"; WORKERS="${3:-}"   # e.g. 0,3,7,15: also time the batch pipeline
REPO="${SLURM_SUBMIT_DIR}"
source "${REPO}/cluster/grete/site.env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO}" || exit 1
OUT="${REPO}/scaling/runs/profile_${TAG}"
mkdir -p "${OUT}"
export PAPER_A_INDEX_DIR="${OUT}/index" PAPER_A_OUT="${OUT}" PAPER_A_MODEL_DIR="${OUT}/models"   # everything stays in scaling/
# TAG=before: the code as it was before the scaling changes (scaling/code_before), else the current src/
[[ "${TAG}" == before* ]] && export PYTHONPATH="${REPO}/scaling/code_before"
echo "code: $("${VENV}/bin/python" -c 'import sci_jepa, os; print(os.path.dirname(sci_jepa.__file__))')"
echo "profile ${TAG} | ${N} papers | job ${SLURM_JOB_ID} on $(hostname) | $(date)"
nvidia-smi -L
"${VENV}/bin/python" scaling/profile_training.py --split "${STORE}/Scientific-Summaries/splits/arxiv_60k" \
  --n-papers "${N}" --out "${OUT}" ${WORKERS:+--workers "${WORKERS}"}
RC=$?
cp "scaling/logs/profile_${SLURM_JOB_ID}.out" "${OUT}/profile.log" 2>/dev/null
echo "done rc=${RC}"
