#!/bin/bash
# GPU checks after the scaling changes: relation-bias gradient, operator trace, speed profile with batch workers.
#   sbatch -A kisski-l3s-llm4sci -p grete:interactive --gres=gpu:3g.40gb:1 --cpus-per-task=16 --mem=96G \
#          --time=01:30:00 -o scaling/logs/after_checks_%j.out scaling/after_checks.sh      (from ~/Alex/SCI-JEPA)
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR}"; cd "${REPO}" || exit 1
source cluster/grete/site.env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
SPLIT="${STORE}/Scientific-Summaries/splits/arxiv_60k"
echo "== 1. relation-bias gradient"; "${VENV}/bin/python" scaling/check_relbias.py
echo "== 2. operator trace"
for s in fast gpu; do
  "${VENV}/bin/python" scaling/trace_step.py --split "${SPLIT}" --index scaling/runs/profile_before/index \
    --out scaling/runs/trace_after --arm reference --speed "${s}" 2>&1 | grep -E "^step:|embedding_dense|Self CUDA time"
done
echo "== 3. speed profile (index of the before profile: the index code did not change)"
export PAPER_A_INDEX_DIR="${REPO}/scaling/runs/profile_after/index"
"${VENV}/bin/python" scaling/profile_training.py --split "${SPLIT}" --n-papers 2000 --out scaling/runs/profile_after \
  --workers 0,3,7,15
echo "done rc=$?"
