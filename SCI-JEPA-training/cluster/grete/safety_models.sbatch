#!/bin/bash
# Safety model checks (calibration, selective prediction, paraphrase, context injection) for every saved JEPA model
# of a finished experiment, plus the result-file checks. Read-only on the models; writes <exp>/safety.json,
# safety.md and dashboard/safety.json.
#   sbatch cluster/grete/safety_models.sbatch                       (all models, 300 queries each)
#   sbatch --export=ALL,ONLY=jepa+rgat_seed0,N=24 cluster/grete/safety_models.sbatch    (quick test, no write)
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --partition=grete:interactive
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=safety
#SBATCH --output=scaling/logs/safety_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}" || exit 1
source cluster/grete/site.env
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT="${OUT:-${STORE}/SCI-JEPA/arxiv60k_S}"
echo "safety | job ${SLURM_JOB_ID} on $(hostname) | $(date)"; nvidia-smi -L
if [[ -n "${ONLY:-}" ]]; then
  "${VENV}/bin/python" - "${OUT}" "${ONLY}" "${N:-24}" <<'PY'
import json, sys
from sci_jepa.evaluation import safety as S
m = json.load(open(f"{sys.argv[1]}/BENCH/{sys.argv[2]}.json"))
r = S.check_model(m["model_dir"], int(sys.argv[3]))
print(json.dumps({k: v for k, v in r.items() if k != "calibration"} | {"ece": (r.get("calibration") or {}).get("ece")}, indent=1, default=str))
PY
else
  "${VENV}/bin/python" -m sci_jepa.evaluation.safety "${OUT}" --models --n "${N:-300}"
fi
echo "done rc=$? $(date)"
