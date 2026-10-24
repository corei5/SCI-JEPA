#!/bin/bash
# SCI-JEPA Paper A on one machine (laptop, workstation; Apple GPU, CUDA or CPU). From the repository root:
#
#   bash scripts/run_local.sh setup            # once: .venv, packages, Jupyter kernel "Python (sci-jepa)", text encoder
#   bash scripts/run_local.sh smoke            # synthetic data (add --full: every arm)
#   bash scripts/run_local.sh                  # 5,000 papers: build KU graph, shape audit, BENCH (3 seeds)
#   N_PAPERS=20000 EXPS="E10 E11" EPOCHS=10 bash scripts/run_local.sh
#   SIZE=S DEVICE=cpu bash scripts/run_local.sh
#   SHARD=0/2 EXPS=BENCH bash scripts/run_local.sh     # two machines share one benchmark (same PAPER_A_OUT),
#   SHARD=1/2 EXPS=BENCH bash scripts/run_local.sh     #   then once without SHARD to write the summary
#
# The notebook (notebooks/paper_a_local.ipynb) runs the same pipeline with a live dashboard. For a cluster
# see cluster/grete/. Every knob below can be set from the environment; EXTRA passes any other config field,
# e.g. EXTRA="split_mode=random lambda_inv=0.3".
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-run}"                      # setup | smoke | build | audit | run
N_PAPERS="${N_PAPERS:-5000}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-100}"
SIZE="${SIZE:-local}"                # d=256, 4 encoder layers: ~0.4 s/step on an Apple GPU at N=8
EPOCHS="${EPOCHS:-40}"               # maximum; early stopping on the validation reasoning index
BATCH="${BATCH:-64}"
SEEDS="${SEEDS:-[0,1,2]}"
EXPS="${EXPS:-BENCH}"
EVAL_N="${EVAL_N:-1000}"
DEVICE="${DEVICE:-auto}"             # auto: CUDA, else Apple GPU (mps), else cpu
EXTRA="${EXTRA:-}"
# SPEED: fast = same model, fp16 + background batch building + 8x faster text embedding (default)
#        fast+multi = + multi-target training (loses accuracy, not recommended) | off = the original code path
SPEED="${SPEED:-fast}"
case "${SPEED}" in
  fast)       SPEED_SET="amp=fp16 prefetch=2 embed_bucket=64 embed_dtype=float16" ;;
  fast+multi) SPEED_SET="amp=fp16 prefetch=2 embed_bucket=64 embed_dtype=float16 targets_per_instance=4" ;;
  *)          SPEED_SET="" ;;
esac
LLM_MODEL="${LLM_MODEL:-}"           # LLM-Reasoning model (default in config: Qwen/Qwen2.5-7B-Instruct)
LLM_SET="${LLM_MODEL:+llm_model=${LLM_MODEL}}"
SHARD="${SHARD:-}"                   # i/n: this machine runs every n-th arm x seed (shared PAPER_A_OUT)

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PAPER_A_PARQUET="${PAPER_A_PARQUET:-${REPO}/data/normalized/arxiv}"
export PAPER_A_INDEX_DIR="${PAPER_A_INDEX_DIR:-${REPO}/runs/paper_a_index}"
export PAPER_A_OUT="${PAPER_A_OUT:-${REPO}/runs/paper_a_results}"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # any op MPS lacks runs on CPU instead of failing
mkdir -p "${PAPER_A_INDEX_DIR}" "${PAPER_A_OUT}"
cd "${REPO}" || exit 1

# ---------------------------------------------------------------- environment
VENV="${VENV:-${REPO}/.venv}"
if [[ "${CMD}" == "setup" || ! -x "${VENV}/bin/python" ]]; then
  echo "== creating ${VENV} =="
  "${PYTHON:-python3}" -m venv "${VENV}" || exit 1
  "${VENV}/bin/pip" install -q --upgrade pip
  "${VENV}/bin/pip" install -q -e ".[notebook,llm]" || { echo "pip install failed"; exit 1; }
  "${VENV}/bin/python" -m ipykernel install --user --name sci-jepa --display-name "Python (sci-jepa)"
  echo "== downloading the text encoder once (gte-modernbert-base, 8192-token context, ~600 MB) =="
  "${VENV}/bin/python" -c "from sentence_transformers import SentenceTransformer as S; S('Alibaba-NLP/gte-modernbert-base')"
  "${VENV}/bin/sci-jepa" fetch-assets
  [[ "${CMD}" == "setup" ]] && { echo "setup done. Open notebooks/paper_a_local.ipynb with kernel 'Python (sci-jepa)'."; exit 0; }
fi
PY="${VENV}/bin/python"

if [[ ! -d "${PAPER_A_PARQUET}" && "${CMD}" != "smoke" ]]; then
  echo "ERROR: parquet folder not found: ${PAPER_A_PARQUET}"
  echo "       prepare the data (docs/data.md) or set PAPER_A_PARQUET=/path/to/normalized/arxiv"
  exit 1
fi

DATA="max_papers=${N_PAPERS} rows_per_shard=${ROWS_PER_SHARD} device=${DEVICE} ${SPEED_SET} ${LLM_SET}"
TRAIN="epochs=${EPOCHS} batch_size=${BATCH} seeds=${SEEDS} eval_instances=${EVAL_N}"

echo "================================================================"
echo " SCI-JEPA Paper A | cmd=${CMD} | $(date)"
echo " data   : ${PAPER_A_PARQUET}"
echo " papers : ${N_PAPERS} (${ROWS_PER_SHARD} rows/shard) | size ${SIZE} | device ${DEVICE}"
echo " train  : ${EPOCHS} epochs, batch ${BATCH}, seeds ${SEEDS} | exps: ${EXPS}"
echo " extra  : ${EXTRA:-<none>}"
echo " output : ${PAPER_A_OUT}"
echo "================================================================"

RC=0
case "${CMD}" in
  smoke)
    "${PY}" -m sci_jepa.smoke_test "${@:2}" || RC=$? ;;
  build)
    "${PY}" -m sci_jepa.data.ku_graph --set ${DATA} ${EXTRA} || RC=$? ;;
  audit)
    "${PY}" -m sci_jepa.training.train --audit-shapes --size "${SIZE}" --set ${DATA} ${EXTRA} || RC=$? ;;
  run)
    # 1. KU graph for this data setting (cached; rebuilt only when data settings change)
    "${PY}" -m sci_jepa.data.ku_graph --set ${DATA} ${EXTRA} || { echo "KU graph build failed"; exit 1; }
    # 2. Theorem A2 audit of the shape vocabulary (no training)
    "${PY}" -m sci_jepa.training.train --audit-shapes --size "${SIZE}" --set ${DATA} ${EXTRA} || RC=$?
    # 3. experiments
    for E in ${EXPS}; do
      echo "################ ${E} ################"
      "${PY}" -m sci_jepa.training.train --exp "${E}" --size "${SIZE}" ${SHARD:+--shard ${SHARD}} --set ${DATA} ${TRAIN} ${EXTRA} || RC=$?
    done ;;
  *)
    echo "unknown command '${CMD}' (setup | smoke | build | audit | run)"; exit 2 ;;
esac
echo "done rc=${RC}; results under ${PAPER_A_OUT}/<exp>/summary.json"
exit ${RC}
