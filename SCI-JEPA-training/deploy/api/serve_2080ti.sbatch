#!/bin/bash
# Serve the Sci-JEPA API (FastAPI, no authentication) on one RTX 2080 Ti.
#
#   MODEL=/path/to/models/<exp>/<arm>_seed0 sbatch deploy/api/serve_2080ti.sbatch
#   MODEL=... INDEX=/path/paper_a_ku_index_<fp>.pt PORT=8000 sbatch deploy/api/serve_2080ti.sbatch
#
# Then, from your machine:  ssh -N -L 8000:<node>:8000 <login host>   and open http://127.0.0.1:8000/v1/docs
# (the node name is printed at the top of the job log).
#
# --gres=gpu:t2080ti:1 asks for a 2080 Ti explicitly (these nodes also hold A5000s).
# 11 GB of GPU memory is plenty: the model (~230M parameters, fp32) and the text encoder use ~2-3 GB. The paper
# vectors stay in CPU memory (~6 GB at 2.07M papers, twice that with corpus context on) next to the index blob
# (~10 GB on disk, more as Python objects), hence --mem.
#SBATCH --job-name=scijepa-api
#SBATCH --partition=p_12G
#SBATCH --gres=gpu:t2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=log/scijepa-api-%j.out
set -euo pipefail

: "${MODEL:?set MODEL=<trained model folder with config.json and model.safetensors>}"
INDEX="${INDEX:-auto}"                 # auto: the index the model was trained on, under $PAPER_A_INDEX_DIR
PORT="${PORT:-8000}"
WEIGHTS="${WEIGHTS:-model}"            # model (end of training) or best (best validation epoch)
EXTRA="${EXTRA:-}"                     # e.g. "--full-texts" (needs much more memory) or "--no-neighbours"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PA="${REPO%/}"; [ -d "${PA}/src/sci_jepa" ] || PA="${PA}/SCI-JEPA-training"
PY="${PY:-${PA}/.venv/bin/python}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "node $(hostname)  port ${PORT}  gpu ${CUDA_VISIBLE_DEVICES:-?}"
echo "tunnel: ssh -N -L ${PORT}:$(hostname):${PORT} <login host>   then http://127.0.0.1:${PORT}/v1/docs"
"${PY}" -c "import fastapi, uvicorn" 2>/dev/null || {
  echo "fastapi / uvicorn missing: cd ${PA} && uv pip install --python .venv/bin/python -e '.[api]'"; exit 1; }

# Supervisor loop: restart the server if it crashes or exits after a CUDA fault (--exit-on-cuda-fault, code 3).
# A clean stop (code 0, or SIGTERM from scancel / the time limit) ends the job. At most MAX_RESTARTS restarts.
MAX_RESTARTS="${MAX_RESTARTS:-5}"
child=0
trap 'kill -TERM "${child}" 2>/dev/null; wait "${child}"; exit 0' TERM INT
for attempt in $(seq 0 "${MAX_RESTARTS}"); do
  "${PY}" -m sci_jepa.api --model "${MODEL}" --weights "${WEIGHTS}" --index "${INDEX}" \
       --device cuda --host 0.0.0.0 --port "${PORT}" --exit-on-cuda-fault ${EXTRA} &
  child=$!
  set +e; wait "${child}"; rc=$?; set -e
  [ "${rc}" -eq 0 ] && exit 0
  echo "$(date -Is) server exited with code ${rc} (attempt ${attempt}/${MAX_RESTARTS}); restarting in 10 s"
  sleep 10
done
echo "giving up after ${MAX_RESTARTS} restarts"; exit 1
