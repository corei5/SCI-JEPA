#!/bin/bash
# Previous vs optimized Paper A code on one GPU type: every configuration below, both code versions, with
# scaling/bench_compare.py (the main training pipeline, train_one). Self-contained: needs only the kit bundle
# and a Python environment with the Paper A dependencies (torch, numpy, pyarrow, nvidia-ml-py, psutil,
# sentence-transformers). No internet, no repository checkout.
#
#   BUNDLE=/path/to/sci-jepa-bench-kit OUT=/path/to/results bash $BUNDLE/bench_kit.sh
#   (on SLURM: sbatch --gres=gpu:4 --cpus-per-task=32 --time=02:00:00 --wrap "BUNDLE=... OUT=... bash .../bench_kit.sh")
#
# Optional: PYTHON (default python), WORKERS (batch worker processes, default 7), CONCURRENT=1 (also run one
# training per visible GPU at once, when >= 2 GPUs are visible), QUICK=1 (only the S and M models, batch 128).
# With >= 2 visible GPUs, one run is also trained on min(GPUs, 4) GPUs at once (PyTorch Lightning DDP).
set -uo pipefail
BUNDLE="${BUNDLE:?set BUNDLE to the unpacked kit folder}"
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-7}"
export HF_HOME="${BUNDLE}/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
GPU_NAME=$("${PY}" -c "import torch; print(torch.cuda.get_device_name(0))" | tr ' ' '_' )
OUT="${OUT:-${BUNDLE}/results}/${GPU_NAME}"
mkdir -p "${OUT}"
echo "kit: ${BUNDLE} | gpu: ${GPU_NAME} | out: ${OUT} | $(date)"
nvidia-smi -L

run() {   # label size batch speed epochs [tag] [trainer] [devices]
  local label=$1 size=$2 batch=$3 speed=$4 epochs=$5 tag=${6:-} trainer=${7:-native} devices=${8:-1}
  local name="${label}_${trainer}${devices}_${size}_b${batch}_${speed}${tag:+_${tag}}"
  echo "=== ${label} ${trainer} x${devices} size ${size} batch ${batch} speed ${speed} epochs ${epochs} ${tag} $(date +%T)"
  PYTHONPATH="${BUNDLE}/code/${label}/src:${PYTHONPATH:-}" "${PY}" "${BUNDLE}/bench_compare.py" --label "${label}" \
    --size "${size}" --batch "${batch}" --speed "${speed}" --epochs "${epochs}" --workers "${WORKERS}" \
    --trainer "${trainer}" --devices "${devices}" \
    --split "${BUNDLE}/split" --index-dir "${BUNDLE}/index" --out "${OUT}" ${tag:+--tag "${tag}"} \
    > "${OUT}/${name}.log" 2>&1
  echo "    rc=$? $(grep '^\[bench\]' "${OUT}/${name}.log" | tail -1)"
}

NG=$("${PY}" -c "import torch; print(torch.cuda.device_count())")
NDDP=$(( NG >= 4 ? 4 : NG ))                          # GPUs for the one-run-on-several-GPUs test
if [[ "${QUICK:-0}" == 1 ]]; then
  CONFIGS=("S 128 gpu 1" "M 128 gpu 0.5")
else
  CONFIGS=("local 64 gpu 1" "S 128 gpu 1" "S 256 gpu 1" "M 128 gpu 0.5")
  run previous local 64 fast 1; run optimized local 64 fast 1     # SPEED=fast does not use the changed path
fi
for c in "${CONFIGS[@]}"; do
  run previous ${c}                                   # previous code, its own training loop
  run optimized ${c}                                  # optimized code, its own training loop
  run optimized ${c} "" lightning 1                   # optimized code, PyTorch Lightning loop, 1 GPU
  set -- ${c}
  if [[ "${NDDP}" -ge 2 && "$1" != "local" ]]; then
    run optimized ${c} "" lightning ${NDDP}           # one run on several GPUs (Lightning DDP, global batch split)
  fi
done

if [[ "${CONCURRENT:-1}" == 1 && "${NG}" -ge 2 ]]; then
  for label in previous optimized; do      # Paper A scales across GPUs by running arms / seeds side by side
    echo "=== ${label}: ${NDDP} trainings at once, one per GPU $(date +%T)"
    for g in $(seq 0 $(( NDDP - 1 ))); do
      CUDA_VISIBLE_DEVICES=$g WORKERS=$(( WORKERS )) run ${label} S 128 gpu 0.5 "concurrent${g}" &
    done
    wait
  done
fi
# where the GPU time goes, operation by operation (torch.profiler), both versions: the evidence for WHY
for size in local S; do
  for label in previous optimized; do
    echo "=== trace ${label} size ${size} $(date +%T)"
    PAPER_A_PARQUET="sci-jepa-mini/normalized/arxiv" PYTHONPATH="${BUNDLE}/code/${label}/src:${PYTHONPATH:-}"       "${PY}" "${BUNDLE}/trace_step.py" --split "${BUNDLE}/split" --index "${BUNDLE}/index" --n-papers 6000       --arm reference --speed gpu --size "${size}" --label "${label}" --out "${OUT}/trace"       > "${OUT}/trace_${label}_${size}.log" 2>&1
    echo "    rc=$? $(grep '^step:' "${OUT}/trace_${label}_${size}.log")"
  done
done
echo "done $(date)"
