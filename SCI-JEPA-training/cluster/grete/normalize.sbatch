#!/bin/bash
#SBATCH --job-name=normalize
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# Normalize one raw subset (scripts/data/normalize_parquet.py) with WORKERS parallel processes. Shards already
# normalized are skipped, so re-submitting resumes. Grete's GPU partitions need a GPU even for CPU work: the
# smallest slice is requested. From the repository root:
#   SUBSET=arxiv sbatch -A kisski-l3s-llm4sci -p grete:interactive -o logs/normalize_%j.log cluster/grete/normalize.sbatch
set -uo pipefail
REPO="${REPO:-$(pwd)}"
source "${SITE_ENV:-${REPO}/cluster/grete/site.env}"
SUBSET="${SUBSET:-arxiv}"
SRC="${SRC:-${DATA_ROOT}/data/${SUBSET}}"
OUT="${OUT:-${DATA_ROOT}/normalized/${SUBSET}}"
WORKERS="${WORKERS:-16}"
PARTS="${OUT}/../_parts_${SUBSET}"
mkdir -p "${OUT}" "${PARTS}"

# split the shards that are not done yet into WORKERS folders of symlinks
ls "${SRC}" | grep '\.parquet$' | while read -r f; do [[ -s "${OUT}/${f}" ]] || echo "${f}"; done > "${PARTS}/todo.txt"
echo "$(wc -l < "${PARTS}/todo.txt") shards to do, ${WORKERS} workers, $(date)"
for k in $(seq 0 $((WORKERS - 1))); do mkdir -p "${PARTS}/in_${k}"; done
i=0; while read -r f; do ln -sf "${SRC}/${f}" "${PARTS}/in_$((i % WORKERS))/${f}"; i=$((i + 1)); done < "${PARTS}/todo.txt"

for k in $(seq 0 $((WORKERS - 1))); do
  PARQUET_FOLDER="${PARTS}/in_${k}" OUT_PATH="${PARTS}/out_${k}" LOG_RENAMES=0 SHOW_MEMORY=0 \
    "${VENV}/bin/python" "${REPO}/scripts/data/normalize_parquet.py" > "${PARTS}/worker_${k}.log" 2>&1 &
done
wait
for k in $(seq 0 $((WORKERS - 1))); do mv "${PARTS}/out_${k}"/*.parquet "${OUT}/" 2>/dev/null; done
echo "normalized ${SUBSET}: $(ls "${OUT}"/*.parquet | wc -l) of $(ls "${SRC}"/*.parquet | wc -l) shards, $(date)"
rm -rf "${PARTS}"
# validate (per shard, constant memory): python -m pytest -q tests/test_normalized_parquet.py with OUT_DIR=${OUT}
