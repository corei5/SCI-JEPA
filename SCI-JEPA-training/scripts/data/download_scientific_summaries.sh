#!/bin/bash
# Download subsets of laion/Scientific-Summaries (raw parquet) and the KU-Fixes index from Hugging Face.
# Resumable: finished files are skipped; retries until every file is present.
#
#   bash scripts/data/download_scientific_summaries.sh                       # arxiv + pubmed into ./data/raw
#   DEST=/big/disk/raw SUBSETS="arxiv pubmed web-papers" bash scripts/data/download_scientific_summaries.sh
#
# Sizes (data/ only): arxiv 153 GB (4,006 files), pubmed 18 GB (347), web-papers 160 GB (3,487),
# bethgelab 15 GB (67). The repository also has search/ indexes (~1.75 TB) that Paper A does not use.
# Needs: pip install "huggingface_hub>=0.30" (the `hf` command; hf_xet is used when installed).
set -uo pipefail

DEST="${DEST:-data/raw}"
SUBSETS="${SUBSETS:-arxiv pubmed}"
REVISION="${REVISION:-232547b7b938a6588ce5d9d7faa96384860b7a16}"   # the revision KU-Fixes was built against
WORKERS="${WORKERS:-8}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

INCLUDE=(--include "*.md" --include "*.json")
for s in ${SUBSETS}; do INCLUDE+=(--include "data/${s}/*"); done   # one --include per pattern

for attempt in $(seq 1 50); do
  echo "=== attempt ${attempt} $(date)"
  if hf download laion/Scientific-Summaries --repo-type dataset --revision "${REVISION}" \
       "${INCLUDE[@]}" --local-dir "${DEST}" --max-workers "${WORKERS}"; then
    echo "=== all files present"; break
  fi
  sleep 30
done
hf download laion/Scientific-Summaries-KU-Fixes --repo-type dataset --local-dir "${DEST}-KU-Fixes"
du -sh "${DEST}"/data/* "${DEST}-KU-Fixes" 2>/dev/null
echo "next: normalize each subset, e.g."
echo "  PARQUET_FOLDER=${DEST}/data/arxiv OUT_PATH=data/normalized/arxiv python scripts/data/normalize_parquet.py"
