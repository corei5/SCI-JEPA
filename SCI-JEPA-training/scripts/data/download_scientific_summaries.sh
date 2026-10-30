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
# Set HF_TOKEN (or `hf auth login`): anonymous downloads are rate-limited (HTTP 429) long before 4,006 files.
#
# `hf download` alone cannot tell when it is done: on a 429 it returns the existing local folder with exit code 0,
# and its file list of this large repository can be incomplete (695 of 4,006 arxiv files). So every attempt ends
# with a check of each data/<subset> file against the Hub's folder listing (name and size); missing or partial
# files are fetched one by one, and only a complete check ends the loop.
set -uo pipefail

DEST="${DEST:-data/raw}"
SUBSETS="${SUBSETS:-arxiv pubmed}"
REVISION="${REVISION:-232547b7b938a6588ce5d9d7faa96384860b7a16}"   # the revision KU-Fixes was built against
WORKERS="${WORKERS:-8}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
PYTHON="${PYTHON:-$(sed -n '1s/^#!//p' "$(command -v hf)")}"       # the interpreter of the `hf` command

INCLUDE=(--include "*.md" --include "*.json")
for s in ${SUBSETS}; do INCLUDE+=(--include "data/${s}/*"); done   # one --include per pattern

check_and_fetch() {   # exit 0 when every data/<subset> file of REVISION is in DEST with the Hub's size
  ${PYTHON} - "${DEST}" "${REVISION}" "${WORKERS}" ${SUBSETS} <<'EOF'
import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
dest, rev, workers, subsets = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
repo = "laion/Scientific-Summaries"
def ok(f):
    p = os.path.join(dest, f.path)
    return os.path.isfile(p) and os.path.getsize(p) == f.size
def fetch(f):
    try:
        hf_hub_download(repo, f.path, repo_type="dataset", revision=rev, local_dir=dest)
    except Exception as e:
        print(f"  {f.path}: {type(e).__name__}: {str(e)[:120]}", flush=True)
missing = 0
for s in subsets:
    files = [f for f in HfApi().list_repo_tree(repo, repo_type="dataset", path_in_repo=f"data/{s}", revision=rev)
             if hasattr(f, "size")]
    todo = [f for f in files if not ok(f)]
    print(f"=== {s}: {len(files) - len(todo)} of {len(files)} files present, fetching {len(todo)}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, todo))
    left = sum(not ok(f) for f in todo)
    if left:
        print(f"=== {s}: {left} files still missing", flush=True)
    missing += left
sys.exit(1 if missing else 0)
EOF
}

for attempt in $(seq 1 50); do
  echo "=== attempt ${attempt} $(date)"
  hf download laion/Scientific-Summaries --repo-type dataset --revision "${REVISION}" \
    "${INCLUDE[@]}" --local-dir "${DEST}" --max-workers "${WORKERS}"   # bulk download; its exit code is not trusted
  if check_and_fetch; then echo "=== all files present"; break; fi
  sleep 60
done
for attempt in $(seq 1 10); do
  hf download laion/Scientific-Summaries-KU-Fixes --repo-type dataset --local-dir "${DEST}-KU-Fixes" && break
  sleep 60
done
du -sh "${DEST}"/data/* "${DEST}-KU-Fixes" 2>/dev/null
echo "next: normalize each subset, e.g."
echo "  PARQUET_FOLDER=${DEST}/data/arxiv OUT_PATH=data/normalized/arxiv python scripts/data/normalize_parquet.py"
