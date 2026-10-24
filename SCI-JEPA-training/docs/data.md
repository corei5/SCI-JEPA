# Data: download, normalize, split

SCI-JEPA reads [laion/Scientific-Summaries](https://huggingface.co/datasets/laion/Scientific-Summaries):
LLM-structured summaries of scientific papers (context, question, method, procedure, results, claims with
evidence, ...), one parquet row per paper. Three steps turn it into training data; each runs once.

```
raw parquet  --(1) download-->  data/raw/data/<subset>/
             --(2) normalize--> data/normalized/<subset>/          one fixed 44-column schema
             --(3) split------> data/splits/<name>/{train,val,test}/  year split, domain-balanced, cleaned
```

## 1. Download

```bash
bash scripts/data/download_scientific_summaries.sh                         # arxiv + pubmed -> data/raw
DEST=/big/disk/raw SUBSETS="arxiv pubmed" bash scripts/data/download_scientific_summaries.sh
```

| Subset | Files | Size | Papers | Notes |
|---|---|---|---|---|
| arxiv | 4,006 | 153 GB | 3,324,592 | shards `new_*` (arXiv ids), `old_*` and `titled_*` (internal file-path ids) |
| pubmed | 347 | 18 GB | 1,732,185 | most rows have no publication year (see below) |
| web-papers | 3,487 | 160 GB | 16.7 M | not used yet: mixed quality, few dates |
| bethgelab | 67 | 15 GB | 0.33 M | not used yet |

The script pins revision `232547b7…`, the one the KU-Fixes index was built against, retries until every
file is present (finished files are skipped), and also downloads `laion/Scientific-Summaries-KU-Fixes`
(duplicate clusters, near-empty summaries), which step 3 uses. The repository's `search/` folder (BM25,
embeddings, FAISS; ~1.75 TB) is not needed.

## 2. Normalize

The raw shards name and type their columns differently. `scripts/data/normalize_parquet.py` maps every row
onto one fixed schema (44 keys), one output file per input shard, in constant memory and resumable:

```bash
PARQUET_FOLDER=data/raw/data/arxiv OUT_PATH=data/normalized/arxiv python scripts/data/normalize_parquet.py
OUT_DIR=data/normalized/arxiv python -m pytest -q tests/test_normalized_parquet.py     # validate
```

On a cluster, `cluster/grete/normalize.sbatch` runs it with 16 parallel workers (arXiv: ~6 minutes).
Every normalized shard has the same row count and row order as its raw shard, so KU-Fixes row positions
still apply.

## 3. Split (train / validation / test), domain-balanced

```bash
sci-jepa split --parquet data/normalized/arxiv,data/normalized/pubmed \
               --ku-fixes data/raw-KU-Fixes --out data/splits/d0 \
               --train-papers 60000 --val-papers 5000 --test-papers 5000
```

Training and benchmarking then use the stored split (`PAPER_A_SPLIT=data/splits/d0`, `--split-dir`, or
`SPLIT=` in the cluster script) and never sample or split again. `--n-papers` / `max_papers` then takes the
same share of every split, in the split's stored random order, so a smaller run uses a random, equally
balanced subset of the same split.

### How the split is made (`src/sci_jepa/data/split.py`)

| Step | What happens |
|---|---|
| 1. Year and domain | every paper's year (`source_year`, `oa_year`, or the arXiv id) and research domain (10 canonical domains from the free-text field label, `ku_schema.domain_of`); metadata columns only |
| 2. Year split | **train ≤ 2023, validation 2024, test 2025–2026** (`--train-until`, `--val-years`, `--test-years`). Papers without a year, or from other years, are left out |
| 3. Cleaning | KU-Fixes: every copy of a duplicated paper except the one to keep (no paper can sit in two splits), and near-empty summaries |
| 4. Domain balance | one set of domain shares from the pooled counts of all splits, by temperature sampling (share ∝ count^α, α = 0.5), applied to **every** split: train, validation and test get the same domain mix. A domain with too few papers in a split gives its leftover to the others |
| 5. Quality | candidates must pass the KU graph's own rules (≥ 8 KUs, a result and a claim), so every stored paper is used; a rejected paper is replaced from the same split and domain |
| 6. Output | `train/`, `val/`, `test/` parquet (normalized schema + `_split`, `_split_rank`), `papers.csv` (one row per paper with its source file and row), `manifest.json` (settings and counts at every step) |

`--balance` also takes `uniform` (every domain equal), `natural` (pooled counts) or `none` (each split keeps
its own mix); `--train-papers 0` keeps every usable paper.

### Why the balance is needed: arXiv + PubMed per split

| Domain | train ≤ 2023 | val 2024 | test 2025–26 |
|---|---|---|---|
| Physics | 813,206 (36%) | 37,440 (22%) | 68,367 (22%) |
| Mathematics | 395,465 (18%) | 27,322 (16%) | 44,815 (14%) |
| Computer Science | 373,552 (17%) | 73,263 (42%) | 147,103 (47%) |
| Quantitative Biology & Medicine | 259,606 (11%) | 8,966 (5%) | 5,810 (2%) |
| Astrophysics | 258,989 (11%) | 11,906 (7%) | 23,296 (7%) |
| EESS & Engineering | 53,911 | 6,627 | 12,468 |
| Other | 47,762 | 1,961 | 2,201 |
| Statistics | 28,862 | 2,896 | 5,305 |
| Economics & Finance | 16,214 | 2,075 | 4,043 |
| Chemistry | 11,113 | 645 | 719 |
| **total (after cleaning)** | **2,258,680** | **173,101** | **314,127** |

Left out: 2,306,936 papers without a year (1.43 M of PubMed's 1.73 M, and 877 k arXiv `titled` / `old`
rows) and 3,932 duplicate copies or near-empty summaries. Without a stored balanced split, Computer Science
would be 17% of training but 47% of the test set.

### Paper ids

Only the `new_*` arXiv shards use arXiv ids (`0704.0001`); `old_*`, `titled_*` and PubMed use internal file
paths from LAION's processing. They are unique keys inside SCI-JEPA; to link a paper to arXiv or PubMed,
use `source_title`, `source_doi` or the OpenAlex columns.
