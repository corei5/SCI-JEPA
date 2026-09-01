# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SCI-JEPA builds a heterogeneous knowledge graph from LLM-summarised scientific papers, pretrains
graph representations with a JEPA-style objective, and evaluates them on reasoning and
link-prediction probes. There is also a faithful Graph-JEPA ablation (RWSE + GINE).

## Repository split

Two independent top-level packages, with **different working directories**:

| Package | Purpose | Run from |
|---|---|---|
| `dataset_creation/` | raw paper JSON → knowledge graph | repo root |
| `Analysis/` | modelling: pretraining, probes, ablations, diagnostics | `Analysis/` |

`Analysis/` scripts are invoked as `python -m train.paper_reason` with `cwd=Analysis`, so
`core` and `train` are top-level importable there. `dataset_creation` is invoked as
`python -m dataset_creation.build_sample` from the repo root. Do not mix the two conventions.

`Analysis/core/data_utils/paper_graph.py` and `papers.py` are **thin compatibility shims** that
insert the repo root on `sys.path` and re-export from `dataset_creation/`. Training scripts still
do `from core.data_utils.paper_graph import build_hetero_graph, ASPECTS`. Edit the real code in
`dataset_creation/`, never the shims.

## Graph construction architecture

`build_hetero_graph()` is deliberately split into two passes across two files:

- **Pass 1 — `dataset_creation/schema.py`** (standard library only). `parse_records()` reduces each
  raw record to a small dict, then `build_tables()` flattens those into node text lists and typed
  edge index lists. No torch, no GPU, no model download.
- **Pass 2 — `dataset_creation/paper_graph.py`**. Encodes every node text with
  `all-MiniLM-L6-v2` (384-d, frozen — nothing is learned here) and packs the tables into a
  `HeteroData`, pickled to `hetero_graphA.pt`.

The split exists so `build_sample.py` and `visualize.py` can reproduce the **exact** node and edge
sets of the full graph on a laptop. Both call the same `build_tables()`, so the sample cannot drift
from what the real pipeline embeds. Keep schema changes in `schema.py`; if you add parsing logic to
`paper_graph.py` directly, you break that guarantee.

### Structural facts that are not obvious from the code

- The result is **one big `HeteroData`, not N per-paper graphs**. Each paper is still its own
  connected reasoning subgraph because edges are typed and intra-paper — with PyG's `to_hetero`,
  message passing is already local. This is why `build_rwse.py` can treat papers as disconnected
  components.
- Two of the nine relations are **imposed, not read from the data**: `(method, produces, result)`
  and `(result, grounds, evidence)`. The LLM summary never says which evidence came from which
  result, so the builder wires a paper's result to *all* of its evidence. This is what makes a paper
  a chain (`paper → method → result → evidence → claim → implication`) rather than a star.
- **Evidence is a link in that chain, not a leaf on a claim.** The edge runs `(evidence, supports,
  claim)` — hence `supports`, not `supported_by`. `contradicting_evidence` is deliberately not read
  into the graph, so there is no `challenges` relation and every evidence node is supporting
  evidence. A claim with no evidence therefore hangs off `has_claim` alone; `coverage_report()`
  counts those as `claims_without_evidence` and the builder prints the count.
- `cites` is a **directed** relation: `(i, j)` means *i* cites *j* (src = citing, dst = cited). It
  is stored as one relation only — `Analysis/` applies `T.ToUndirected()`, which synthesises
  `rev_cites`, so adding a `cited_by` edge type would double those edges. `build_tables` drops
  three things and counts each: references outside the corpus (dangling — the vast majority, so
  citation edges are very sparse), self-loops (records listing their own `openalex_id`), and
  duplicate pairs. `schema.citation_report()` verifies the orientation against `oa_year` and
  reports how many edges cite a *newer* paper; that number should stay near zero.
- Node rows are assigned in corpus order with files read in **sorted** order, so tables are
  deterministic. (Upstream this used unsorted `glob`.)
- `field` nodes are shared hubs and are the only cross-paper structure besides `cites`. The label is
  coarsened by default: `"Medicine — Pulmonology"` → `"Medicine"`.

### Two different dataset constructions

Both read the same raw records but **share no code path** — don't unify them:

- `dataset_creation/paper_graph.py` → the Part 2/3 reasoning KG (7 node types, one big graph).
- `dataset_creation/papers.py` → `PapersDataset`, used only by Part 1 pretraining. One small graph
  *per paper*, nodes fully connected, richer 14-field node-type vocabulary, `y` = `field_subfield`
  class id, rare classes dropped via `min_class_size`.

### Record formats

Three layouts appear in the corpus and all are handled by `schema._coerce_summary`:
Format A/B nests fields under `summarization` (often a **stringified** JSON) → `summary`, with
claims under `details`; Format C is flat at the top level with claims under `description`. Records
whose `summary` is `null` (`NON_SCIENTIFIC_TEXT`) are skipped; `PARTIAL_SCIENTIFIC_TEXT` is kept.
`claims` and `oa_referenced_works` are frequently stringified JSON and get re-parsed.

## Data availability — read before attempting any build

The corpus is **not in this repository**, and no code change fixes it:

- `Analysis/dataset/papers/zip/raw.zip`, `processed/hetero_graphA.pt` and `processed/rwse_pe.pt` are
  each **exactly 262,144 bytes (256 KiB)** — on disk *and* in git history. They were committed
  truncated via a web upload. The zip has no central directory, so `unzip` fails; only ~17 of
  57,903 papers are recoverable by walking local file headers.
- `Analysis/dataset/papers/raw/` does not exist.
- The source dataset [`ai4sci-tib/LAION_arxiv-open`](https://huggingface.co/datasets/ai4sci-tib/LAION_arxiv-open)
  is **gated**. The token at `~/.cache/huggingface/token` is a stale 825-char OAuth token and returns
  401 "signature verification failed". A valid `hf_...` user access token in `$HF_TOKEN` is required.

`dataset_creation/fetch_hf.py` downloads the corpus using stdlib `urllib` only (no `datasets` or
`huggingface_hub` install needed). Build subsets rather than the whole corpus — `parse_records()`
accepts `field=` and `domain=` filters, applied *before* `limit` is counted.

## Commands

```bash
# --- dataset creation (repo root, standard library only) ---
python -m dataset_creation.build_sample --snippet 400      # sample_raw/ -> sample/
python -m dataset_creation.visualize                       # sample -> static SVG page
python -m dataset_creation.explore --kg <kg-dir>           # full KG -> interactive explorer.html
python -m dataset_creation.build_sample --raw-dir <dir> --out <dir> --limit 500
python -m dataset_creation.fetch_hf --list                 # needs HF_TOKEN

# slice the 6 GB zip by directory (--prefix is repeatable, and rejects records
# without opening them; --label-source field is REQUIRED on the arXiv corpus)
python -m dataset_creation.build_sample \
    --raw-dir dataset_creation/data/LAION_arxiv-open.zip \
    --prefix physical_sciences/materials_science/ \
    --prefix physical_sciences/chemical_engineering/ \
    --label-source field --out dataset_creation/data/kg_materials_chemeng

# --- modelling (cwd must be Analysis/) ---
cd Analysis
python -m train.papers --config train/configs/papers.yaml device 0   # Part 1
python -m train.paper_reason                                          # Part 2
python -m train.build_rwse && python -m train.paper_reason_gjepa      # Part 3
python -m diag.cli --stages baseline,ridge --corpus papers            # diagnostics
bash JEPA-rea.sh                                                      # or: sbatch JEPA-rea.sh
```

There is **no test suite, no linter config, and no requirements/pyproject file** in this repo.
Dependencies (`torch`, `torch_geometric`, `sentence-transformers`, `yacs`, `scikit-learn`,
`matplotlib`) are documented only in `README.md` and must be installed by hand.

## Gotchas when running anything under `Analysis/`

- **Absolute paths are hardcoded to another machine.** `RAW_DIR`, `CACHE_PATH`, `RWSE_CACHE` and
  `CKPT_DIR` are `/nfs/home/rabbyg/JEPA/Graph-JEPA/...` constants at the top of `train/paper_reason.py`,
  `train/build_rwse.py`, `train/paper_reason_frozen_features.py`, `core/get_data.py`, and the four
  `.sh` drivers. Every one must be edited before a local run.
- The pipeline is **cache-driven**: each stage writes to `dataset/papers/processed/` and later stages
  reuse it. `JEPA-rea.sh` exposes `RUN_PART1/2/3` stage toggles and `REBUILD_HETERO` / `REBUILD_RWSE`
  flags. Part 3 requires `hetero_graphA.pt` to already exist.
- Config uses **yacs**: defaults in `core/config.py`, overridden by `train/configs/*.yaml`, then by
  trailing `key value` CLI pairs via `update_cfg`. `papers_nout` in `papers.yaml` must be set by hand
  to the `num_classes` that `PapersDataset` prints.
- `train/` contains several parallel forks of the same experiment
  (`paper_reason_gjepa_old_1.py`, `_old_2.py`, `_todo_experiments.py`,
  `paper_reason_frozen_features.py`). `paper_reason.py` and `paper_reason_gjepa.py` are the current
  ones; treat the rest as frozen history and don't refactor them in step with the live scripts.

When updating something always explain each code change and logic decision in simple terms before applying it.