# Rank-2-Trap



Rank-2-Trap builds a heterogeneous knowledge graph from a corpus of scientific papers (authors, works, concepts, citations, etc.), learns self-supervised graph representations with a JEPA-style objective, and evaluates those representations on **reasoning** and **link-prediction** tasks. It also includes a faithful Graph-JEPA ablation using Random-Walk Structural Encodings (RWSE) with a GINE backbone.

---

## ✨ Features

- **Heterogeneous reasoning graph** built directly from raw paper JSON (nodes for papers/authors/concepts, edges such as `cites` and `makes`).
- **Graph-JEPA self-supervised pretraining** (masked, within-paper predictive objective).
- **Reasoning + link-prediction heads** for evaluating learned embeddings.
- **RWSE positional encodings** and a faithful Graph-JEPA ablation (GINE + masked JEPA).
- **SLURM-ready** end-to-end pipeline with cached, resumable stages.

---

## 📁 Repository Structure

```
SCI-JEPA/
├── pyproject.toml              # uv project (Lightning, PyG, TensorBoard)
├── Analysis/
│   ├── core/                   # model, data, Lightning trainer, dashboard
│   ├── train/                  # dataset entrypoints and YAML configs (file map: train/README.md)
│   ├── diag/                   # rebuttal diagnostics
│   ├── dataset/papers/         # raw JSON and cached tensors
│   └── JEPA-rea.sh             # SLURM pipeline
├── runs/                       # Lightning JSONL logs (created on train)
└── graphify-out/               # codebase knowledge graph
```

See [train/README.md](train/README.md) for what each `train/` script does; several `paper_reason*` names no longer match their contents. The repository-wide map is [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 🧩 Pipeline Overview

The full workflow is orchestrated by `JEPA-rea.sh` and split into three toggleable parts:

| Part | Script | Description |
|------|--------|-------------|
| **Part 1** | `train.papers` | Graph-JEPA self-supervised pretraining. |
| **Part 2** | `core.data_utils.paper_graph` → `train.paper_reason` | Builds the heterogeneous reasoning graph (`hetero_graphA.pt`), then trains reasoning (L2a) + link prediction (L2b). |
| **Part 3** | `train.build_rwse` → `train.paper_reason_gjepa` | Builds RWSE positional encodings (`rwse_pe.pt`), then runs the faithful Graph-JEPA ablation (A1 reasoning probe + A2 patch retrieval). |

Each stage caches its output to `dataset/papers/processed/`, so later stages can reuse expensive computation (node embedding, RWSE) without recomputing.

---

## 🚀 Getting Started

### 1. Environment

From the repository root (the directory that contains `pyproject.toml`):

```bash
uv sync
uv sync --extra data
```

The training entrypoints default to PyTorch Lightning and write into the dashboard in `SCI-JEPA-training` (`uv run graph-jepa dashboard`). See the root [README](../README.md) for multi-GPU runs and `SCIJEPA_DASHBOARD_ROOT`.

A conda environment still works if you install the same libraries (`torch`, `torch-geometric`, `lightning`, `yacs`, `scikit-learn`). The SLURM script activates `kgjepa` and exports `SCIJEPA_DATA_ROOT`.

### 2. Data

**Dataset:** [`ai4sci-tib/LAION_arxiv-open`](https://huggingface.co/datasets/ai4sci-tib/LAION_arxiv-open) on Hugging Face.

Place your raw paper JSON files in:

```
dataset/papers/raw/
```

Each `.json` file is expected to contain fields such as `openalex_id` and `oa_referenced_works` (used to construct `cites` edges).

### 3. Run the pipeline

Edit the paths and stage toggles at the top of `JEPA-rea.sh`, then submit:

```bash
sbatch JEPA-rea.sh
```

Or run locally (bash):

```bash
bash JEPA-rea.sh
```

---

## ⚙️ Configuration

Key knobs in `JEPA-rea.sh`:

```bash
# Which stages to run (1 = run, 0 = skip)
RUN_PART1=0    # Graph-JEPA self-supervised pretraining
RUN_PART2=1    # Reasoning graph + link prediction
RUN_PART3=1    # Faithful Graph-JEPA ablation (RWSE + probes)

# Cache rebuild flags (1 = force rebuild, 0 = reuse cached tensors)
REBUILD_HETERO=1   # Rebuild hetero_graphA.pt
REBUILD_RWSE=1     # Rebuild rwse_pe.pt
```

Training hyperparameters and dataset settings are defined in `train/configs/papers.yaml`.

You can also run individual stages directly:

```bash
# Part 1
python -m train.papers --config train/configs/papers.yaml device 0

# Part 2
python -m train.paper_reason

# Part 3
python -m train.build_rwse
python -m train.paper_reason_gjepa
```

---

## 💾 Caches

| Cache file | Produced by | Purpose |
|------------|-------------|---------|
| `dataset/papers/processed/hetero_graphA.pt` | `core.data_utils.paper_graph` | Heterogeneous reasoning graph |
| `dataset/papers/processed/rwse_pe.pt` | `train.build_rwse` | RWSE positional encodings |
| `papers_coarse.pt`, `papers_full.pt`, `label_map.json` | `train.papers` (Part 1) | Part-1 dataset caches |

Set the corresponding `REBUILD_*` flag to `0` to reuse an existing cache and skip recomputation.

---

## 📊 Logs

All runs write a combined, timestamped log to:

```
log/papers_run_<jobid>_<timestamp>.log
```

Under SLURM, separate stdout/stderr are also written to `log/papers_%j.out` and `log/papers_%j.err`.

---

## 📝 Notes

- If `openalex_id` coverage in the raw data is low, `cites` edges will be sparse and Part 2 falls back to predicting `makes` edges.
- Parts 2 and 3 share the hetero-graph cache — running Part 3 requires that `hetero_graphA.pt` already exists (build it in Part 2 first, or point to an existing cache).

---

## 📄 License

_No license specified yet — consider adding one (e.g., MIT, Apache-2.0)._

## 🙌 Acknowledgements

Built on top of Graph-JEPA / JEPA ideas and PyTorch Geometric.
