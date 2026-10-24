# Architecture

This repository holds two research codebases that share one dashboard. Read this page first, then open the generated views at the bottom when you need detail.

| Codebase | Folder | Command | What it is |
|---|---|---|---|
| **Graph-JEPA** | [`Analysis/`](Analysis/README.md) | `graph-jepa` (root `pyproject.toml`) | Graph-JEPA pretraining on molecular/TU benchmarks and on a heterogeneous graph of scientific papers, plus the reasoning experiments and rebuttal diagnostics built on it |
| **Paper A (SCI-JEPA)** | [`SCI-JEPA-training/`](SCI-JEPA-training/README.md) | `sci-jepa` (its own `pyproject.toml`) | JEPA over typed knowledge units of a paper, a five-task reasoning benchmark, baselines, reports and the live dashboard |

The two do not import each other. The only link is on disk: Graph-JEPA runs write `runs.json` and `runs/<name>/live.json` into Paper A's dashboard folder, so both kinds of run appear on one page.

## How the pieces connect

```mermaid
flowchart LR
  subgraph GJ["Analysis/ · Graph-JEPA"]
    direction TB
    cfg["train/configs/*.yaml"] --> entry["train/&lt;dataset&gt;.py<br/>papers, zinc, mutag, ..."]
    entry --> data["core/get_data.py<br/>core/data_utils · core/transform.py"]
    entry --> model["core/get_model.py → core/model.py<br/>core/model_utils (GNN, MLP-Mixer, attention)"]
    entry --> trainer["core/trainer.py<br/>run / run_k_fold"]
    trainer -->|"backend: lightning"| engine["core/engine.py · core/lit.py<br/>core/objectives.py · callbacks.py"]
    trainer -->|"backend: legacy"| legacy["original loops in train/&lt;dataset&gt;.py"]
    engine --> exp["core/experiment.py"]
    hetero["core/data_utils/paper_graph.py<br/>hetero_graphA.pt"] --> reason["train/paper_reason*.py<br/>reasoning experiments"]
    rwse["train/build_rwse.py<br/>rwse_pe.pt"] --> reason
    hetero --> rwse
    reason -.->|"GJEPA_TRAIN_HOOK"| diag["diag/<br/>rebuttal diagnostics"]
  end
  subgraph PA["SCI-JEPA-training/ · Paper A"]
    direction TB
    pdata["data/<br/>parquet → KU graph → instances"] --> ptrain["models/ · training/"]
    ptrain --> peval["evaluation/<br/>tasks, baselines, validity"]
    peval --> prep["reporting/<br/>figures, PDF"]
    ptrain --> dash["dashboard/<br/>serve.py · dashboard.html"]
  end
  exp -->|"runs.json + live.json"| dash
  graph[("graphify-out/")] -.->|"Graphs tab"| dash
```

## Graph-JEPA (`Analysis/`)

Every Python command runs from inside `Analysis/` (`python -m train.papers`), because the packages are `core`, `train` and `diag`. `bash Analysis/run_local.sh <dataset>` does the `cd` for you.

| Layer | Files | Role |
|---|---|---|
| Config | `core/config.py`, `train/configs/*.yaml` | yacs defaults, one YAML per dataset, CLI overrides (`device 0`, `train.backend legacy`) |
| Data | `core/get_data.py`, `core/data_utils/`, `core/transform.py` | Loads ZINC, TU, EXP and the papers corpus; METIS partitioning into patches; positional encodings |
| Model | `core/get_model.py`, `core/model.py`, `core/model_utils/` | `GraphJepa`: context encoder, EMA target encoder, predictor |
| Training | `core/trainer.py` → `core/engine.py`, `core/lit.py`, `core/objectives.py` | `run` / `run_k_fold` dispatch to Lightning (default) or the legacy loops; `distance` or `vicreg` objective; DDP or one seed per GPU |
| Records | `core/experiment.py`, `core/callbacks.py`, `core/log.py`, `core/paths.py`, `core/seeds.py` | Run directories under `runs/`, dashboard files, logs, fixed seed lists |
| Reasoning experiments | `train/paper_reason*.py`, `train/build_rwse.py` | Experiments on the heterogeneous paper graph; see the [file map](Analysis/train/README.md), because the names are misleading |
| Diagnostics | `diag/` | `python -m diag.cli --stages ...` for the rebuttal stages; `diag/stages_w126.py` for the W1/W2/W6 checks |

### The papers pipeline (`Analysis/JEPA-rea.sh`)

| Part | Command | Produces |
|---|---|---|
| 1 | `python -m train.papers` | Graph-JEPA pretraining on the homogeneous paper graph |
| 2 | `core.data_utils.paper_graph` → `python -m train.paper_reason` | `dataset/papers/processed/hetero_graphA.pt`; reasoning (L2a) and link prediction (L2b) |
| 3 | `python -m train.build_rwse` → `python -m train.paper_reason_gjepa` | `rwse_pe.pt`; faithful Graph-JEPA ablation and the Euclidean-fix experiments |
| 4 | `python -m train.paper_reason_todo_experiments` (`JEPA-rea_todo_experiments.sh`) | The extended stage list (information ladder, rho sweep, figures) |

`run_LLM_rebuttal.sh` drives `diag.cli`; `run_LLM_rebuttal_w126.sh` drives `diag.stages_w126`.

## Paper A (`SCI-JEPA-training/`)

An installable package, `src/sci_jepa/`, with `data/ → models/ + training/ → evaluation/ → reporting/` and a `dashboard/` server. The package's own [README](SCI-JEPA-training/README.md) and [`docs/`](SCI-JEPA-training/docs/) cover it fully. Cluster jobs live in `cluster/grete/`, and throughput checks in `scaling/`.

## Generated views (Graphify)

Graphify builds a graph of every function, class, file and import edge, then groups related nodes into named communities. Each view below is made from that graph:

| View | Open | Best for |
|---|---|---|
| Architecture call-flow | `graphify-out/ARCHITECTURE.html` | One Mermaid diagram per project area (the 14 areas above) with call tables. Start here |
| Collapsible tree | `graphify-out/GRAPH_TREE.html` | Browsing area → community → symbol without the hairball |
| Report | [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) | Named communities, most connected symbols, surprising cross-file links |
| Full graph | `graphify-out/graph.html` | Free exploration and search; also shown in the dashboard's Graphs tab |

Open them in a browser (`open graphify-out/ARCHITECTURE.html` on macOS). From a terminal, answer one question instead of reading the whole map:

```bash
graphify query "how does the EMA target get updated"
graphify path "run_k_fold" "GraphJepaLitModule"
graphify explain "build_hetero_graph"
```

Refresh everything with `bash scripts/graph-refresh.sh` (incremental) or `bash scripts/graph-refresh.sh --relabel` (also renames every community). The git hooks already keep `graph.json` current after each commit, checkout and merge.
