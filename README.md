# SCI-JEPA

Two training stacks live here. [`SCI-JEPA-training/`](SCI-JEPA-training/README.md) is Paper A: the `sci-jepa` command and the dashboard. [`Analysis/`](Analysis/README.md) is Graph-JEPA pretraining. It uses Lightning and writes into that same dashboard. This file is how to install and run the Graph-JEPA side.

| Read | For |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | What each folder does and how the parts connect |
| [Analysis/train/README.md](Analysis/train/README.md) | Which `train/` script is which (several names are misleading) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branches, pull requests, uploading through the web UI |
| [AGENTS.md](AGENTS.md) | Instructions that Claude Code, Codex and Cursor read |
| [docs/sciku-bench-paper-guide.md](docs/sciku-bench-paper-guide.md) | The SciKU-Bench reasoning-benchmark paper plan |

## Install

The project uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --group dev
uv sync --extra data --group dev   # sentence-transformers, only to embed raw papers
```

`uv sync` creates `.venv` and installs PyTorch, PyTorch Geometric, and Lightning. The model also imports `torch-scatter`, which is not published as a macOS wheel on PyPI (PyTorch Geometric 2.8 no longer depends on it). After sync, install the build that matches your torch:

```bash
uv run python -c "import torch; print(torch.__version__)"
uv pip install torch-scatter --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

Change `2.5.0+cpu` to your torch version and CUDA tag (`cu124`, `cu126`, ...).

## Train

From `Analysis/` (or via the helper, which cds there):

```bash
bash Analysis/run_local.sh papers device 0
bash Analysis/run_local.sh zinc
```

The same entry from the uv environment:

```bash
uv run graph-jepa train papers device 0
```

### Changing hyperparameters

Defaults live in `Analysis/core/config.py`; each dataset's YAML in `Analysis/train/configs/` overrides them. On the command line, the common options have flags and every other key is set as `KEY VALUE` or `KEY=VALUE`:

```bash
uv run graph-jepa train papers --epochs 50 --lr 3e-4 metis.n_patches 16 jepa.num_targets=4
uv run graph-jepa train papers --help          # all flags
uv run graph-jepa options papers ema           # keys matching "ema", their values and help
uv run graph-jepa config papers --lr 3e-4 > my.yaml   # resolved config, nothing runs
uv run graph-jepa train papers --config my.yaml
```

Order of precedence: `config.py`, the dataset YAML, `--config`, `KEY VALUE` pairs, then flags. A mistyped key stops the run and suggests the closest key. `python -m train.<dataset>` accepts the same arguments, so the SLURM scripts' `--config FILE device 0` still works.

Lightning Trainer switches sit under `trainer.*` (unset keys keep Lightning's default):

```bash
uv run graph-jepa train papers --fast-dev-run                 # one batch through train and val
uv run graph-jepa train papers trainer.limit_train_batches 0.1 trainer.profiler simple
uv run graph-jepa train papers trainer.max_time 00:23:30:00   # stop cleanly before a SLURM limit
```

### Checkpoints and resuming

Each fit writes `checkpoints/[fold<k>/]last.ckpt` in its run directory after every epoch (`train.save_last`). With a fixed run name, a rerun continues where it stopped; finished folds and seeds return straight to their probe:

```bash
uv run graph-jepa train papers --name papers-lr3e4 --lr 3e-4 --resume auto
```

`--resume path/to/last.ckpt` resumes a single split run with one seed.

`train.backend` defaults to `lightning`. Set `train.backend legacy` to use the original loops. Papers use the VICReg objective (`train.objective: vicreg`); the TU and ZINC scripts use the distance objective. The target network is updated with `train.ema_scope: full` on every step.

One process can use several GPUs, or several seeds can run at once:

```yaml
train:
  devices: 2          # DDP across 2 GPUs, one seed at a time
  strategy: ddp
  parallel_runs: false
```

```yaml
train:
  devices: 0          # keep the single GPU in `device`
  parallel_runs: true # one seed per GPU
```

`parallel_runs` needs `python -m train.<dataset>` (the helper does this) and at least two visible GPUs. It is ignored when DDP is on.

## Logs and the dashboard

Paper A already has the dashboard in `SCI-JEPA-training` (`sci-jepa serve`, or `bash SCI-JEPA-training/scripts/present_dashboard.sh`). Graph-JEPA training writes into that same folder: `runs.json` plus `runs/<name>/live.json`. The loss curve is the **Training dynamics** chart (`train JEPA loss`). The Graphs tab includes the Graphify map of the repository when `graphify-out/graph.html` exists.

Each seed also keeps a local copy:

```
runs/<name or dataset-timestamp>/seed-<seed>/
  events.jsonl
  config.yaml
  summary.json
  tb/[latest|fold<k>]/  # TensorBoard: curves, HPARAMS tab
  csv/                  # with --loggers tensorboard,csv
  checkpoints/
```

`logging.loggers` (flag `--loggers`) picks Lightning loggers: `tensorboard` (default), `csv`, `wandb`, `mlflow`. W&B and MLflow are not installed by default (`uv pip install wandb` or `mlflow`); their project or experiment name is `logging.project`. Every logger receives the full flattened config as hyperparameters, and the probe results at the end of each fit as `final/*` metrics, so TensorBoard's HPARAMS tab can sort runs by `final/balanced_acc` or `final/mae`. Per-step loss is logged as `train_loss_step` every `trainer.log_every_n_steps` steps. Compare all runs at once:

```bash
uv run graph-jepa tensorboard        # --logdir runs/, port 6006 (PORT=... to change)
```

Open the dashboard:

```bash
uv run graph-jepa dashboard
# http://127.0.0.1:8765
```

Point a run at another dashboard folder with `SCIJEPA_DASHBOARD_ROOT` or `logging.dashboard_root`. The default folder is `SCI-JEPA-training/runs/paper_a_results/dashboard`. The Paper A command stays `sci-jepa`; Graph-JEPA training is `graph-jepa`, so the two entry points do not overwrite each other.

## Data paths

Datasets resolve to `Analysis/dataset/<name>` unless you override them.

| Variable | Meaning |
|---|---|
| `SCIJEPA_DATA_ROOT` | Parent of `papers/`, `ZINC/`, `TUD/`, `EXP/` |
| `cfg.data.root` | Full root for the dataset in this run |
| `SCIJEPA_CHECKPOINT_ROOT` | Reasoning-probe checkpoints |

The SLURM script `Analysis/JEPA-rea.sh` sets the first and the third from `PROJECT_DIR`.

## Code graph

Graphify builds a graph of the repository's functions, classes, files and imports, and groups them into named communities. [ARCHITECTURE.md](ARCHITECTURE.md#generated-views-graphify) lists the views; start with `graphify-out/ARCHITECTURE.html` (one diagram per project area) or `graphify-out/GRAPH_TREE.html` (collapsible tree).

One-time setup on each machine:

```bash
uv tool install "graphifyy[mcp]"
graphify hook install          # refresh graph.json after every commit, checkout and merge
```

Refresh names and views by hand:

```bash
bash scripts/graph-refresh.sh            # incremental update, name new communities, rebuild views
bash scripts/graph-refresh.sh --relabel  # rename every community (uses the claude CLI)
```

Coding agents are wired to query the graph before reading files:

| Agent | Instructions | Graph access |
|---|---|---|
| Claude Code | `CLAUDE.md` → `AGENTS.md` | PreToolUse hook in `.claude/settings.json`, MCP server in `.mcp.json`, `/graphify` skill |
| Codex (and any agent that reads `AGENTS.md`) | `AGENTS.md` | `.codex/hooks.json`, `.codex/skills/graphify` |
| Cursor | `.cursor/rules/graphify.mdc` | MCP server in `.cursor/mcp.json` |

All three MCP configs run `scripts/graphify-mcp.sh`. Restart the editor after changing them.
