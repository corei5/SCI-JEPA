# Agent guide

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the map of this repository. In short: `Analysis/` is Graph-JEPA (packages `core`, `train`, `diag`; command `graph-jepa`), and `SCI-JEPA-training/` is Paper A (package `sci_jepa`; command `sci-jepa`). They do not import each other.

## Commands

- Install: `uv sync --group dev` at the repository root. Paper A has its own `pyproject.toml` in `SCI-JEPA-training/`.
- Tests: `uv run pytest` at the root (Graph-JEPA infra); `cd SCI-JEPA-training && pytest` (Paper A).
- Train: `bash Analysis/run_local.sh <dataset> [overrides]`, or `cd Analysis && python -m train.<dataset>`. Modules are imported as `core.*`, `train.*` and `diag.*`, so Python must run from `Analysis/`.
- Dashboard: `uv run graph-jepa dashboard`, then open http://127.0.0.1:8765.

## Rules for this repository

- Keep paths stable. Collaborators run `Analysis/JEPA-rea*.sh` and `run_LLM_rebuttal*.sh` on a SLURM cluster and upload files through the GitHub web UI, so a renamed module breaks their jobs and comes back at the old path on the next upload. Propose moves; do not make them silently.
- `Analysis/train/paper_reason_gjepa_old_1.py` and `_old_2.py` are live libraries imported by `paper_reason_gjepa.py` and `paper_reason_todo_experiments.py`. Their names and docstrings disagree; see [Analysis/train/README.md](Analysis/train/README.md).
- Never commit data or outputs: `Analysis/dataset/` raw data, `runs/`, `log/`, `checkpoints/`, `*.pt` (apart from the caches already tracked), or TensorBoard files.
- Do not commit, push or open pull requests unless the user asks. Branch and pull-request conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).
- Hard-coded cluster paths (`/nfs/home/rabbyg/...`) live only in the SLURM scripts. Python code resolves paths through `core/paths.py` and the `SCIJEPA_DATA_ROOT` and `SCIJEPA_CHECKPOINT_ROOT` variables.
- Match the surrounding style. The research scripts have long banner docstrings that record experiment history; keep them when editing.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
