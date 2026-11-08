# Contributing

Four people work in this repository, some from a SLURM cluster and some through the GitHub web UI. These conventions keep `main` usable and keep everyone's work from overwriting everyone else's.

## Branches

`main` is the version that runs. It changes only through pull requests.

| Prefix | Use | Example |
|---|---|---|
| `feat/` | New code or infrastructure | `feat/lightning-engine` |
| `exp/` | An experiment or paper result that may never merge | `exp/euclidean-fix-rebuttal` |
| `fix/` | A bug fix | `fix/rwse-cache-path` |
| `docs/` | Documentation only | `docs/architecture-map` |

Long-lived topic branches already on the remote (`sci-jepa-benchmarking`, `kg-sci-jepa`, `multimodal-sci-jepa`) keep their names. Merge `main` into them, or rebase them onto `main`, before each pull request.

## Working from a terminal

```bash
git switch main && git pull
git switch -c feat/<short-name>
# edit, then commit in small steps that each say why
git push -u origin feat/<short-name>
gh pr create --base main
```

## Working from the GitHub web UI

"Add files via upload" commits straight to whatever branch is selected and carries no description, which makes history hard to follow.

1. Pick your branch in the branch selector before uploading, never `main`.
2. Replace the default "Add files via upload" message with a one-line summary of what changed.
3. Upload files to the path they already live at. A file uploaded next to its old location creates a second copy instead of replacing it.
4. Open a pull request into `main` from that branch.

## What not to commit

Data, outputs and machine-specific files are ignored in `.gitignore`: datasets (other than the small caches already tracked), `runs/`, `log/`, `checkpoints/`, `*.egg-info/`, `.venv/`. If `git status` shows one of these, leave it out of the commit.

## Pull requests

- Say what changed, why, and how you checked it (`uv run pytest`, a short training run, a figure).
- Keep moves and renames in their own pull request and update every script that references the old path. `Analysis/JEPA-rea*.sh` and `run_LLM_rebuttal*.sh` import modules by name.
- `graphify-out/` is gitignored; build it locally with `graphify update .`.

## Code graph

Install the CLI once with `uv tool install "graphifyy[mcp]"`, then `graphify hook install` to keep the graph current after each commit, checkout and merge. See [ARCHITECTURE.md](ARCHITECTURE.md) for the views.
