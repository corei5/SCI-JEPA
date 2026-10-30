# SCI-JEPA

**Joint-embedding predictive learning over scientific knowledge units, and a benchmark for reasoning about papers (Paper A).**

A paper's structured summary is split into typed **knowledge units (KUs)**: context, research question, method, procedure, result, interpretation, limitation, claim, evidence and implication. Typed edges link them, e.g. a claim is *grounded in* a result and *supported by* evidence. SCI-JEPA trains a model that reads a paper and related papers with **one KU hidden** and **predicts the hidden KU's representation**. It is tested by ranking the true KU against on-topic distractors and against edited copies of itself, on five reasoning tasks: forward prediction, experiment design, abduction, consistency checking and intervention. Six comparison systems run on exactly the same queries: frozen text, BM25, SciNCL, BGE-M3, a supervised regressor and an LLM.

## Repository layout

```
SCI-JEPA/
├── src/sci_jepa/            the Python package (pip install -e .)
│   ├── config.py            every setting and every experiment preset (PaperAConfig, EXPERIMENTS)
│   ├── data/                parquet → KU graph → cached index → masked training / evaluation instances
│   ├── models/              the JEPA model (graph transformer / GNN encoders, predictor), losses, shapes
│   ├── training/            train_one, run_experiment (arms × seeds, resumable, shardable over GPUs)
│   ├── evaluation/          retrieval, reasoning tasks, comparison systems, validity checks, hypotheses
│   ├── reporting/           paper figures, PDF report, animated graph pages
│   ├── dashboard/           live training dashboard (page, server, run telemetry)
│   ├── inference.py         use a trained model: embed / predict / rank / consistency
│   ├── pipeline.py          the full notebook pipeline as a script (sci-jepa run)
│   ├── evaluate.py          standalone evaluation (sci-jepa evaluate)
│   ├── smoke_test.py        synthetic-data test of every component
│   └── cli.py               the `sci-jepa` command
├── notebooks/               paper_a_local.ipynb: the pipeline with a live dashboard
├── scripts/                 run_local.sh, check_environment.py, data/ (download, normalize; split: sci-jepa split)
├── cluster/grete/           Slurm: run_paper_a.sh (setup / submit / dist / tunnel), site.env, normalize.sbatch
├── docs/                    CODE_DOCUMENTATION.md (how the code works), paper_a.md (method), evaluation.md, data.md, cluster_grete.md
├── tests/                   fast package tests, normalizer tests, the slow smoke test
├── pyproject.toml           package metadata, dependencies, the `sci-jepa` command
└── requirements.txt         exact versions of the reported runs
```

## Install

Python ≥ 3.9 and PyTorch (CUDA, Apple GPU or CPU).

```bash
git clone <this repository> SCI-JEPA && cd SCI-JEPA
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm,notebook,dev]"          # or, for the exact versions of the reported runs:
                                               #   pip install -r requirements.txt && pip install -e .
sci-jepa fetch-assets                          # dashboard JS libraries, once (needed on machines without internet)
```

`bash scripts/run_local.sh setup` does all of this on a laptop or workstation and also registers a Jupyter kernel.

## Quick start

```bash
sci-jepa smoke-test                     # 1. everything works: synthetic data, CPU (prints [ok] per check)

bash scripts/data/download_scientific_summaries.sh       # 2. data: download + normalize (docs/data.md)
PARQUET_FOLDER=data/raw/data/arxiv OUT_PATH=data/normalized/arxiv python scripts/data/normalize_parquet.py

sci-jepa split --parquet data/normalized/arxiv --ku-fixes data/raw-KU-Fixes --out data/splits/d0   # 3. split once:
                                        #    train <= 2023, val 2024, test 2025-2026, same domain mix in every split
export PAPER_A_SPLIT=data/splits/d0     #    every later run uses it (no re-sampling, no re-splitting)

sci-jepa run --n-papers 500 --experiments           # 4. a small real run: reference model only, live dashboard
sci-jepa run                                        # 5. the notebook configuration: 500 papers, BENCH, 3 seeds

sci-jepa evaluate results runs/paper_a_results/BENCH   # 6. leaderboard + hypothesis tests of a finished experiment
```

`sci-jepa run` prints the dashboard address (`http://localhost:8765/`). Results, trained models and figures go to `runs/paper_a_results/`. On a Slurm cluster, use `cluster/grete/run_paper_a.sh` ([docs/cluster_grete.md](docs/cluster_grete.md)); it also splits a run over several GPUs and opens the dashboard in your local browser.

To share the dashboard with the team, it is also hosted on Vercel behind a team password. With `SCIJEPA_PUSH_URL` and `SCIJEPA_PUSH_TOKEN` set, everyrun  pushes its dashboard there while it trains (on Grete, `run_paper_a.sh submit` pushes from the login node instead). `scripts/import_logs.py` adds earlier runs from their SLURM logs. See [deploy/vercel/README.md](deploy/vercel/README.md).

Graph-JEPA pretraining in [`Analysis/`](../Analysis/README.md) uses the same dashboard. Those runs are Lightning, write `runs/<name>/live.json` into `runs/paper_a_results/dashboard`, and show up in All runs and in Training dynamics. Start them with `uv run graph-jepa train papers` from the repository root (the `sci-jepa` command stays this package).

## Configuration

`sci-jepa run` takes the parameters of the notebook's Parameters cell, as options or environment variables of the same name. The defaults are the notebook's:

| Parameter | Default | Meaning |
|---|---|---|
| `N_PAPERS` | 500 | papers in the corpus (500 = benchmark check, 5000 = results, 60000 = the paper's D0) |
| `SIZE` | local | model size: `local` (5 M parameters), `S` (the paper's, 46 M), `M` (~130 M) |
| `EPOCHS` / `PATIENCE` | 40 / 5 | maximum epochs; early stopping on the validation reasoning index |
| `SEEDS` | 0 1 2 | the paper reports 5 seeds |
| `EXPERIMENTS` (`EXPS`) | BENCH | BENCH, ALL, APPX_STRUCT, APPX_EVAL, SPEED, E1–E5, E10–E12, E15, E17–E20 ([docs/paper_a.md](docs/paper_a.md)) |
| `SPEED` | fast | fp16 + background batching (same accuracy); `fast+multi` loses accuracy; `off` |
| `LLM_MODEL` / `LLM_QUERIES` | Qwen/Qwen2.5-1.5B-Instruct / 300 | the LLM-Reasoning comparison system |
| `LLM_ON_MAC` | 1 | run LLM-Reasoning without a CUDA GPU (small models) |
| `FIELD_BALANCE` / `FIELD_ALPHA` | temperature / 0.5 | research-domain balance of the corpus |
| `OPEN_BROWSER` | 1 | open the live dashboard (cluster script) |
| `PAPER_A_SPLIT` (`--split-dir`) | – | a stored split (`sci-jepa split`); `N_PAPERS` then takes the same share of every split (0 = all) |

Data and output folders: `PAPER_A_SPLIT` (or `PAPER_A_PARQUET` without a stored split), `PAPER_A_INDEX_DIR` and `PAPER_A_OUT` ([docs/data.md](docs/data.md)). Without a stored split, papers are sampled and split by year on the fly (`time_T` = 2023: train ≤ 2023, val 2024, test ≥ 2025). Every other setting is a field of `PaperAConfig` in `src/sci_jepa/config.py`, and each experiment changes a few of them (`EXPERIMENTS` in the same file).

## Commands

| Command | What it does |
|---|---|
| `sci-jepa split --out DIR` | make the stored, domain-balanced train / val / test split, once ([docs/data.md](docs/data.md)) |
| `sci-jepa run` | full pipeline: KU graph, dashboard, reference run, experiments, findings, report |
| `sci-jepa evaluate model DIR` / `system NAME` / `results DIR` | evaluate a saved model, a comparison system, or a finished experiment ([docs/evaluation.md](docs/evaluation.md)) |
| `sci-jepa train --exp E10 --size S` | one experiment (all arms × seeds) without the notebook extras |
| `sci-jepa bench --data real --papers 300` | BENCH or ALL end to end, with the PDF report |
| `sci-jepa build-index --set max_papers=60000` | build and cache the KU graph |
| `sci-jepa validity audit` | counterfactual audit, faithfulness, annotation sheets, contamination |
| `sci-jepa findings` / `figures` / `report DIR` | hypotheses, paper figures, PDF of a finished experiment |
| `sci-jepa serve DIR` | serve a saved dashboard |
| `sci-jepa smoke-test` | synthetic-data test of every component |

`sci-jepa <command> --help` shows each command's options.

## Using a trained model

```python
from sci_jepa.inference import SciJEPA
m = SciJEPA.from_pretrained("<SCI-JEPA>/models/<run>/BENCH/reference_seed0", weights="best")   # PAPER_A_MODEL_DIR
paper = {"research_context": "...", "research_question_hypothesis": "...", "methodological_details": "...",
         "procedures_architectures": "...", "key_results": "...", "claims": [...], "field_subfield": "..."}
m.embed(paper)                                        # one vector per KU + a paper vector
m.rank(paper, hide="result", candidates=[...])        # which candidate is the most likely result
m.consistency(claim_text, evidence_text)              # P(supports), P(contradicts)
```

## Long runs on a cluster (e.g. 60,000 papers)

Training runs checkpoint and resume (`RESUME=1`), and `cluster/grete/submit_chain.sh` chains 12-hour jobs that continue each other. Each job records its hardware and measures GPUs, CPU and throughput every second (dashboard tab **Infrastructure (live)**). Every report ends with the compute per method in GPU-hours. See [docs/training_infrastructure.md](docs/training_infrastructure.md).

## Reproducibility

* Every run stores its full configuration, seed, data fingerprint and per-query scores. `sci-jepa evaluate model` re-evaluates a saved model with the training-time code and seeds and reproduces its test score exactly.
* Experiments resume: an arm × seed whose result file exists with the same settings is loaded instead of retrained. Splitting over GPUs (`--stage shard`, `cluster/grete … dist`) does not change any run.
* `requirements.txt` pins the versions used for the reported runs.

## Tests

```bash
python -m pytest -q -m "not slow"        # package, CLI and normalizer tests (seconds)
python -m pytest -q -m slow              # the full synthetic smoke test (minutes on a laptop, ~25 min on a cluster CPU slice)
```

## Documentation

* [docs/paper_a.md](docs/paper_a.md): the method, every experiment, the benchmark, validity checks and findings, in detail
* [docs/evaluation.md](docs/evaluation.md): how the evaluation works (masking, candidates, metrics, tasks)
* [docs/data.md](docs/data.md): download, normalize and split Scientific-Summaries; domain imbalance per split; known properties of the data
* [docs/cluster_grete.md](docs/cluster_grete.md): Slurm setup, several GPUs, the dashboard tunnel, troubleshooting
* [docs/training_infrastructure.md](docs/training_infrastructure.md): what every job records and measures (live every second), GPU-hours for the paper, checkpoints and chained jobs for long runs
* [MIGRATION.md](MIGRATION.md): where every file of the earlier `SCI-JEPA-training/Analysis` layout went
