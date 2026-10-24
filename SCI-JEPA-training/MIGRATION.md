# Migration from SCI-JEPA-training/Analysis

The code was reorganised into an installable package. **No algorithm changed:** 23 of the 25 modules are
identical apart from their import lines, and the other two (`config.py`, `evaluation/baselines.py`) changed only
machine-specific paths. Re-evaluating a saved model with the old and the new code gives identical results
(every metric of the evaluation output; only the measured run time differs). Existing KU index caches, finished experiment results and saved
models load unchanged.

## Where things went

| Before | Now | Import as |
|---|---|---|
| `train/paper_a/baselines.py` | `src/sci_jepa/evaluation/baselines.py` | `sci_jepa.evaluation.baselines` |
| `train/paper_a/bench_all.py` | `src/sci_jepa/training/bench_all.py` | `sci_jepa.training.bench_all` |
| `train/paper_a/config.py` | `src/sci_jepa/config.py` | `sci_jepa.config` |
| `train/paper_a/diagnostics.py` | `src/sci_jepa/evaluation/diagnostics.py` | `sci_jepa.evaluation.diagnostics` |
| `train/paper_a/figures.py` | `src/sci_jepa/reporting/figures.py` | `sci_jepa.reporting.figures` |
| `train/paper_a/findings.py` | `src/sci_jepa/evaluation/findings.py` | `sci_jepa.evaluation.findings` |
| `train/paper_a/graph_index.py` | `src/sci_jepa/data/graph_index.py` | `sci_jepa.data.graph_index` |
| `train/paper_a/graph_viz.py` | `src/sci_jepa/reporting/graph_viz.py` | `sci_jepa.reporting.graph_viz` |
| `train/paper_a/inference.py` | `src/sci_jepa/inference.py` | `sci_jepa.inference` |
| `train/paper_a/instances.py` | `src/sci_jepa/data/instances.py` | `sci_jepa.data.instances` |
| `train/paper_a/ku_graph.py` | `src/sci_jepa/data/ku_graph.py` | `sci_jepa.data.ku_graph` |
| `train/paper_a/ku_schema.py` | `src/sci_jepa/data/ku_schema.py` | `sci_jepa.data.ku_schema` |
| `train/paper_a/losses.py` | `src/sci_jepa/models/losses.py` | `sci_jepa.models.losses` |
| `train/paper_a/model.py` | `src/sci_jepa/models/model.py` | `sci_jepa.models.model` |
| `train/paper_a/neurons.py` | `src/sci_jepa/evaluation/neurons.py` | `sci_jepa.evaluation.neurons` |
| `train/paper_a/report_pdf.py` | `src/sci_jepa/reporting/report_pdf.py` | `sci_jepa.reporting.report_pdf` |
| `train/paper_a/serve.py` | `src/sci_jepa/dashboard/serve.py` | `sci_jepa.dashboard.serve` |
| `train/paper_a/shapes.py` | `src/sci_jepa/models/shapes.py` | `sci_jepa.models.shapes` |
| `train/paper_a/smoke_test.py` | `src/sci_jepa/smoke_test.py` | `sci_jepa.smoke_test` |
| `train/paper_a/synthetic.py` | `src/sci_jepa/data/synthetic.py` | `sci_jepa.data.synthetic` |
| `train/paper_a/tasks.py` | `src/sci_jepa/evaluation/tasks.py` | `sci_jepa.evaluation.tasks` |
| `train/paper_a/telemetry.py` | `src/sci_jepa/dashboard/telemetry.py` | `sci_jepa.dashboard.telemetry` |
| `train/paper_a/train.py` | `src/sci_jepa/training/train.py` | `sci_jepa.training.train` |
| `train/paper_a/validity.py` | `src/sci_jepa/evaluation/validity.py` | `sci_jepa.evaluation.validity` |
| `train/paper_a/worldview.py` | `src/sci_jepa/evaluation/worldview.py` | `sci_jepa.evaluation.worldview` |
| `train/paper_a/dashboard.html` | `src/sci_jepa/dashboard/dashboard.html` | |
| `train/paper_a/placeholder_shapes.json` | `src/sci_jepa/models/placeholder_shapes.json` | |
| `train/paper_a/README.md` | `docs/paper_a.md` | |
| `paper_a_run.py` | `src/sci_jepa/pipeline.py` | `sci-jepa run` |
| `evaluation/evaluate_paper_a.py` | `src/sci_jepa/evaluate.py` | `sci-jepa evaluate` |
| `evaluation/README.md` | `docs/evaluation.md` | |
| `server_check.py` | `scripts/check_environment.py` | |
| `run_paper_a.sh` (Grete) | `cluster/grete/run_paper_a.sh` + `cluster/grete/site.env` | |
| `run_paper_a_mac.sh` | `scripts/run_local.sh` | |
| `paper_a_local.ipynb` | `notebooks/paper_a_local.ipynb` (kernel "Python (sci-jepa)") | |
| `Alex/testing/parquet_to_json.py` | `scripts/data/normalize_parquet.py` | |
| `Alex/testing/test_normalized_json.py` | `tests/test_normalized_parquet.py` | |

## What changed for users

* `python -m train.paper_a.X` → `python -m sci_jepa.<subpackage>.X`, or the `sci-jepa <command>` CLI.
* Install once with `pip install -e .` instead of adding the folder to `PYTHONPATH`.
* Path defaults are relative to the working directory (`data/normalized/arxiv`, `runs/paper_a_index`,
  `runs/paper_a_results`) instead of `/nfs/home/rabbyg/...`; the `PAPER_A_*` environment variables work as before.
* Cluster settings (account, storage, partitions) live in `cluster/grete/site.env`, not in the job script.

## Changed after the migration (intended)

* **Split years:** `time_T` is 2023 (train ≤ 2023, validation 2024, test ≥ 2025) instead of 2021. Runs with the
  new default are not resumed from results made with 2021.
* **Stored split:** `sci-jepa split` (`src/sci_jepa/data/split.py`) makes a cleaned, domain-balanced
  train / val / test split once; `PAPER_A_SPLIT` / `--split-dir` makes training and benchmarking use it
  ([docs/data.md](docs/data.md)). Without it, papers are sampled and split on the fly as before.
