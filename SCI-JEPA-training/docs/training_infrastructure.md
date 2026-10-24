# Training infrastructure

Every job records what it ran on and measures, once per second, what it used. Every experiment's report then carries a hardware/software table, compute per method and resource timelines, ready for the paper's "Training infrastructure" paragraph, a table and an appendix figure. This page explains what is recorded, how it is computed, and how long runs survive the cluster's job time limits.

## What is recorded

| Record | Written by | When | Contents |
|---|---|---|---|
| `infra/<job>/system.json` | `sci_jepa.infra.sysinfo` | job start | host; physical GPUs (name, memory, MIG mode, power limit) and MIG slices (profile, memory); NVIDIA driver and CUDA driver; PyTorch, CUDA build, cuDNN; GPUs visible to PyTorch (compute capability, streaming multiprocessors); CPU model, allocated cores, node RAM; Slurm job, partition, account, time limit, start/end, allocated resources; package versions; storage |
| `infra/<job>/timeseries.csv` | `sci_jepa.infra.monitor` | every second | per MIG slice: memory in use; per physical GPU: power, temperature, SM clock, utilization (full GPUs only); job CPU use (% of its allocated cores), resident memory, node RAM; total training steps/s; runs training |
| `dashboard/infra/live.json` | monitor | every second | the latest sample and the last 10 minutes, for the dashboard's **Infrastructure (live)** tab |
| `infra/progress/<run>.json` | every training process (`train_one`) and LLM evaluation | about every 5 s | step, epoch, steps/s, instances/s, maximum time left, GPU memory allocated / peak (PyTorch), best epoch so far, parameters, batch size |
| `infra/plan.json` | pipeline (build stage) | job start | every arm × seed the experiment will run (the "runs finished" count) |
| `<EXP>/<arm>_seed<s>.json` | `run_experiment` | run end | the run's own timer (`train_sec`: training + its validation + final evaluation), epochs, steps, samples, parameters |

**MIG slices.** On Grete's `grete:interactive` partition a "GPU" is a MIG slice of an A100 80 GB (e.g. `3g.40gb` = 3 of 7 compute slices, 40 GB). NVIDIA's management library reports memory per slice but **no utilization per slice**. Power and temperature belong to the **whole physical GPU**, which other jobs' slices share. The report labels them that way and never attributes them to a run. Per-run GPU memory comes from PyTorch inside each process.

## How compute is counted

* **GPU-hours per run** = the run's own wall-clock timer (`train_sec`) on the one GPU or slice it used. For a JEPA model that includes its per-epoch validation and final test evaluation. For a comparison system it is its whole evaluation (an LLM's scoring of every candidate).
* **A100-equivalent hours** = GPU-hours × (slice compute fraction): `1g` = 1/7, `2g` = 2/7, `3g` = 3/7, full GPU = 1. It is reported only when all jobs used one slice type.
* **Allocated vs used:** a 2-GPU job holds both GPUs for its whole duration. The report counts the time each run actually trained, which is what the paper's compute statement should give.

## Outputs for the paper

Written for every finished experiment (`<results>/<EXP>/`):

| File | Use |
|---|---|
| `infrastructure.md` | readable summary: hardware/software table, compute per method, peaks and means of the per-second measurements |
| `infrastructure.tex` | two booktabs tables (`tab:infrastructure`, `tab:compute`) to `\input` in the paper |
| `infrastructure.json` | everything above, machine-readable |
| `figures/infra_compute.{png,pdf}` | GPU-hours and minutes per run, per method |
| `figures/infra_gpu_memory`, `infra_throughput`, `infra_gpu_power`, `infra_cpu_ram` `.{png,pdf}` | timelines over the whole experiment (all chained jobs) |
| `figures/fig13_llms.{png,pdf}` | the LLMs of different families against the best JEPA model and the strongest other system |
| `<EXP>_report.pdf` | the experiment report, now ending with "LLM-Reasoning across model families" and "Training infrastructure" pages |

All figures are written as 300-dpi PNG and as vector PDF, with editable text, in the style of the other paper figures. `sci-jepa figures <EXP dir>` rewrites them, and `python -m sci_jepa.infra.report <EXP dir>` rewrites the infrastructure files alone.

A sentence for the paper, filled from `infrastructure.md`: *"All models were trained on the GWDG Grete cluster on NVIDIA A100 80 GB GPUs partitioned into 40 GB MIG slices (3/7 of an A100), one slice per run, with PyTorch {version} and CUDA {version}; the benchmark took {N} runs and {X} GPU-hours ({Y} A100-equivalent hours) in total (Table~\ref{tab:compute})."*

## Long runs: checkpoints and chained jobs

`grete:interactive` jobs end after 12 hours, and each user may have 2 jobs, counting queued ones. At 60,000 papers one training run can take longer than one job. Two mechanisms make that work:

**Checkpoints (`RESUME=1`, `train_one(..., ckpt_path=...)`).** A training run writes a checkpoint every `CKPT_EVERY` steps (default 2,000), at every epoch end, and once more when training ends. It contains:

* the model with its EMA target encoder, the optimizer, the learning-rate schedule and the fp16 gradient scaler (none with bf16);
* every random-number state (Python, NumPy, PyTorch CPU and CUDA, the sampler's own generator);
* the sampler's exact position: each pool's shuffled order and pointer, the passes done, frozen neighbourhoods (E4);
* the loss history, per-epoch validation results, the best epoch's weights and the early-stopping state.

The file is written atomically, so a job killed while saving keeps the previous checkpoint. A job that starts a run finds its checkpoint (`<EXP>/checkpoints/<arm>_seed<s>.pt`) and continues after the saved step. It draws exactly the batches the uninterrupted run would have drawn, because the background batch builder's sampler state is captured at that step, not when the training loop reads it. With batch worker processes (`BATCH_WORKERS`), the anchors are still drawn in order in the main process and step s is built from its own random stream (seed, s), so the batches are the same for any number of workers and after a resume. `tests/test_resume.py` checks that a run stopped and resumed produces the same loss at every step and the same test score as a run that was never stopped. A checkpoint from different settings is ignored. The checkpoint is deleted once the run's result is saved.

**Chains (`cluster/grete/submit_chain.sh`).** A chain is a series of jobs. When a job starts it submits its successor, which waits until it ends, so at most two jobs exist at a time. Every job resumes all unfinished runs. The job that finishes the last run writes the summaries, findings, figures, infrastructure files and PDF, then cancels its successor. A job that runs out of time before every run is done only records what is missing (`infra/status.json`).

```bash
bash cluster/grete/submit_chain.sh 10 -- PAPER_A_OUT=/path/to/results_d60k SPLIT=/path/to/splits/d60k \
     N_PAPERS=0 SIZE=local EXPS=BENCH "LLM_MODELS=Qwen/Qwen3-8B,allenai/Olmo-3-7B-Instruct"
```

At most `CKPT_EVERY` steps of work are lost when a job ends, plus a comparison system's evaluation, which is not checkpointed. A 7B LLM's evaluation therefore has to fit in one job.
