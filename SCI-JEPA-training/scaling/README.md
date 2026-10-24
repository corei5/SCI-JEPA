# Scaling SCI-JEPA training to 60k papers

What limited training speed, what was changed, and the measured effect. All code, logs, index caches, models
and results of this work are in this folder; the data is read from the stored split
`/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/Scientific-Summaries/splits/arxiv_60k`.

## 1. Where the time went (before)

Profile on 2,000 real papers of the split, one A100 MIG slice (3g.40gb), model size `local`, batch 64,
the `fast` settings the runs used (fp16, one prefetch thread). `scaling/profile_training.py`,
log `runs/profile_before/profile.log`.

| arm | building a batch (CPU, 1 thread) | training step (GPU) | per-epoch validation |
|---|---|---|---|
| reference | 138 ms | 204 ms | 125 s |
| jepa+rgat | 138 ms | 205 ms | 126 s |
| jepa+gps | 138 ms | 227 ms | 130 s |
| paper_as_text | 175 ms | 165 ms | 125 s |

The operator trace of one step (`trace_step.py`, `runs/trace/`) showed **~70% of the GPU time in one
operation**: the backward pass of the relation-bias embedding (`embedding_dense_backward`). Every attention
layer looks up a learned bias for each of 64 x 189 x 189 = 2.3 million token pairs, and PyTorch computes that
gradient by sorting and reducing all 2.3 million positions into ~15 rows. The model itself is small
(5.2M trainable parameters, 1.8 GB on the GPU), so nothing else came close.

The batch is built in Python (citation neighbourhood + 8 papers x ~22 KUs per instance) in one background
thread that shares the interpreter lock with the training loop.

## 2. What was changed

| change | where | effect on the model / results |
|---|---|---|
| relation-bias gradient as one one-hot matrix product (same forward lookup) | `models/model.py: _RelBias` | none: identical forward values; gradient equal up to float summation order (max rel. diff 1e-4, `check_relbias.py`) |
| batches built in worker processes; anchors still drawn in order in the main process; step s builds from its own random stream (seed, s) | `training/train.py: _batches`, `data/instances.py: Sampler.draw / build`, config `batch_workers` | same model, objective and data distribution; batches independent of the number of workers; exact resume (tests) |
| neighbour scores cached per paper, subfield / field pools drawn without shuffling the whole group | `data/instances.py` | same distribution; time independent of group size (thousands of papers per subfield at 60k) |
| `SPEED=gpu` preset: bf16 (no loss scaler), fused attention (SDPA), TF32, fused AdamW | `config.py: FAST_GPU` | standard mixed-precision settings; accuracy check below |
| pinned memory + asynchronous host-to-GPU copies | `models/model.py: to_device` | none |
| several training processes per GPU (`SHARDS_PER_GPU`), batch workers per process (`BATCH_WORKERS`) | `cluster/grete/run_paper_a.sh dist` | none |
| dynamic sharding: each process claims the next free arm x seed run: quick comparison systems first (early results), then trained arms, then LLMs | `training/train.py: _claim`, `run_paper_a.sh` | none; GPUs finish at about the same time |

Defaults on Grete (`cluster/grete/site.env`): `SPEED=gpu`; in `dist` mode `BATCH_WORKERS` = the process's CPUs
minus one, at most 8. `SHARDS_PER_GPU=1`: after the changes a step keeps the slice busy, so a second process per slice
would only share it.

## 3. Measured effect (after)

Same 2,000 papers, same slice. `runs/profile_after/profile.log`.

| arm | step, old settings | step, `SPEED=gpu` | batch pipeline, 7 workers |
|---|---|---|---|
| reference | 204 ms | **66 ms** | 38 ms |
| jepa+rgat | 205 ms | **69 ms** | 44 ms |
| jepa+gps | 227 ms | **89 ms** | 46 ms |
| paper_as_text | 165 ms | **59 ms** | 46 ms |

The batch pipeline is now faster than the GPU step, so a step costs the GPU time alone.

Open point: the per-epoch validation took 146-159 s after the changes vs 125-130 s before (two profile jobs on
different nodes; validation builds its instances in the main process and is CPU-bound). Not yet isolated; at 60k it
is ~2% of an epoch (validation is fixed-size, an epoch is ~22 min of training).

**One epoch at 60k papers** (20,265 steps of 64; validation included):

| arm | before | after | speed-up |
|---|---|---|---|
| reference | ~1.2 h (GPU 204 ms/step, if batch building overlapped perfectly; up to 1.9 h) | **~0.42 h** | ~3x |
| jepa+gps | ~1.3 h (up to 2.1 h) | **~0.55 h** | ~2.5x |

## 4. Accuracy check (before vs after)

`accuracy_check.py`: 500 papers of the split, 10 epochs, seeds 0 1 2, arms reference and jepa+rgat; before =
the code as it was (`code_before/`, PYTHONPATH) with `SPEED=fast`; after = the current code with `SPEED=gpu` and
7 batch workers. Test split, mean ± standard deviation over the seeds. `runs/accuracy_{before,after}/check.json`.

| arm | balanced MRR before | after | reasoning index before | after |
|---|---|---|---|---|
| reference | 0.807 ± 0.032 | 0.845 ± 0.011 | 4.33 ± 0.14 | 4.47 ± 0.13 |
| jepa+rgat | 0.992 ± 0.009 | 0.987 ± 0.016 | 5.29 ± 0.18 | 5.35 ± 0.23 |

The differences are within the seed-to-seed spread: no loss of accuracy. (At 500 papers a run is only
1,440 steps and the per-epoch validation dominates its wall time, so this check is about accuracy; the speed
numbers come from the 2,000-paper profile above, where training steps dominate as they will at 60k.)

## 5. Files

| path | what |
|---|---|
| `profile_training.py`, `profile.sbatch` | speed profile (index, CPU batch, GPU step per setting, validation, batch pipeline) |
| `trace_step.py` | torch.profiler operator trace of one step (Chrome trace in `runs/trace*/`) |
| `check_relbias.py` | the relation-bias gradient: same values, timing |
| `accuracy_check.py`, `accuracy_check.sbatch` | before / after accuracy check |
| `after_checks.sh` | the GPU checks after the changes, in one job |
| `code_before/` | snapshot of `src/sci_jepa` before the changes (for the "before" runs) |
| `runs/` | index caches, results, models, traces, `profile.json` of every run |
| `logs/` | Slurm logs |
| `results/before_after.json` | all numbers above |
