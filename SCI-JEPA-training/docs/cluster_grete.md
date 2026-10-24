# Running on a Slurm cluster (GWDG Grete)

`cluster/grete/run_paper_a.sh` runs everything from the login node. Site settings (account, storage,
Python, GPU types) live in `cluster/grete/site.env`; copy and edit that file for another Slurm cluster.

## Once

```bash
bash cluster/grete/run_paper_a.sh setup          # venv, packages, every model, dashboard assets (needs internet)
bash cluster/grete/run_paper_a.sh submit check   # GPU + real data + text encoder + dashboard (minutes)
bash cluster/grete/run_paper_a.sh submit smoke   # every component on synthetic data (~25 min)
```

Grete's compute nodes have no internet: `setup` downloads the text encoder, the comparison encoders
(SciNCL, BGE-M3, bge-base, MiniLM), the LLMs (Qwen2.5-1.5B and 7B) and the test model into `HF_HOME`, and
jobs run with `HF_HUB_OFFLINE=1`.

## Make the data split

```bash
SPLIT_ARGS="--parquet $STORE/Scientific-Summaries/normalized/arxiv,$STORE/Scientific-Summaries/normalized/pubmed \
  --ku-fixes $STORE/Scientific-Summaries-KU-Fixes --out $STORE/splits/d0 \
  --train-papers 60000 --val-papers 5000 --test-papers 5000" bash cluster/grete/run_paper_a.sh submit split
```

## Runs

```bash
SPLIT=$STORE/splits/d0 bash cluster/grete/run_paper_a.sh submit           # one GPU
SPLIT=$STORE/splits/d0 bash cluster/grete/run_paper_a.sh submit dist      # the job's GPUs, then merged
SPLIT=$STORE/splits/d0 N_PAPERS=0 SIZE=S SEEDS="0 1 2 3 4" EXPS="BENCH" bash cluster/grete/run_paper_a.sh submit dist
PARTITION=kisski GRES=gpu:A100:4 CPUS=64 TIME=2-00:00:00 ... submit dist  # 4 full A100s
bash cluster/grete/run_paper_a.sh tunnel [JOBID]                          # reconnect the browser
```

Every Parameters-cell setting of the notebook is an environment variable (`sci-jepa run --help`).

**`dist`** runs one job with N GPUs: a dashboard server; the KU graph built once; one process per GPU, each
training every N-th arm × seed (`run_experiment(shard=(i, N))`); a merge step that loads every finished run
and writes summaries, findings, leaderboards, figures and the PDF. Each arm × seed is trained exactly as on one
GPU, so splitting over GPUs does not change results. A run a failed shard left unfinished is trained in the
merge step; re-submitting resumes (finished runs are kept).

## Speed settings

On Grete `SPEED=gpu` is the default (`cluster/grete/site.env`): bf16, fused attention, TF32 and fused AdamW, and in
`dist` mode the training batches are built in `BATCH_WORKERS` processes per training process (default: its CPUs
minus one, at most 8). Together with a faster gradient for the relation biases this makes a training step about 3x faster on
an A100 slice, at the same accuracy. `SHARDS_PER_GPU` (default 1) puts several training processes on one GPU, and
every process claims the next unfinished arm x seed run: the quick comparison systems first, then the trained arms, then the LLMs. Measurements and the accuracy check:
[scaling/README.md](../scaling/README.md).

## Long experiments: chained 12-hour jobs

A run at 60,000 papers can take longer than one `grete:interactive` job (12 h). `cluster/grete/submit_chain.sh` starts a chain of jobs that continue each other from checkpoints. Each job submits its successor, so the 2-jobs-per-user limit is respected. The job that finishes the last run writes the reports and cancels the rest:

```bash
bash cluster/grete/submit_chain.sh 10 -- PAPER_A_OUT=$STORE/SCI-JEPA/d60k SPLIT=$STORE/Scientific-Summaries/splits/d60k \
     N_PAPERS=0 SIZE=local EXPS=BENCH
```

A chain holds both of your `grete:interactive` job slots. Details, and what each job records about the hardware: [training_infrastructure.md](training_infrastructure.md).

## Browser

`submit` and `tunnel` forward the job's dashboard port to `localhost:8765` (or the next free port) on the
login node. In VS Code (Remote-SSH) these settings open it in your browser automatically
(`~/.vscode-server/data/Machine/settings.json`):

```json
{
  "remote.autoForwardPorts": true,
  "remote.autoForwardPortsSource": "process",
  "remote.portsAttributes": { "8765-8799": { "label": "Paper A dashboard", "onAutoForward": "openBrowser" } }
}
```

Without VS Code: `ssh -N -L 8765:localhost:<port> <user>@<the same login node>` on your computer.

## Grete facts that matter

| Fact | Consequence |
|---|---|
| Account `kisski-l3s-llm4sci` may use `grete:interactive`, `kisski`, `kisski-h100` | `grete`, `grete-h100`, `scc-gpu` reject the account |
| `grete:interactive`: A100 MIG slices (1g.10gb, 2g.20gb, 3g.40gb), 12 h per job, **at most 2 jobs and 2 GPUs per user** (pending jobs count) | default for runs; starts within minutes |
| `kisski` (4 × A100 80 GB) and `kisski-h100` (4 × H100) | full GPUs, 2 days per job, but the queue can be days to weeks |
| Login nodes set `OMP_NUM_THREADS=1` and allow one CPU core | the job script resets threads to the job's cores; run heavy work in jobs |
| Login nodes are shared, ports can be taken | the tunnel picks the next free port |
| `/tmp` is per login node | keep working files on project storage |
| Compute nodes have no internet | `setup` downloads every model; jobs run offline |
