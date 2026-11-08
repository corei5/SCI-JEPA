"""Paper A pipeline: the notebook (notebooks/paper_a_local.ipynb) as a script, for GPU servers.

Same pipeline as the notebook, same ICML configuration:
  1. build (or load) the KU graph from the normalized parquet
  2. look at the data (KU counts, cleaning outcomes, years / fields, one paper)
  3. live dashboard + animated graphs
  4. shape audit (Theorem A2, no training)
  5. one training run, streamed live to the dashboard
  6. experiments (BENCH by default), validity checks, findings, leaderboards
  7. a trained model through the inference API

Differences from the notebook (it runs headless on a GPU node):
  - the Parameters cell is command-line options (or env vars of the same name)
  - tables are printed, figures are saved as PNG under PAPER_A_OUT/figures_run/
  - the dashboard starts before the KU graph build, on the first free port from --port, and writes
    <PAPER_A_OUT>/dashboard/server_<job>.json (on a cluster, cluster/grete/run_paper_a.sh tunnel reads it
    and forwards the port to your computer)
  - the dashboard keeps serving after training until --keep-serving minutes have passed (0 = exit)

    sci-jepa run --n-papers 5000 --size S --experiments BENCH --seeds 0 1 2
    python -m sci_jepa.pipeline ...                        # the same without installing the CLI

Paths (environment): PAPER_A_PARQUET (default: $PAPER_A_DATA_ROOT/normalized/<dataset>, data root default
./data), PAPER_A_INDEX_DIR (./runs/paper_a_index), PAPER_A_OUT (./runs/paper_a_results).
"""
import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PAPER_A_DATA_ROOT", "data"))   # holds normalized/<dataset>/*.parquet


def _env(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else type(default)(v)


def _bool(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v.strip().lower() in ("1", "true", "yes", "on")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- the notebook's Parameters cell (same names as env vars, same defaults as paper_a_local.ipynb) ---
    p.add_argument("--n-papers", type=int, default=_env("N_PAPERS", 500),
                   help="N_PAPERS: 500 = local benchmark run, 5000 = local results; the paper's D0 = 60000")
    p.add_argument("--size", default=_env("SIZE", "local"),
                   help='SIZE: "local" = small (d 256, 4+2 layers, 5 M) | "S" = the paper\'s model (d 512, 8+6, 46 M) '
                        '| "M" = medium (d 768, 12+8, ~130 M)')
    p.add_argument("--epochs", type=int, default=_env("EPOCHS", 40),
                   help="EPOCHS: MAXIMUM passes over every hidden KU; early stopping on the validation reasoning index")
    p.add_argument("--patience", type=int, default=_env("PATIENCE", 5),
                   help="PATIENCE: epochs without improvement before a run stops (0 = run all EPOCHS)")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[int(s) for s in _env("SEEDS", "0 1 2").replace(",", " ").strip("[]").split()],
                   help="SEEDS: the paper reports 5 seeds: 0 1 2 3 4")
    p.add_argument("--experiments", nargs="*", default=os.environ.get("EXPS", os.environ.get("EXPERIMENTS", "BENCH")).split(),
                   help="EXPERIMENTS (env EXPS): BENCH | ALL | APPX_STRUCT | APPX_EVAL | SPEED | E1-E5, E10-E12, E15, "
                        "E17-E20 (none = only the reference run)")
    p.add_argument("--speed", default=_env("SPEED", "fast"), choices=["fast", "gpu", "fast+multi", "off"],
                   help='SPEED: "fast" = same model and accuracy, faster | "gpu" = fast + bf16, fused attention, TF32 '
                        'and fused AdamW (A100 / H100) | "fast+multi" loses accuracy | "off"')
    p.add_argument("--batch-workers", type=int, default=_env("BATCH_WORKERS", 0),
                   help="BATCH_WORKERS: processes that build the training batches (0 = one background thread)")
    p.add_argument("--llm-model", default=_env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                   help="LLM_MODEL: LLM-Reasoning baseline (any Hugging Face causal LM, downloaded once with internet access)")
    p.add_argument("--llm-on-mac", type=int, choices=[0, 1], default=int(_bool("LLM_ON_MAC", True)),
                   help="LLM_ON_MAC: 1 = run LLM-Reasoning even without a CUDA GPU (on the GPU nodes it always runs on CUDA)")
    p.add_argument("--llm-queries", type=int, default=_env("LLM_QUERIES", 300),
                   help="LLM_QUERIES: queries per retrieval task for the LLM (each ranks ~100 candidates)")
    p.add_argument("--field-balance", default=_env("FIELD_BALANCE", "temperature"), choices=["temperature", "uniform", "none"],
                   help="FIELD_BALANCE: research-domain balance of the corpus")
    p.add_argument("--field-alpha", type=float, default=_env("FIELD_ALPHA", 0.5),
                   help="FIELD_ALPHA: 1 = natural shares, 0.5 = square root, 0 = equal")
    p.add_argument("--open-browser", type=int, choices=[0, 1], default=int(_bool("OPEN_BROWSER", True)),
                   help="OPEN_BROWSER: 1 = the cluster submit script opens the live dashboard in your browser (via a tunnel)")
    p.add_argument("--dataset", default=_env("DATASET", "arxiv"), choices=["arxiv", "pubmed"],
                   help="normalized subset under Scientific-Summaries/normalized/ (Paper A was built on arxiv)")
    p.add_argument("--split-dir", default=_env("PAPER_A_SPLIT", ""),
                   help="stored train / val / test split (sci-jepa split): used as it is, never re-sampled or re-split; "
                        "--n-papers then takes the same share of every split (0 = all)")
    # --- several GPUs: the same runs, split over processes (each arm x seed is trained exactly as on one GPU) ---
    p.add_argument("--llm-models", nargs="*", default=os.environ.get("LLM_MODELS", "").replace(",", " ").split(),
                   help="LLM_MODELS: several LLM-Reasoning arms, one per Hugging Face model id (replaces --llm-model "
                        "in BENCH): e.g. Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--arms", nargs="*", default=os.environ.get("ARMS", "").replace(",", " ").split(),
                   help="ARMS: run only these arms of each experiment (e.g. reference jepa+rgat world_model motif "
                        "motif+world); default: every arm")
    p.add_argument("--live-evals", type=int, default=_env("LIVE_EVALS", 10),
                   help="LIVE_EVALS: quick validation passes per run for the dashboard's curves (0 = none; the epoch-end "
                        "validation that selects the best epoch always runs). Each costs minutes at 2M papers")
    # --- long runs: resume across jobs, live infrastructure monitor ---
    p.add_argument("--resume", type=int, choices=[0, 1], default=int(_bool("RESUME", False)),
                   help="RESUME: 1 = every training run checkpoints and a later job continues it (chained jobs)")
    p.add_argument("--ckpt-every", type=int, default=_env("CKPT_EVERY", 2000),
                   help="CKPT_EVERY: with --resume, also checkpoint every this many steps (0 = epoch ends only)")
    p.add_argument("--keep-ckpts", type=int, default=_env("KEEP_CKPTS", 10),
                   help="KEEP_CKPTS: keep the weights at this many evenly spaced steps of every training run, in its "
                        "model folder under checkpoints/ (0 = none)")
    p.add_argument("--monitor-interval", type=float, default=_env("MONITOR_INTERVAL", 1.0),
                   help="MONITOR_INTERVAL: seconds between infrastructure samples (0 = no monitor)")
    p.add_argument("--stage", default=_env("STAGE", "all"), choices=["all", "build", "shard", "merge", "serve"],
                   help="all = everything in one process | build = KU graph, graphs, shape audit | shard = this "
                        "process's share of the runs (--shard i/n) | merge = load every finished run, write summaries, "
                        "leaderboards, report | serve = only the dashboard server (one per job)")
    p.add_argument("--shard", default=_env("SHARD", ""), help="i/n for --stage shard")
    # --- server-only knobs ---
    p.add_argument("--port", type=int, default=_env("PORT", 8765), help="first port to try for the dashboard")
    p.add_argument("--keep-serving", type=int, default=_env("KEEP_SERVING", 30),
                   help="minutes to keep the dashboard up after the run (inside the job's time limit)")
    p.add_argument("--skip-single", action="store_true", default=bool(_env("SKIP_SINGLE", 0)),
                   help="skip section 5 (the single reference run)")
    p.add_argument("--skip-validity", action="store_true", default=bool(_env("SKIP_VALIDITY", 0)),
                   help="skip section 6a (surface-cue audit, faithfulness, annotation export)")
    p.add_argument("--skip-inference", action="store_true", default=bool(_env("SKIP_INFERENCE", 0)),
                   help="skip section 6b (inference API demo)")
    return p.parse_args()


A = parse_args()
SHARD = tuple(int(x) for x in A.shard.split("/")) if A.stage == "shard" else None
assert A.stage != "shard" or (SHARD and len(SHARD) == 2 and 0 <= SHARD[0] < SHARD[1]), "--stage shard needs --shard i/n"
DO_GRAPHS = A.stage in ("all", "build")                                   # sections 2b-4: data views, graphs, audit
DO_SINGLE = not A.skip_single and (A.stage == "all" or (SHARD is not None and SHARD[0] == 0))
DO_EXPS = A.stage in ("all", "shard", "merge")
DO_REPORT = A.stage in ("all", "merge")                                   # validity, leaderboards, figures, inference

os.environ.setdefault("PAPER_A_PARQUET", str(DATA_ROOT / "normalized" / A.dataset))
if A.split_dir:
    os.environ["PAPER_A_SPLIT"] = A.split_dir
os.environ.setdefault("PAPER_A_INDEX_DIR", "runs/paper_a_index")
os.environ.setdefault("PAPER_A_OUT", "runs/paper_a_results")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")                                  # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sci_jepa.config import (PaperAConfig, SIZES, EXPERIMENTS as EXP_ARMS, ICML_FIXED,
                             PAPER_EXPERIMENTS, resolve_device, FAST, FAST_GPU, FAST_MULTI)
from sci_jepa.data import ku_schema
from sci_jepa.data.graph_index import get_index, HUB
from sci_jepa.data.instances import Sampler
from sci_jepa.models.shapes import SlotLabels, audit_tightness
from sci_jepa.training.train import train_one, run_experiment
from sci_jepa.dashboard.telemetry import LiveRun
from sci_jepa.reporting import graph_viz as V
from sci_jepa.dashboard.serve import setup_dashboard, start_server, register_graph

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)
OUT = Path(os.environ["PAPER_A_OUT"])
MODELS = Path(os.environ.get("PAPER_A_MODEL_DIR") or OUT / "models")   # trained models (site.env: SCI-JEPA/models/<run>)
FIG = OUT / "figures_run"
FIG.mkdir(parents=True, exist_ok=True)


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100, flush=True)


def display(obj):
    """Notebook display() for a terminal / Slurm log."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string(), flush=True)
    else:
        print(obj, flush=True)


def savefig(name):
    path = FIG / f"{name}.png"
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close("all")
    print("figure:", path, flush=True)


def free_port(start):
    for port in range(start, start + 100):
        with socket.socket() as s:
            try:
                s.bind(("", port)); return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in {start}..{start + 99}")


if A.split_dir:
    assert Path(A.split_dir, "manifest.json").is_file(), f"no stored split in {A.split_dir} (create it: sci-jepa split)"
else:
    assert Path(os.environ["PAPER_A_PARQUET"]).is_dir(), \
        f"parquet folder not found: {os.environ['PAPER_A_PARQUET']} (run the normalization first)"
print("torch", torch.__version__, "| device ->", resolve_device("auto"),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")
print("data    :", f"stored split {A.split_dir}" if A.split_dir else os.environ["PAPER_A_PARQUET"])
print("results :", OUT)

def heal_runs_index(dash):
    """runs.json is read-modify-written by every LiveRun; with several processes an entry can be lost when two
    runs register at the same moment. Re-add every run folder that has a live.json (atomic replace)."""
    idx = Path(dash, "runs.json")
    try:
        runs = json.loads(idx.read_text()) if idx.exists() else []
    except (json.JSONDecodeError, OSError):
        return
    have = {r.get("name") for r in runs}
    new = [{"name": d.name, "started": json.loads((d / "live.json").read_text()).get("started", ""),
            "path": f"runs/{d.name}/live.json"}
           for d in sorted(Path(dash, "runs").glob("*")) if d.name not in have and (d / "live.json").exists()]
    if new:
        tmp = idx.with_suffix(".heal.tmp")
        tmp.write_text(json.dumps(runs + new)); os.replace(tmp, idx)


# the dashboard starts first, so the browser opens as soon as the job runs (graphs and runs appear as they come).
# With several GPUs one `serve` process per job serves the shared dashboard folder; the shards only write to it.
DASH = os.path.join(OUT, "dashboard")
if A.stage in ("all", "serve"):
    # serve.py downloads two JS files without a timeout, and cluster nodes often have no internet: copy them
    # from PAPER_A_VENDOR (or the package's dashboard/vendor/, filled by `sci-jepa fetch-assets`), and fail
    # fast otherwise (the pages then load them from the CDN)
    VENDOR = Path(os.environ.get("PAPER_A_VENDOR", PACKAGE / "dashboard" / "vendor"))
    for sub in ("vendor", "graphs/vendor"):
        dst = Path(DASH, sub); dst.mkdir(parents=True, exist_ok=True)
        for f in VENDOR.glob("*.js"):
            if not (dst / f.name).exists() or (dst / f.name).stat().st_size < 10_000:
                shutil.copy(f, dst / f.name)
    socket.setdefaulttimeout(10)
    setup_dashboard(DASH)
    socket.setdefaulttimeout(None)
    URL = start_server(DASH, free_port(A.port))          # serve.py binds 127.0.0.1; the tunnel goes through ssh
    PORT = int(URL.rstrip("/").rsplit(":", 1)[1])         # start_server may move on to the next free port
    NODE = socket.gethostname()
    SERVER = {"node": NODE, "port": PORT, "url_on_node": URL, "job": os.environ.get("SLURM_JOB_ID"),
              "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name in ("server.json", f"server_{SERVER['job']}.json"):
        Path(DASH, name).write_text(json.dumps(SERVER, indent=1))
    print(f"dashboard on {NODE}:{PORT}  (wrote {DASH}/server.json)")
    print(f"open {URL} (on a cluster: cluster/grete/run_paper_a.sh tunnel forwards it to your computer)", flush=True)
    if A.monitor_interval > 0:                           # GPUs / CPU / progress every second (dashboard: Infrastructure)
        from sci_jepa.infra.monitor import start as start_monitor
        MONITOR = start_monitor(str(OUT), A.monitor_interval)
    # hosted dashboard from the job's first minute (KU graph build included), not only once a run trains; the
    # shards' LiveRuns share the folder's .autopush.lock, so one process uploads
    from sci_jepa.dashboard.push import start_autopush
    start_autopush(DASH)
elif A.stage == "shard" and A.monitor_interval > 0:
    # every job reports on the Infrastructure page, also the jobs without a server that share another job's results
    # folder (one per node / GPU type): the first shard of each job that gets its lock monitors it
    from sci_jepa.infra.monitor import start as start_monitor
    MONITOR = start_monitor(str(OUT), A.monitor_interval)
if A.stage == "serve":                                     # runs until the job script stops it
    while True:
        heal_runs_index(DASH)
        time.sleep(20)

# ------------------------------------------------------------------ 1. the ICML configuration
section("1. The ICML configuration")
bad = [e for e in A.experiments if e not in PAPER_EXPERIMENTS]
assert not bad, f"not an ICML experiment: {bad}; choose from {list(PAPER_EXPERIMENTS)}"
_known = {t for e in A.experiments for t, _ in EXP_ARMS[e]}
assert all(a in _known for a in A.arms), \
    f"ARMS not in {A.experiments}: {[a for a in A.arms if a not in _known]}; choose from {sorted(_known)}"
on_gpu = torch.cuda.is_available()
cfg = PaperAConfig().override(**SIZES[A.size], max_papers=A.n_papers, split_dir=A.split_dir,
                              epochs=float(A.epochs), patience=A.patience,
                              seeds=list(A.seeds), device="auto", batch_size=64 if A.size in ("tiny", "local") else 128,
                              rows_per_shard=100 if A.n_papers <= 20000 else 500,
                              llm_model=A.llm_model, llm_allow_cpu=bool(A.llm_on_mac), llm_eval_instances=A.llm_queries,
                              llm_models=tuple(A.llm_models), arms=tuple(A.arms),
                              resume_runs=bool(A.resume), ckpt_every=A.ckpt_every, keep_ckpts=A.keep_ckpts,
                              field_balance=A.field_balance, field_alpha=A.field_alpha,
                              batch_workers=A.batch_workers,
                              **{"off": {}, "fast": FAST, "gpu": FAST_GPU, "fast+multi": FAST_MULTI}[A.speed])
display(pd.Series({**ICML_FIXED, "papers": f"{A.n_papers or 'all'} from " + (f"stored split {A.split_dir}" if A.split_dir else A.dataset)
                   + f" (model size {A.size}, {len(A.seeds)} seeds)",
                   "speed": f"{A.speed}: amp={cfg.amp}, prefetch={cfg.prefetch}, targets/instance={cfg.targets_per_instance}, "
                            f"text encoder {cfg.embed_dtype}, padding {cfg.embed_bucket}, batch workers {cfg.batch_workers}"}).to_frame("fixed configuration"))
display(pd.DataFrame({"question": PAPER_EXPERIMENTS, "arms": {k: ", ".join(t for t, _ in EXP_ARMS[k]) for k in PAPER_EXPERIMENTS},
                      "run now": {k: "yes" if k in A.experiments else "" for k in PAPER_EXPERIMENTS}}))

# ------------------------------------------------------------------ 2. KU graph
section("2. Build (or load) the KU graph")
t0 = time.time()
gi = get_index(cfg)
S = gi.summary()
print(f"{S['papers']} papers, {S['kus']} KUs ({S['kus_per_paper']} per paper) in {time.time() - t0:.0f}s")
print("splits:", S["splits"], "| fields:", S["fields"], "| subfields:", S["subfields"])
print("in-corpus citations:", S["paper_edges_cites"], "| proxy grounded edges:", S["paper_edges_proxy"])
from sci_jepa.training.train import mask_plan      # run length from the data: EPOCHS passes over every hidden KU
n_steps, _, sched = mask_plan(cfg, Sampler(gi, cfg, "train", seed=0))
cfg = cfg.override(steps=n_steps, log_every=max(n_steps // 50, 1),
                   eval_every=max(n_steps // A.live_evals, 1) if A.live_evals > 0 else 0,   # dashboard only
                   # warm-up: one epoch (at least 1,000 steps), at most 10% of the run
                   warmup=min(max(1000, int(n_steps / max(A.epochs, 1e-9))), max(n_steps // 10, 1)))
print(f"at most {A.epochs} epochs over {sum(sched['anchors_per_type'].values())} hidden KUs = {n_steps} steps of {cfg.batch_size}"
      + (f" (early stopping after {A.patience} epochs without improvement)" if A.patience else ""))
if n_steps < 1000:
    print("WARNING: fewer than 1,000 training steps -- too few for the predictor to leave the average; "
          "use more papers or epochs before reading the results")
display(pd.DataFrame({"KUs": S["per_type"], "per paper": {k: round(v / S["papers"], 2) for k, v in S["per_type"].items()}}))

# sections 2b-4 once per run: the build stage (several GPUs) or the single process (one GPU)
if DO_GRAPHS:
    print("\nCleaning outcomes per KU field:")
    out = pd.Series(S["build"]["ku_outcomes"]).rename("count").to_frame()
    out.index = pd.MultiIndex.from_tuples([tuple(i.split(":")) for i in out.index], names=["field", "outcome"])
    tab = out["count"].unstack(fill_value=0)
    tab["ok %"] = (100 * tab.get("ok", 0) / tab.sum(axis=1)).round(1)
    display(tab)

    yrs = pd.Series(gi.paper_year[gi.paper_year > 0]).value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    yrs.plot.bar(ax=ax[0], title="papers per year"); ax[0].axvline(list(yrs.index).index(cfg.time_T) if cfg.time_T in yrs.index else 0, c="r", ls="--")
    pd.Series([gi.field_names[f] for f in gi.paper_field]).value_counts().head(15).plot.barh(ax=ax[1], title="top fields")
    savefig("years_fields")

    print("\nOne paper's knowledge units:")
    p = int(gi.papers("train")[0])
    rows = [{"type": ku_schema.COMP_TYPES[gi.comp_type[c]],
             "quantitative": bool(gi.comp_quantitative[c]), "cites_figure": bool(gi.comp_cites_figure[c]),
             "challenged": bool(gi.comp_challenged[c])} for c in gi.paper_comps[p]]
    print("paper:", gi.paper_ids[p], "| year", gi.paper_year[p], "| field", gi.field_names[gi.paper_field[p]],
          "| type", ku_schema.PAPER_TYPES[gi.paper_type[p]])
    edges = [(ku_schema.COMP_TYPES[gi.comp_type[c]], ku_schema.ROLE_RELS[r - 3], ku_schema.COMP_TYPES[gi.comp_type[d]])
             for c in gi.paper_comps[p] for d, r in gi.comp_edges.get(int(c), []) if 3 <= r < 3 + len(ku_schema.ROLE_RELS)]
    print("role edges:", pd.Series(edges).value_counts().to_dict())
    display(pd.DataFrame(rows).value_counts("type").to_frame("KUs"))

    # ------------------------------------------------------------------ 3. dashboard
    section("3. Live dashboard and animated graphs")
    G = os.path.join(DASH, "graphs")
    p_show = max(gi.papers("train")[:500], key=lambda p: len(gi.paper_comps[int(p)]))
    V.paper_html(gi, int(p_show), f"{G}/paper_subgraph.html")
    register_graph(DASH, f"Paper subgraph ({gi.paper_ids[int(p_show)]})", "graphs/paper_subgraph.html")
    for t in cfg.target_types:
        inst = Sampler(gi, cfg.override(target_types=(t,)), "train", seed=1).sample(1)[0]
        V.instance_html(gi, inst, cfg, f"{G}/instance_{t}.html")
        register_graph(DASH, f"Training instance: masked {t} (N={cfg.n_subgraphs})", f"graphs/instance_{t}.html")
    V.corpus_html(gi, f"{G}/corpus_papers.html", n_papers=150, split="train")
    register_graph(DASH, "Corpus graph: 150 papers", "graphs/corpus_papers.html")
    V.corpus_html(gi, f"{G}/corpus_kus.html", n_papers=30, ku_level=True, split="train")
    register_graph(DASH, "Corpus graph: 30 papers with their KUs", "graphs/corpus_kus.html")
    print("graphs:", G)

    # ------------------------------------------------------------------ 4. shape audit
    section("4. Shape audit (Theorem A2, no training)")
    display(pd.DataFrame(audit_tightness(cfg.shape_spec, SlotLabels(gi), gi)))

if A.stage in ("build", "all"):                       # every arm x seed this job set will run (infra.html)
    from sci_jepa.training.train import _skipped, expand_llm_arms
    plan = {e: [f"{t}_seed{s}" for t, o in expand_llm_arms(EXP_ARMS[e], cfg) if not _skipped(cfg.override(**o), o)
                for s in cfg.override(**o).seeds] for e in A.experiments}
    Path(OUT, "infra").mkdir(parents=True, exist_ok=True)
    Path(OUT, "infra", "plan.json").write_text(json.dumps({"experiments": plan, "written": time.strftime("%Y-%m-%d %H:%M:%S")}))
if A.stage == "build":
    print("\nbuild stage done: KU graph cached in", os.environ["PAPER_A_INDEX_DIR"], flush=True)
    sys.exit(0)


def domain_balance_table(b):
    """Candidates and selected papers per research domain. A stored split (sci-jepa split) records them per split
    ({split: {domain: n}}), a corpus balanced at build time once ({domain: n})."""
    pool, sel = b.get("pool_by_domain") or {}, b.get("selected_by_domain") or {}
    if any(isinstance(v, dict) for v in pool.values()):
        cols = {}
        for s in pool:
            cols[f"{s} candidates"] = pool[s]
            cols[f"{s} selected"] = sel.get(s, {})
        return pd.DataFrame(cols).fillna(0).astype(int)
    return pd.DataFrame({"candidates": pool, "selected": sel}).fillna(0).astype(int)


def flat(m):
    def g(*path):                                  # missing values (e.g. the no-training baseline) -> NaN
        d = m
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return float("nan") if d is None else d
    return {"balanced MRR": g("retrieval_balanced", "MRR"), "balanced chance": g("retrieval_balanced", "chance_MRR"),
            "balanced Hits@10": g("retrieval_balanced", "Hits@10"),
            "standard MRR": g("retrieval_standard", "MRR"), "standard chance": g("chance_MRR_standard"),
            "shortcut margin (MRR)": g("shortcut_margin_MRR", "mean"),
            "chi (target, type-centred)": g("target_space", "chi"), "chi raw": g("target_space", "chi_raw"),
            "eff. rank (target)": g("target_space", "eff_rank"),
            "partition probe acc": g("partition_probe", "probe_acc"),
            "partition probe chance": g("partition_probe", "probe_chance"),
            "between-subgraph var share": g("partition_probe", "between_subgraph_var_share"),
            "relation sensitivity": g("relation_sensitivity"), "inverse-relation acc": g("inverse_acc"),
            "audit: flagged frac": g("audit", "flagged_frac"), "audit: mean max ctx sim": g("audit", "mean_max_ctx_sim"),
            "train sec": g("train_sec"), "trainable params": g("params_trainable")}


def epoch_table(m):
    rows = []
    for r in m.get("epoch_results", []):
        rows.append({"epoch": r["epoch"], "train JEPA": r["train_jepa"], "val balanced MRR": r["bal_MRR"],
                     "chance": r["bal_chance"], "x chance": r["bal_MRR"] / max(r["bal_chance"], 1e-9),
                     "Hits@10": r["bal_H10"], f"acc (sim >= {cfg.sim_threshold:.2f})": r.get("acc_sim"),
                     "chi": r["chi"], "probe": r["probe_acc"],
                     "types learned": f"{r['types_learned']}/{len(r['per_type'])}",
                     "identity neurons": (r.get("neurons") or {}).get("identity"),
                     "predictor dims learned": (r.get("neurons") or {}).get("pred_learned"), "sec": r["sec"]})
    return pd.DataFrame(rows).set_index("epoch") if rows else pd.DataFrame()


def epoch_types(m):
    R = m.get("epoch_results", [])
    return pd.DataFrame({f"epoch {r['epoch']}": {t: v["MRR"] / max(v["chance"], 1e-9) for t, v in r["per_type"].items()}
                         for r in R}).round(2)


from sci_jepa.evaluation.tasks import TASK_ORDER, TASK_LABEL, task_summary


def task_table(tasks):
    if not tasks or "note" in tasks:
        return pd.DataFrame({"note": [tasks.get("note", "no task results") if tasks else "no task results"]})
    rows = {}
    for k in ("forward", "inverse", "abduction"):
        t = tasks[k]
        rows[TASK_LABEL[k]] = {"query": t["query"], "score": t.get("ratio"), "metric": "MRR / chance",
                               "MRR": t.get("MRR"), "chance": t.get("chance"), "Hits@10": t.get("Hits@10"),
                               f"acc (sim >= {cfg.sim_threshold:.2f})": t.get("acc_sim"), "n": t["n"],
                               "by target": ", ".join(f"{g} {b['ratio']:.2f}x" for g, b in t.get("by_target", {}).items())}
    c = tasks["consistency"]
    rows[TASK_LABEL["consistency"]] = {"query": c["query"], "score": c.get("bal_acc"), "metric": "balanced acc.",
                                       "chance": 0.5, "n": c["n"],
                                       "by target": f"supports recall {c.get('recall_supports')}, contradicts recall {c.get('recall_contradicts')}"}
    iv = tasks["intervention"]
    for name in ("condition", "method"):
        v = iv[name]
        rows[f"{TASK_LABEL['intervention']} ({name})"] = {"query": iv["query"], "score": v.get("rel_shift"),
                                                          "metric": "relative shift", "n": v["n"],
                                                          "by target": f"toward donor {v.get('toward_donor')}, away from own {v.get('away_from_own')}"}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ 5. one training run
MODEL_DIR = None
SINGLE_CKPT = None
if DO_SINGLE and cfg.resume_runs:                       # chained jobs: a stable name, so a later job resumes or skips it
    RUN_NAME = f"single__N{cfg.n_subgraphs}__seed{A.seeds[0]}"
    SINGLE_CKPT = os.path.join(OUT, "checkpoints", RUN_NAME + ".pt")
    if os.path.exists(os.path.join(MODELS, "single", RUN_NAME, "summary.json")):
        MODEL_DIR = os.path.join(MODELS, "single", RUN_NAME)
        print(f"reference run already finished ({MODEL_DIR}): not trained again", flush=True)
        DO_SINGLE = False
if DO_SINGLE:
    section("5. One training run (live)")
    if SINGLE_CKPT is None:
        RUN_NAME = f"single__N{cfg.n_subgraphs}__" + time.strftime("%Y%m%d_%H%M%S")
    live = LiveRun(DASH, RUN_NAME, cfg, gi, extra={"experiment": "single", "seed": A.seeds[0]})
    print(f"streaming to http://localhost:8765/?run={RUN_NAME}  (through the tunnel)")
    t0 = time.time()
    MODEL_DIR = os.path.join(MODELS, "single", RUN_NAME)
    m = train_one(cfg, gi, A.seeds[0], live=live, save_dir=MODEL_DIR, ckpt_path=SINGLE_CKPT)   # model saved here
    if SINGLE_CKPT and os.path.exists(SINGLE_CKPT):
        os.remove(SINGLE_CKPT)                           # finished: the resume file is no longer needed
    print(f"trained {cfg.steps} steps in {time.time() - t0:.0f}s on {resolve_device(cfg.device)}")

    h = pd.DataFrame(m["history"]).set_index("step")
    h[[c for c in h.columns if c not in ("sec", "flagged")]].plot(logy=True, figsize=(7, 3), title="training losses")
    savefig("single_losses")
    display(pd.Series(flat(m)).to_frame("value"))

    print("\nEpoch by epoch (single run):")
    et = epoch_table(m)
    if len(et):
        display(et.round(4))
        print("balanced MRR / chance for every KU type at every epoch:")
        display(epoch_types(m))
        fig, ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].plot(et.index, et["val balanced MRR"], "o-", label="val balanced MRR"); ax[0].plot(et.index, et["chance"], "k:", label="chance")
        ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(et.index, et["train JEPA"], "o-", c="C3"); ax[1].set_yscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_title("train JEPA loss (epoch mean)")
        savefig("single_epochs")
    else:
        print("no end-of-epoch results: set --epochs >= 1")

    print("\nReasoning-task evaluation matrix (single run):")
    display(task_table(m.get("tasks")))
    R = [r for r in m.get("epoch_results", []) if r.get("tasks") and "note" not in r["tasks"]]
    if R:
        print("every task at every epoch (validation):")
        display(pd.DataFrame({f"epoch {r['epoch']}": task_summary(r["tasks"]) for r in R}).round(3))

# ------------------------------------------------------------------ 6. experiments
all_results = {}
if A.experiments and DO_EXPS:
    section("6. Experiments (live)")
    for e in A.experiments:
        print(e, "->", [tag for tag, _ in EXP_ARMS[e]])
    print("runs:", sum(len(EXP_ARMS[e]) for e in A.experiments) * len(A.seeds), f"x {A.epochs} epochs")
    from sci_jepa.reporting.report_pdf import build as build_pdf      # leaderboards + method catalogue as PDF
    INCOMPLETE = []
    for e in A.experiments:
        print("#" * 30, e, "#" * 30, flush=True)
        if A.stage == "merge":                         # a chained job that ran out of time: the next job continues
            from sci_jepa.training.train import missing_runs
            miss = missing_runs(cfg, e)
            if miss:
                print(f"{e}: {len(miss)} arm x seed runs not finished yet (e.g. {miss[:4]}); no summary or report "
                      f"until they are; submit the next job to continue", flush=True)
                INCOMPLETE.append(e)
                continue
        res, summ = run_experiment(cfg, gi, e, log=lambda *a: print(*a, flush=True), live_root=DASH, shard=SHARD)
        if SHARD is not None:                          # this GPU's share is done; the merge stage writes the rest
            continue
        print("PDF report:", build_pdf(os.path.join(OUT, e)))
        print("paper figures (PDF + PNG) and leaderboard.tex:", os.path.join(OUT, e, "figures"))
        all_results[e] = res
        rows = [{"exp": e, "arm": tag, "seed": r["seed"], **flat(r)} for tag, runs in res.items() for r in runs]
        display(pd.DataFrame(rows).groupby(["exp", "arm"], sort=False).mean(numeric_only=True)
                [["balanced MRR", "balanced chance", "standard MRR", "chi (target, type-centred)",
                  "partition probe acc", "inverse-relation acc", "audit: flagged frac", "train sec"]].round(4))
    if A.stage == "merge":                             # read by the cluster script (a finished chain cancels the rest)
        Path(OUT, "infra").mkdir(parents=True, exist_ok=True)
        Path(OUT, "infra", "status.json").write_text(json.dumps({
            "complete": not INCOMPLETE, "incomplete": INCOMPLETE, "experiments": A.experiments,
            "job": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
    if INCOMPLETE and not all_results:                 # nothing complete to report yet (chained jobs)
        print(f"merge: {', '.join(INCOMPLETE)} not complete; reports follow when every run has finished", flush=True)
        sys.exit(0)

    # 6a. validity, domains, findings
    if DO_REPORT and not A.skip_validity:
        section("6a. Benchmark validity, research domains and main findings")
        from sci_jepa.evaluation import validity as VAL
        from sci_jepa.evaluation.findings import analyse
        print("research domains in this corpus:", gi.build_stats.get("domains"))
        if gi.build_stats.get("domain_balance"):
            b = gi.build_stats["domain_balance"]
            display(domain_balance_table(b))
        aud = VAL.audit(gi, cfg); aud.pop("flagged")
        display(pd.DataFrame(aud["by_kind"]).T.rename_axis("counterfactual kind"))
        print("removed kinds:", aud.get("removed_kinds"), "| share of edits removed:", f"{aud['flagged_share']:.0%}")
        fa = VAL.faithfulness(gi, cfg, n=60); fa.pop("lowest", None)
        if fa.get("n"):
            display(pd.DataFrame(fa["by_type"]).T.round(2).rename_axis("KU type"))
        ann = VAL.export(gi, cfg, os.path.join(OUT, "annotation"), n=200)
        print("annotation sheets for two experts:", ann)
        for e in [x for x in A.experiments if x not in INCOMPLETE]:
            F = analyse(os.path.join(OUT, e))
            if F.get("hypotheses"):
                display(pd.DataFrame([{k: h.get(k) for k in ("id", "claim", "a", "b", "difference", "ci95", "outcome")}
                                      for h in F["hypotheses"]]).set_index("id"))
                print(f"{e} conclusion:", F["conclusion"])

    # leaderboards
    if not DO_REPORT:
        print(f"shard {SHARD[0]}/{SHARD[1]} finished", flush=True)
        sys.exit(0)
    section("Leaderboards (test split)")

    def _g(d, *path):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    def reasoning_row(m):
        T = m.get("tasks") or {}
        sc = {"forward": _g(T, "forward", "ratio"), "experiment design": _g(T, "inverse", "ratio"),
              "abduction": _g(T, "abduction", "ratio"), "consistency (bal. acc)": _g(T, "consistency", "bal_acc"),
              "intervention: toward new result": _g(T, "intervention", "condition", "toward_donor"),
              "intervention: shift": _g(T, "intervention", "condition", "rel_shift")}
        v = [sc["forward"], sc["experiment design"], sc["abduction"],
             None if sc["consistency (bal. acc)"] is None else sc["consistency (bal. acc)"] / 0.5,
             None if sc["intervention: toward new result"] is None else sc["intervention: toward new result"] / 0.5]
        v = [x for x in v if x is not None and x == x]
        acc = [_g(T, k, "acc_sim") for k in ("forward", "inverse", "abduction")]
        acc = [x for x in acc if x is not None and x == x]
        return {"reasoning index": float(np.mean(v)) if v else float("nan"), **sc,
                f"accuracy (sim >= {cfg.sim_threshold:.2f}), 3 retrieval tasks": float(np.mean(acc)) if acc else float("nan")}

    def health_row(m):
        rb = m["retrieval_balanced"]; st = m.get("structure") or {}; ac = m.get("actions") or {}
        P = m.get("per_type") or {}
        return {"test MRR / chance": rb["MRR"] / max(rb.get("chance_MRR") or 1e-9, 1e-9),
                f"test accuracy (sim >= {cfg.sim_threshold:.2f})": _g(m, "similarity", "acc"),
                "... and ranked first": _g(m, "similarity", "acc_top1"),
                "... false accept (best wrong also >= thr)": _g(m, "similarity", "false_accept"),
                "KU types learned": sum(v["MRR_bal"] >= 1.5 * v["chance_bal"] for v in P.values()) / max(len(P), 1),
                "from structure": (st["with"] - st["without"]) if st else float("nan"),
                "action swap": _g(ac, "swap", "direction_cos"), "action flip": _g(ac, "flip", "direction_cos"),
                "train sec": m.get("train_sec")}

    def board(fn, sort_col):
        rows = [{"exp": e, "arm": tag, **fn(r)} for e, res in all_results.items() for tag, runs in res.items() for r in runs]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).apply(pd.to_numeric, errors="ignore")
        g = df.groupby(["exp", "arm"], sort=False)
        mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
        out = mean.round(3).astype(str) + std.round(3).map(lambda x: "" if x != x else f" ± {x}")
        return out.loc[mean.sort_values(sort_col, ascending=False).index]

    print("Leaderboard 1 - reasoning benchmark"); display(board(reasoning_row, "reasoning index"))
    print("Leaderboard 2 - retrieval and model health"); display(board(health_row, "test MRR / chance"))

    # epoch by epoch, every variation
    section("Epoch by epoch: every variation")
    for e, res in all_results.items():
        ratio, learned = {}, {}
        for tag, runs in res.items():
            for r in runs:
                key = f"{tag} (seed {r['seed']})"
                ratio[key] = {f"epoch {x['epoch']}": x["bal_MRR"] / max(x["bal_chance"], 1e-9) for x in r.get("epoch_results", [])}
                learned[key] = {f"epoch {x['epoch']}": f"{x['types_learned']}/{len(x['per_type'])}" for x in r.get("epoch_results", [])}
        if not any(ratio.values()):
            continue
        print(e, "- balanced MRR / chance:"); display(pd.DataFrame(ratio).T.round(2))
        tk = {f"{tag} (seed {r['seed']})": task_summary(r.get("tasks")) for tag, runs in res.items() for r in runs}
        if any(tk.values()):
            print(e, "- reasoning tasks (test): retrieval x chance, consistency balanced acc., intervention toward donor:")
            display(pd.DataFrame(tk).T.round(3))
        print(e, "- KU types learned:"); display(pd.DataFrame(learned).T)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for k, v in ratio.items():
            ax.plot([int(c.split()[1]) for c in v], list(v.values()), "o-", label=k, lw=2.5 if "mixed" in k else 1)
        ax.axhline(1, c="k", ls=":"); ax.axhline(1.5, c="g", ls="--", lw=0.8)
        ax.set_xlabel("epoch"); ax.set_ylabel("val MRR / chance"); ax.set_title(e); ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1, 1))
        savefig(f"epochs_{e}")

    if "E10" in all_results:
        r = all_results["E10"]
        tags = list(r)
        xs = [int(t.split("=")[1]) for t in tags]
        get = lambda k1, k2: [np.mean([x[k1][k2] for x in r[t]]) for t in tags]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3))
        ax[0].plot(xs, get("retrieval_balanced", "MRR"), "o-", label="balanced MRR")
        ax[0].plot(xs, get("retrieval_balanced", "chance_MRR"), "k:", label="chance"); ax[0].legend(); ax[0].set_xlabel("N")
        ax[1].plot(xs, get("target_space", "chi"), "o-"); ax[1].set_title("chi (type-centred)"); ax[1].set_xlabel("N")
        ax[2].plot(xs, get("partition_probe", "between_subgraph_var_share"), "o-"); ax[2].axhline(0.3, c="r", ls="--")
        ax[2].set_title("between-subgraph variance share (G1 < 0.30)"); ax[2].set_xlabel("N")
        for a in ax: a.set_xscale("log", base=2)
        savefig("E10_neighbourhood")

# ------------------------------------------------------------------ 6b. use a trained model
if A.stage == "merge" and not A.skip_single:
    singles = sorted(OUT.glob("models/single__*"), key=lambda d: d.stat().st_mtime)
    MODEL_DIR = str(singles[-1]) if singles else None
if MODEL_DIR and DO_REPORT and not A.skip_inference:
    section("6b. Use a trained model")
    from sci_jepa.inference import SciJEPA
    api = SciJEPA.from_pretrained(MODEL_DIR, weights="best", index=gi)
    paper = {
        "research_context": "Graph neural networks struggle with long-range dependencies in molecular graphs.",
        "research_question_hypothesis": "Does adding a virtual node improve long-range message passing?",
        "methodological_details": "A virtual node connected to every atom is added to a GIN model trained on ZINC.",
        "procedures_architectures": "Five layers, hidden size 300, Adam optimiser, 200 epochs.",
        "key_results": "The virtual node reduces the mean absolute error by 15% compared with the baseline.",
        "claims": [{"description": "Virtual nodes improve long-range information flow.",
                    "supporting_evidence": "Error drops by 15% on ZINC.", "contradicting_evidence": "",
                    "implications": "Global context matters for molecular property prediction."}],
        "field_subfield": "Computer Science — Machine Learning",
    }
    print("facet embeddings:", [t for t, _ in api.embed(paper)["kus"]])
    for score, text in api.rank(paper, hide="result", task="forward", candidates=[
            "The virtual node reduces the mean absolute error by 15% compared with the baseline.",
            "The virtual node increases the mean absolute error by 15% compared with the baseline.",
            "Protein structure prediction accuracy improves with more templates."]):
        print(f"{score:+.3f}  {text}")
    print(api.consistency("Virtual nodes improve long-range information flow.", "Error drops by 15% on ZINC."))

# ------------------------------------------------------------------ 7. files
section("7. Dashboard and files")
print("KU graph cache :", os.environ["PAPER_A_INDEX_DIR"])
print("results        :", OUT)
for p in sorted(OUT.glob("*/summary.json")):
    print("  ", p.relative_to(OUT))
print("figures        :", FIG)
print("reopen later   : sci-jepa serve", DASH)
if A.keep_serving > 0 and A.stage == "all":          # several GPUs: the job script keeps the server
    print(f"dashboard stays up for {A.keep_serving} more minutes at http://localhost:8765/ (through the tunnel)", flush=True)
    time.sleep(60 * A.keep_serving)
