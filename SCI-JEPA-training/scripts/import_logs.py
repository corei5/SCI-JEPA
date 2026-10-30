#!/usr/bin/env python3
"""
Rebuild earlier training runs from their SLURM logs, so the dashboard shows them as previous runs.

    python3 scripts/import_logs.py ../log runs/paper_a_results/dashboard        # local dashboard folder
    python3 scripts/import_logs.py ../log /tmp/archive && \\
        python3 scripts/push_dashboard.py /tmp/archive --once --no-squeue        # straight to the hosted one

Reads every *.log / *.out under the log folder (the stdout of `sci-jepa run` / run_paper_a.sh, one or many
runs per file) and writes, per run, archive/<name>/live.json in the LiveRun format plus the list
runs_archive.json. The dashboard lists those before the runs in runs.json; they are kept apart so the
cluster's own runs.json (rewritten by every run) never drops them, and push_dashboard.py --prune keeps them.

What a log holds, and so what an imported run shows: the logged training steps (loss curves), one validation
row per epoch (balanced MRR, chance, similarity accuracy, KU types learned) and the final test line (balanced
and standard MRR, chi, probe, relation sensitivity, inverse accuracy, audit). Per-type scores, reasoning tasks,
neurons and examples are not in the log. A run resumed in a later job ([resume] line) is joined to the run it
continues; the same run printed by two files of one job (job .out and per-GPU log) is merged. Run names are
log-<results folder>__<experiment>__<arm>__seed<k>__job<id>. Re-running replaces the runs it wrote before.
Needs only python3.
"""
import argparse
import json
import math
import os
import re
import sys
import time

HEADER = re.compile(r"^\s*\[paper_a\] (\S+) \| (.+?) \| seed (\d+)(?:\s+\[worker gpu \d+\])?\s*$")
OTHER_PAPER_A = re.compile(r"^\s*\[paper_a\] ")
RESULTS = re.compile(r"^\s*(?:results|output)\s*:\s*(\S+)|\[paper_a\] live dashboard data -> (\S+?)/dashboard\s*$")
KU_MODE = re.compile(r"ku_mode='(\w+)'")
CONTEXT_N = re.compile(r"^\s*context\s+N = (\d+)")
SIZE = re.compile(r"\(model size (\w+), (\d+) seeds?\)")
MASK_STEPS = re.compile(r"\[mask\] schedule=\S+ steps=(\d+)")
TRAIN = re.compile(r"^\s*train\s*:\s*(\d+) epochs, batch (\d+)")
STEPS = re.compile(r"at most (\d+) epochs over (\d+) hidden KUs = (\d+) steps of (\d+)")
INDEX = re.compile(r"\[paper_a\] index: (\d+) papers, (\d+) KUs \(([\d.]+)/paper\), splits (\{[^}]*\})")
STEP_ROW = re.compile(r"^\s*step=\d+ ")
RESUME = re.compile(r"\[resume\] .*continuing after step (\d+)")
EPOCH = re.compile(r"\[epoch ([\d.]+)\] val balanced MRR (\S+) \(chance (\S+)\)"
                   r"(?: acc\(sim>=\S+\) (\S+) \(top-1 (\S+), false accept (\S+)\))?"
                   r" train jepa (\S+) types learned (\d+)/(\d+)(?: (\w+) (\S+))?")
FINAL = re.compile(r"^\s*-> balanced MRR (\S+) \(chance (\S+)\) \| standard MRR (\S+) \(chance (\S+)\) \| chi (\S+) \| "
                   r"probe (\S+) \| rel-sens (\S+) \| inv (\S+) \| audit flagged (\S+)")
BEST = re.compile(r"\[best\] testing the weights of epoch ([\d.]+)")
EARLY = re.compile(r"\[early stop\]")
JOB = re.compile(r"_(\d{4,})(?:_|\.)")


def num(s):
    """'0.83' -> 0.83, 'nan' -> None (JSON has no NaN), '2' -> 2."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return s
    if math.isnan(v) or math.isinf(v):
        return None
    return int(v) if v.is_integer() and "." not in s and "e" not in s.lower() else v


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.@+\-]", "_", s.replace("=", "-"))


def parse_file(path, dataset=None):
    """One dict per run segment in the file, with the header context in force when it started. dataset: the
    results folder if the file does not name it (per-GPU worker logs of a job whose .out does)."""
    ctx = {"dataset": dataset, "config": {}, "index": None}
    seg = None
    job = (JOB.search(os.path.basename(path)) or [None, None])[1]

    def close():
        nonlocal seg
        if seg is not None and (seg["history"] or seg["epochs"] or seg["final"]):
            yield_list.append(seg)
        seg = None

    yield_list = []
    with open(path, errors="replace") as f:
        for line in f:
            m = HEADER.match(line)
            if m:
                close()
                seg = {"exp": m[1], "arm": m[2].strip(), "seed": int(m[3]), "job": job, "file": os.path.basename(path),
                       "dataset": ctx["dataset"], "config": dict(ctx["config"]), "index": ctx["index"],
                       "resumed_at": None, "history": [], "epochs": [], "final": None, "best_epoch": None,
                       "early_stop": False, "failed": False}
                continue
            if OTHER_PAPER_A.match(line) and "index:" not in line and "live dashboard" not in line:
                close()                                   # "already finished", "skipped", "done; exiting", ...
            if (m := RESULTS.match(line)):
                ctx["dataset"] = os.path.basename((m[1] or m[2]).rstrip("/"))
            if (m := KU_MODE.search(line)):
                ctx["config"]["ku_mode"] = m[1]
            if (m := CONTEXT_N.match(line)):
                ctx["config"]["n_subgraphs"] = int(m[1])
            if (m := SIZE.search(line)):
                ctx["config"]["size"] = m[1]
            if (m := STEPS.search(line)):
                ctx["config"].update(epochs=int(m[1]), hidden_kus=int(m[2]), steps=int(m[3]), batch_size=int(m[4]))
                if seg is not None:
                    seg["config"].update(ctx["config"])
            if (m := TRAIN.match(line)):
                ctx["config"].setdefault("epochs", int(m[1]))
                ctx["config"].setdefault("batch_size", int(m[2]))
            if (m := MASK_STEPS.search(line)) and seg is not None:
                seg["config"].setdefault("steps", int(m[1]))
            if (m := INDEX.search(line)):
                try:
                    splits = json.loads(m[4].replace("'", '"'))
                except ValueError:
                    splits = {}
                ctx["index"] = {"papers": int(m[1]), "kus": int(m[2]), "kus_per_paper": float(m[3]), "splits": splits}
            if seg is None:
                continue
            if STEP_ROW.match(line):
                rec = {}
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        rec[k] = num(v)
                if isinstance(rec.get("step"), int):
                    seg["history"].append(rec)
            elif (m := RESUME.search(line)):
                seg["resumed_at"] = int(m[1])
            elif (m := EPOCH.search(line)):
                rec = {"epoch": num(m[1]), "bal_MRR": num(m[2]), "bal_chance": num(m[3]), "acc_sim": num(m[4]),
                       "acc_sim_top1": num(m[5]), "false_accept": num(m[6]), "train_jepa": num(m[7]),
                       "types_learned": int(m[8]), "n_types": int(m[9]),
                       "step": seg["history"][-1]["step"] if seg["history"] else None}
                if m[10]:
                    rec["select_metric"], rec["select_score"] = m[10], num(m[11])
                seg["epochs"].append(rec)
            elif (m := FINAL.match(line)):
                seg["final"] = {"retrieval_balanced": {"MRR": num(m[1]), "chance_MRR": num(m[2])},
                                "retrieval_standard": {"MRR": num(m[3])}, "chance_MRR_standard": num(m[4]),
                                "target_space": {"chi": num(m[5])}, "partition_probe": {"probe_acc": num(m[6])},
                                "relation_sensitivity": num(m[7]), "inverse_acc": num(m[8]),
                                "audit": {"flagged_frac": num(m[9])}}
                close()
            elif (m := BEST.search(line)):
                seg["best_epoch"] = num(m[1])
            elif EARLY.search(line):
                seg["early_stop"] = True
            elif line.startswith("Traceback") or "CUDA out of memory" in line:
                seg["failed"] = True
    close()
    return yield_list, ctx["dataset"]


def by_key(rows, key):
    out = {}
    for r in rows:
        if r.get(key) is not None:
            out[r[key]] = r                               # later files win (a resumed job repeats the last rows)
    return [out[k] for k in sorted(out)]


def build(runs_segments, mtimes):
    s0 = runs_segments[0]
    segs = runs_segments
    cfg = {}
    for s in segs:
        cfg.update(s["config"])
    history = by_key([r for s in segs for r in s["history"]], "step")
    epochs = by_key([r for s in segs for r in s["epochs"]], "epoch")
    final = next((s["final"] for s in reversed(segs) if s["final"]), None)
    best = next((s["best_epoch"] for s in reversed(segs) if s["best_epoch"] is not None), None)
    trained = bool(history or epochs)
    steps_total = cfg.get("steps") if trained else 0
    spe = steps_total / cfg["epochs"] if trained and steps_total and cfg.get("epochs") else None
    for e in epochs:
        if e["step"] is None and spe:
            e["step"] = int(round(e["epoch"] * spe)) - 1
    if final is not None:
        final["best_epoch"] = best
        status = "done"
    elif any(s["failed"] for s in segs[-1:]):
        status = "failed (see log)"
    else:
        status = "stopped (log ends)"
    jobs = sorted({s["job"] for s in segs if s["job"]}, key=int)
    idx = s0["index"] or {}
    name = safe("log-%s__%s__%s__seed%d__job%s" % (s0["dataset"] or "unknown", s0["exp"], s0["arm"], s0["seed"],
                                                    jobs[0] if jobs else "na"))
    t0 = min(mtimes[s["file"]] for s in segs)
    t1 = max(mtimes[s["file"]] for s in segs)
    return {
        "name": name, "status": status, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
        "steps_total": steps_total, "steps_per_epoch": spe, "epochs_total": cfg.get("epochs") if trained else None,
        "experiment": s0["exp"], "arm": s0["arm"], "seed": s0["seed"],
        "about": "rebuilt from the job log (%s): loss curves, per-epoch validation and the final test line only"
                 % ", ".join(sorted({s["file"] for s in segs})),
        "family": None if trained else "comparison",
        "overrides": {},
        "config": {"ku_mode": cfg.get("ku_mode", "section"), "n_subgraphs": cfg.get("n_subgraphs"),
                   "steps": cfg.get("steps"), "epochs": cfg.get("epochs"), "batch_size": cfg.get("batch_size"),
                   "size": cfg.get("size"), "baseline": None if trained else s0["arm"], "device": "cuda"},
        "index": {"papers": idx.get("papers"), "kus": idx.get("kus"), "kus_per_paper": idx.get("kus_per_paper"),
                  "per_type": {}, "splits": idx.get("splits") or {}, "paper_edges_cites": None,
                  "paper_edges_proxy": None, "fields": None},
        "rel_names": [], "comp_types": [], "role_rels": [],
        "history": history,
        "evals": [{k: e.get(k) for k in ("step", "epoch", "bal_MRR", "bal_chance", "acc_sim", "acc_sim_top1",
                                          "false_accept")} | {"per_type": {}, "per_type_chance": {}} for e in epochs],
        "epoch_results": [{**e, "per_type": {}} for e in epochs],
        "relbias": None, "final": final,
        "resumed": [{"step": s["resumed_at"], "job": s["job"]} for s in segs if s["resumed_at"] is not None],
        "imported": {"from": "log", "files": sorted({s["file"] for s in segs}), "jobs": jobs},
    }


def collect(log_dir):
    files = [os.path.join(d, f) for d, _, fs in os.walk(log_dir) for f in fs
             if f.endswith((".log", ".out")) and not f.startswith(".")]
    jobnum = lambda p: int((JOB.search(os.path.basename(p)) or [0, 0])[1])
    files.sort(key=lambda p: (jobnum(p), os.path.basename(p)))
    mtimes = {os.path.basename(p): os.path.getmtime(p) for p in files}
    runs = []                                             # [[segment, ...], ...]
    latest = {}                                           # (dataset, exp, arm, seed) -> runs entry
    job_dataset = {}
    for p in files:
        job = (JOB.search(os.path.basename(p)) or [None, None])[1]
        segs, dataset = parse_file(p, job_dataset.get(job))
        if dataset and job:
            job_dataset.setdefault(job, dataset)
        for seg in segs:
            key = (seg["dataset"], seg["exp"], seg["arm"], seg["seed"])
            prev = latest.get(key)
            if prev is not None and (seg["resumed_at"] is not None or seg["job"] in {s["job"] for s in prev}):
                prev.append(seg)
            else:
                prev = [seg]
                runs.append(prev)
                latest[key] = prev
    return [build(r, mtimes) for r in runs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", help="folder of SLURM logs (*.log, *.out)")
    ap.add_argument("dashboard", help="dashboard folder to write archive/ and runs_archive.json into")
    opts = ap.parse_args(argv)
    runs = collect(opts.log_dir)
    os.makedirs(os.path.join(opts.dashboard, "archive"), exist_ok=True)
    idx_path = os.path.join(opts.dashboard, "runs_archive.json")
    try:
        index = json.load(open(idx_path))
    except (OSError, ValueError):
        index = []
    names = {r["name"] for r in runs}
    index = [r for r in index if not r["name"].startswith("log-") or r["name"] in names]
    index = [r for r in index if r["name"] not in names]
    for r in sorted(runs, key=lambda r: r["started"]):
        d = os.path.join(opts.dashboard, "archive", r["name"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "live.json"), "w") as f:
            json.dump(r, f, separators=(",", ":"))
        index.append({"name": r["name"], "started": r["started"], "path": "archive/%s/live.json" % r["name"]})
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=1)
    for r in runs:
        fin = (r["final"] or {}).get("retrieval_balanced", {})
        print("%-80s %-20s steps %6d  epochs %3d  test bal.MRR %s" % (
            r["name"], r["status"], len(r["history"]), len(r["epoch_results"]), fin.get("MRR", "-")))
    print("%d runs -> %s" % (len(runs), os.path.abspath(opts.dashboard)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
