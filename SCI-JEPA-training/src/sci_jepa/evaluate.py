"""
SCI-JEPA Paper A -- standalone evaluation.

Evaluates on the Paper A benchmark with exactly the code and seeds the training uses (train/paper_a/
diagnostics.py, tasks.py, worldview.py, baselines.py, findings.py), so the numbers match the ones written
at the end of training. How the evaluation works: docs/evaluation.md.

    # 1. a trained model folder (written by training: model.safetensors, best.safetensors, config.json)
    sci-jepa evaluate model  runs/paper_a_results/BENCH/models/reference_seed0 --weights best

    # 2. a comparison system on the same papers, candidates, counterfactuals and tasks
    sci-jepa evaluate system bm25 --config runs/paper_a_results/BENCH/models/reference_seed0/config.json
    #    systems: text | bm25 | scincl | bge | bge-m3 | minilm | supervised | llm

    # 3. a finished experiment folder (<arm>_seed<s>.json files): leaderboards + hypothesis tests, no GPU
    sci-jepa evaluate results runs/paper_a_results/BENCH

Every command writes <out>.json (all metrics) and <out>.md (a readable report).
Data paths stored in a model's config.json are those of the machine it was trained on; point them at
your copy with --parquet (normalized Scientific-Summaries folder(s), comma-separated) and --index-dir.
The KU index is rebuilt from the parquet if it is not in --index-dir (same settings -> same papers,
splits and candidates; the data fingerprint is checked against the model's summary.json).
"""
import argparse
import glob
import json
import math
import os
import sys
import time



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_data=True):
        sp.add_argument("--out", default=None, help="output path without extension (default: next to the input)")
        if needs_data:
            sp.add_argument("--parquet", default=os.environ.get("PAPER_A_PARQUET"),
                            help="normalized parquet folder(s), comma-separated (default: as in the config)")
            sp.add_argument("--split-dir", default=os.environ.get("PAPER_A_SPLIT"),
                            help="stored split folder (default: as in the config)")
            sp.add_argument("--index-dir", default=os.environ.get("PAPER_A_INDEX_DIR"),
                            help="KU index cache folder (default: as in the config)")
            sp.add_argument("--split", default="test", choices=["test", "val"])
            sp.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")

    m = sub.add_parser("model", help="evaluate a saved model folder")
    m.add_argument("model_dir")
    m.add_argument("--weights", default="best", choices=["best", "model"],
                   help="best = best validation epoch (what training tests), model = end of training")
    m.add_argument("--no-tasks", action="store_true", help="skip the five reasoning tasks")
    m.add_argument("--no-worldview", action="store_true", help="skip structure reliance / action prediction")
    common(m)

    s = sub.add_parser("system", help="evaluate a comparison system with the same protocol")
    s.add_argument("name", help="text | bm25 | scincl | bge | bge-m3 | minilm | supervised | llm")
    s.add_argument("--config", required=True,
                   help="config.json of a model folder, or a <arm>_seed<s>.json result: fixes the data settings")
    s.add_argument("--seed", type=int, default=0, help="as in training: the system is evaluated with seed 1234 + seed")
    s.add_argument("--llm-model", default=None, help="for name=llm: any Hugging Face causal LM")
    common(s)

    r = sub.add_parser("results", help="leaderboards + hypothesis tests from a finished experiment folder")
    r.add_argument("exp_dir")
    r.add_argument("--no-findings", action="store_true", help="skip the bootstrap hypothesis tests")
    common(r, needs_data=False)
    return p.parse_args()


A = parse_args()

import numpy as np                                     # noqa: E402

from sci_jepa.config import PaperAConfig, resolve_device   # noqa: E402
from sci_jepa.evaluation.tasks import reasoning_index, task_summary   # noqa: E402

TASKS = ("forward", "inverse", "abduction")


# ------------------------------------------------------------------ helpers
def load_cfg(path):
    """PaperAConfig from a model's config.json or a result json (its "config"), as inference.from_pretrained does."""
    raw = json.load(open(path))
    raw = raw.get("config", raw)
    base = PaperAConfig()
    over = {k: v for k, v in raw.items() if hasattr(base, k) and k not in ("embedder",)}
    for k, v in list(over.items()):                    # json turns tuples into lists
        if isinstance(getattr(base, k), tuple) and isinstance(v, list):
            over[k] = tuple(v)
    cfg = base.override(**over)
    if getattr(A, "parquet", None):
        cfg = cfg.override(parquet_dirs=tuple(A.parquet.split(",")))
    if getattr(A, "index_dir", None):
        cfg = cfg.override(index_dir=A.index_dir)
    if getattr(A, "split_dir", None):
        cfg = cfg.override(split_dir=A.split_dir)
    return cfg


def get_gi(cfg):
    from sci_jepa.data.graph_index import get_index
    for d in ([] if cfg.split_dir else cfg.parquet_dirs):
        if not os.path.isdir(d):
            sys.exit(f"parquet folder not found: {d} (pass --parquet)")
    t0 = time.time()
    gi = get_index(cfg)
    print(f"[eval] KU index: {gi.summary()['papers']} papers, {gi.summary()['kus']} KUs ({time.time() - t0:.0f}s)")
    return gi


def g(d, *path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def headline(m, average="macro"):
    """The numbers the benchmark reports for one run."""
    T = m.get("tasks") or {}
    rb = m.get("retrieval_balanced") or {}
    mp = [g(T, t, "mp_pair") for t in TASKS]
    mp = [x for x in mp if x is not None]
    return {
        "reasoning index (macro)": reasoning_index(T, "macro"),
        "reasoning index (micro)": reasoning_index(T, "micro"),
        "forward x chance": g(T, "forward", "ratio"),
        "experiment design x chance": g(T, "inverse", "ratio"),
        "abduction x chance": g(T, "abduction", "ratio"),
        "consistency bal. acc": g(T, "consistency", "bal_acc"),
        "intervention toward donor": g(T, "intervention", "condition", "toward_donor"),
        "counterfactual robustness": float(np.mean(mp)) if mp else None,
        "balanced MRR": rb.get("MRR"), "balanced chance MRR": rb.get("chance_MRR"),
        "balanced MRR x chance": rb["MRR"] / rb["chance_MRR"] if rb.get("MRR") is not None and rb.get("chance_MRR") else None,
        "Hits@1": rb.get("Hits@1"), "Hits@10": rb.get("Hits@10"),
        "standard MRR": g(m, "retrieval_standard", "MRR"),
        "chi (target)": g(m, "target_space", "chi"),
        "partition probe acc": g(m, "partition_probe", "probe_acc"),
        "inverse-relation acc": m.get("inverse_acc"),
        "audit: flagged frac": g(m, "audit", "flagged_frac"),
    }


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def write(out, payload, md):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    json.dump(payload, open(out + ".json", "w"), indent=2, default=str)
    open(out + ".md", "w").write(md)
    print(f"[eval] wrote {out}.json and {out}.md")


def run_report(title, m, extra=""):
    T = m.get("tasks") or {}
    rows = "\n".join(f"| {k} | {fmt(v)} |" for k, v in headline(m).items())
    per_task = ""
    if T and "note" not in T:
        lines = []
        for k in TASKS:
            t = T.get(k) or {}
            lines.append(f"| {t.get('label', k)} | {t.get('query', '')} | MRR / chance | {fmt(t.get('ratio'))} | "
                         f"{fmt(t.get('MRR'))} | {fmt(t.get('chance'))} | {fmt(t.get('Hits@10'))} | {fmt(t.get('mp_pair'))} | {t.get('n', '')} |")
        c = T.get("consistency") or {}
        lines.append(f"| {c.get('label', 'consistency')} | hide claim-evidence edges, classify | balanced acc. | "
                     f"{fmt(c.get('bal_acc'))} | | 0.5 | | | {c.get('n', '')} |")
        iv = g(T, "intervention", "condition") or {}
        lines.append(f"| Intervention (condition) | swap the procedure for a donor paper's | toward donor | "
                     f"{fmt(iv.get('toward_donor'))} | | 0.5 | | | {iv.get('n', '')} |")
        per_task = ("\n## Reasoning tasks\n\n| task | query | metric | score | MRR | chance | Hits@10 | "
                    "counterfactual pair acc. | n |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n")
    per_type = ""
    if m.get("per_type"):
        per_type = "\n## Per KU type (balanced MRR / chance)\n\n| KU type | MRR | chance | x chance |\n|---|---|---|---|\n" + "\n".join(
            f"| {t} | {fmt(v.get('MRR_bal'))} | {fmt(v.get('chance_bal'))} | "
            f"{fmt(v['MRR_bal'] / v['chance_bal'] if v.get('MRR_bal') is not None and v.get('chance_bal') else None)} |"
            for t, v in m["per_type"].items()) + "\n"
    return f"# {title}\n\n{extra}\n## Headline numbers\n\n| metric | value |\n|---|---|\n{rows}\n{per_task}{per_type}"


# ------------------------------------------------------------------ model
def cmd_model():
    import torch
    from safetensors.torch import load_file
    from sci_jepa.evaluation.diagnostics import evaluate
    from sci_jepa.data.ku_graph import data_fingerprint
    from sci_jepa.models.model import PaperAModel
    from sci_jepa.models.shapes import ShapeBank, SlotLabels

    d = A.model_dir.rstrip("/")
    cfg = load_cfg(os.path.join(d, "config.json"))
    gi = get_gi(cfg)
    summ = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    fp = data_fingerprint(cfg)
    if summ.get("data_fingerprint") and summ["data_fingerprint"] != fp:
        print(f"[eval] WARNING: data fingerprint {fp} differs from the model's {summ['data_fingerprint']}: "
              "different papers / splits than in training, numbers are not comparable")
    if gi.comp_x.shape[1] != cfg.feat_dim:
        cfg = cfg.override(feat_dim=int(gi.comp_x.shape[1]))
    dev = torch.device(resolve_device(A.device))
    model = PaperAModel(cfg)
    shapes = None
    if cfg.lambda_shacl > 0 or cfg.shape_spec:          # as train_one builds it
        shapes = ShapeBank(cfg.shape_spec, SlotLabels(gi), cfg.d_model, cfg)
        model.shapes = shapes
    f = os.path.join(d, f"{A.weights}.safetensors")
    if not os.path.exists(f):
        print(f"[eval] {f} not found, using model.safetensors")
        f = os.path.join(d, "model.safetensors")
    missing, unexpected = model.load_state_dict(load_file(f), strict=False)
    if missing or unexpected:
        sys.exit(f"weights do not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model = model.to(dev).eval()
    print(f"[eval] {d} ({A.weights}) on {dev}, split {A.split}")
    t0 = time.time()
    m = evaluate(model, gi, cfg, dev, shapes, split=A.split)
    if cfg.eval_worldview and not A.no_worldview:
        from sci_jepa.evaluation import worldview as W
        m["structure"] = W.structure_reliance(model, gi, cfg, dev, A.split, base=m)
        m["actions"] = W.action_eval(model, gi, cfg, dev, A.split)
    if cfg.eval_tasks and not A.no_tasks:
        from sci_jepa.evaluation.tasks import task_eval
        m["tasks"] = task_eval(model, gi, cfg, dev, A.split)
    m["eval_sec"] = round(time.time() - t0, 1)
    m["model_dir"], m["weights"], m["data_fingerprint"] = d, A.weights, fp
    trained = g(summ, "test", "balanced_MRR")
    extra = (f"Model `{d}` ({A.weights} weights), split `{A.split}`, {gi.summary()['papers']} papers, "
             f"data fingerprint `{fp}`.\n" +
             (f"\nBalanced MRR written at the end of training: {fmt(trained)}; now: "
              f"{fmt(m['retrieval_balanced']['MRR'])}.\n" if trained is not None and A.split == "test" else ""))
    write(A.out or os.path.join(d, f"eval_{A.split}_{A.weights}"), m, run_report("Paper A evaluation: model", m, extra))


# ------------------------------------------------------------------ system
def cmd_system():
    from sci_jepa.evaluation.baselines import run_isolated, text_baseline
    cfg = load_cfg(A.config).override(baseline=A.name)
    if A.llm_model:
        cfg = cfg.override(llm_model=A.llm_model)
    gi = get_gi(cfg)
    seed = 1234 + A.seed                                # as run_experiment evaluates comparison systems
    t0 = time.time()
    if A.name == "llm":                                 # own process: GPU memory fully isolated (as in training)
        m = run_isolated(cfg, A.split, seed=seed)
    else:
        m = text_baseline(cfg, gi, A.split, seed=seed)
    m["eval_sec"] = round(time.time() - t0, 1)
    extra = f"System `{A.name}`{' (' + cfg.llm_model + ')' if A.name == 'llm' else ''}, split `{A.split}`, seed {seed}.\n"
    write(A.out or f"eval_{A.name}_{A.split}_seed{A.seed}", m, run_report(f"Paper A evaluation: {A.name}", m, extra))


# ------------------------------------------------------------------ results
def cmd_results():
    files = sorted(f for f in glob.glob(os.path.join(A.exp_dir, "*_seed*.json")))
    if not files:
        sys.exit(f"no <arm>_seed<s>.json files in {A.exp_dir}")
    rows = []
    for f in files:
        m = json.load(open(f))
        rows.append({"arm": m.get("tag", os.path.basename(f).rsplit("_seed", 1)[0]), "seed": m.get("seed"),
                     "family": m.get("family", ""), **headline(m)})
    import pandas as pd
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("arm", "seed", "family")]
    gr = df.groupby("arm", sort=False)
    mean, std, n = gr[num].mean(), gr[num].std(), gr.size()
    board = mean.sort_values("reasoning index (macro)", ascending=False)
    cols = ["reasoning index (macro)", "forward x chance", "experiment design x chance", "abduction x chance",
            "consistency bal. acc", "intervention toward donor", "counterfactual robustness", "balanced MRR x chance"]

    def cell(arm, c):
        v, s = mean.loc[arm, c], std.loc[arm, c]
        return "–" if v != v else (f"{v:.3f}" + ("" if s != s else f" ± {s:.3f}"))

    md = [f"# Paper A leaderboard: {os.path.basename(A.exp_dir.rstrip('/'))}\n",
          f"{len(files)} runs, {len(mean)} arms; mean ± std over seeds; sorted by the macro reasoning index "
          "(mean of the five task scores on one x-chance scale, each research domain weighted equally).\n",
          "| rank | arm | family | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 4)]
    fam = df.groupby("arm", sort=False)["family"].first()
    for i, arm in enumerate(board.index, 1):
        md.append(f"| {i} | {arm} | {fam[arm]} | {n[arm]} | " + " | ".join(cell(arm, c) for c in cols) + " |")
    payload = {"exp_dir": A.exp_dir, "runs": rows,
               "leaderboard": {arm: {c: {"mean": mean.loc[arm, c], "std": std.loc[arm, c]} for c in num} for arm in board.index}}
    if not A.no_findings:
        try:
            from sci_jepa.evaluation.findings import analyse
            F = analyse(A.exp_dir)
            payload["findings"] = F
            if F.get("hypotheses"):
                md += ["\n## Hypotheses (2,000 bootstrap resamples of the test queries, pooled over seeds)\n",
                       "| id | claim | A | B | difference | 95% CI | outcome |", "|---|---|---|---|---|---|---|"]
                for h in F["hypotheses"]:
                    ci = h.get("ci95") or [None, None]
                    md.append(f"| {h.get('id')} | {h.get('claim')} | {h.get('a')} | {h.get('b')} | {fmt(h.get('difference'))} | "
                              f"{fmt(ci[0])} .. {fmt(ci[1])} | {h.get('outcome')} |")
                if any(h.get("outcome") != "NOT RUN" for h in F["hypotheses"]):
                    md.append(f"\n**Conclusion:** {F.get('conclusion', '')}\n")
                else:                                   # findings.py writes a conclusion even with nothing tested
                    md.append("\nNo hypothesis could be tested yet: the arms it compares are not all finished.\n")
        except Exception as e:                          # noqa: BLE001  (findings need every arm of BENCH)
            md.append(f"\n(hypothesis tests skipped: {e})\n")
    out = A.out or os.path.join(A.exp_dir, "evaluation")
    pd.DataFrame(rows).to_csv(out + "_runs.csv", index=False)
    write(out, payload, "\n".join(md) + "\n")


{"model": cmd_model, "system": cmd_system, "results": cmd_results}[A.cmd]()
