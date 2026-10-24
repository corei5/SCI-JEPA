"""
Main findings: pre-registered hypotheses, tested on an experiment's results (BENCH or ALL) with bootstrap
confidence intervals, and a conclusion written from the outcomes. The hypotheses and decision rules are
fixed before the large-scale runs, so the conclusion cannot be tuned to the results.

  H1 main         the best JEPA model (any encoder family) beats the strongest comparison system on the
                  reasoning index
  H2 graph        reading the KU graph with a GNN helps: the best JEPA + GNN arm beats KU-Graph JEPA
                  (graph transformer, same objective) on the reasoning index
  H3 world model  action-conditioned training improves intervention: Action-Conditioned JEPA beats KU-Graph
                  JEPA, and Motif + Action JEPA beats Motif-Target JEPA, on the intervention response
  H4 robustness   the best JEPA model resists counterfactuals better than the strongest comparison system
                  (share of edits ranked below the true answer, mean over forward / design / abduction)
  H5 LLM          the best JEPA model beats LLM-Reasoning on the reasoning index (when the LLM arm ran)
  C  contamination check: no system's retrieval score drops by more than 0.5x chance on test papers after
                  the text encoders' pretraining cutoff

Statistics: every run stores its per-query outcomes; a score's 95% interval is the 2.5-97.5 percentile of
2,000 bootstrap resamples of the queries (pooled over seeds, each task resampled separately). With
primary_average="macro" (default) the reasoning index is the mean over canonical research domains (each domain
weighs the same, domains with fewer than min_domain_queries test queries left out), and the bootstrap resamples
within every domain (stratified), so large domains cannot decide a hypothesis on their own. A difference
is SUPPORTED when its whole interval is above 0, CONTRADICTED when it is below 0, INCONCLUSIVE otherwise.
With >= 2 seeds, a Welch t-test over seeds is reported as well.

    python -m sci_jepa.evaluation.findings paper_a_results/BENCH      # -> findings.json + printed conclusion
"""
import glob
import json
import os
import sys

import numpy as np

B = 2000
TASKS = ("forward", "inverse", "abduction")


def _arm(tag):
    return tag.split("__", 1)[1] if "__" in tag else tag


def load_runs(exp_dir):
    from sci_jepa.config import EXPERIMENTS, method_family
    lookup = {t: o for arms in EXPERIMENTS.values() for t, o in arms}
    runs = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, "*_seed*.json"))):
        try:
            m = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        tag = m.get("tag")
        m["_family"] = m.get("family") or method_family(lookup.get(tag, {}))
        runs.setdefault(tag, []).append(m)
    return runs


def _pool(ms, task, field="q"):
    out = []
    for m in ms:
        T = m.get("tasks") or {}
        if task == "intervention":                       # shift relative to the run's typical gap (= rel_shift)
            c = (T.get("intervention") or {}).get("condition") or {}
            g = c.get("base_gap") or 1.0
            out += [[x[0], x[1] / max(g, 1e-12), x[2]] for x in (c.get(field) or [])]
        else:
            out += (T.get(task) or {}).get(field) or []
    return out


def _task_scores(q_fwd, q_inv, q_abd, q_con, q_int):
    def ratio(q):
        if not len(q):
            return np.nan
        q = np.asarray(q, float)
        return q[:, 0].mean() / max(q[:, 1].mean(), 1e-9)
    def bal(q):
        if not len(q):
            return np.nan
        q = np.asarray(q)
        rec = [(q[q[:, 0] == c, 1] == c).mean() for c in (0, 1) if (q[:, 0] == c).any()]
        return float(np.mean(rec))
    toward = np.asarray(q_int, float)[:, 0].mean() if len(q_int) else np.nan
    return {"forward": ratio(q_fwd), "inverse": ratio(q_inv), "abduction": ratio(q_abd), "consistency": bal(q_con),
            "toward": toward}


def _index(s):
    v = [s["forward"], s["inverse"], s["abduction"], s["consistency"] / 0.5, s["toward"] / 0.5]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else np.nan


def _boot_task(q, rng, score, average, min_n, min_domains):
    """(B,) bootstrap of one task's score. Macro: stratified by domain (q[:, 3]) when >= min_domains domains have
    >= min_n queries (resample within each, score each, average); otherwise, as in tasks.add_domains, micro."""
    if not len(q):
        return np.full(B, np.nan)
    if average == "macro" and q.ndim == 2 and q.shape[1] > 3:
        doms = [d for d in np.unique(q[:, 3]) if (q[:, 3] == d).sum() >= min_n]
        if len(doms) >= min_domains:
            parts = []
            for d in doms:
                g = q[q[:, 3] == d]
                parts.append(score(g, rng.integers(0, len(g), (B, len(g)))))
            return np.nanmean(np.stack(parts), 0)
    return score(q, rng.integers(0, len(q), (B, len(q))))


def bootstrap(ms, rng, average="micro", min_n=20, min_domains=3):
    """-> dict of (B,) arrays: index, each task, shift; or None without per-query data. The same rules as the
    point scores (tasks.add_domains), so a point estimate and its interval always describe the same statistic."""
    qs = [np.asarray(_pool(ms, t), float) for t in TASKS] + [np.asarray(_pool(ms, "consistency"), float),
                                                           np.asarray(_pool(ms, "intervention"), float)]
    if not any(len(q) for q in qs):
        return None
    ratio = lambda g, i: g[i, 0].mean(1) / np.maximum(g[i, 1].mean(1), 1e-9)
    def bal(g, i):
        y, yh = g[i, 0], g[i, 1]
        rec = [((yh == c) & (y == c)).sum(1) / np.maximum((y == c).sum(1), 1) for c in (0, 1) if (g[:, 0] == c).any()]
        return np.mean(rec, 0)
    out = {k: _boot_task(q, rng, ratio, average, min_n, min_domains) for k, q in zip(TASKS, qs[:3])}
    out["consistency"] = _boot_task(qs[3], rng, bal, average, min_n, min_domains)
    out["toward"] = _boot_task(qs[4], rng, lambda g, i: g[i, 0].mean(1), average, max(min_n // 2, 5), min_domains)
    out["shift"] = _boot_task(qs[4], rng, lambda g, i: g[i, 1].mean(1), average, max(min_n // 2, 5), min_domains)
    parts = np.stack([out["forward"], out["inverse"], out["abduction"], out["consistency"] / 0.5, out["toward"] / 0.5])
    out["index"] = np.nanmean(parts, 0)
    return out


def _avg(ms):
    return ((ms[0].get("config") or {}).get("primary_average") or "micro") if ms else "micro"


def point(ms):
    from sci_jepa.evaluation.tasks import reasoning_index
    v = [reasoning_index(m.get("tasks") or {}, _avg(ms)) for m in ms]
    v = [x for x in v if x is not None]
    shift = [((m.get("tasks") or {}).get("intervention") or {}).get("condition", {}).get("rel_shift") for m in ms]
    shift = [x for x in shift if x is not None]
    cf = []
    for m in ms:
        T = m.get("tasks") or {}
        c = [(T.get(k) or {}).get("mp_pair") for k in TASKS]
        c = [x for x in c if x is not None]
        if c:
            cf.append(float(np.mean(c)))
    return {"index": float(np.mean(v)) if v else None, "index_seeds": v,
            "shift": float(np.mean(shift)) if shift else None, "shift_seeds": shift,
            "cf": float(np.mean(cf)) if cf else None, "cf_seeds": cf}


def _ci(x):
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def _welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    from scipy import stats
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def compare(name, claim, A, B_, runs, boots, pts, metric="index"):
    """A beats B_ on metric? -> outcome dict."""
    if A not in runs or B_ not in runs:
        return {"id": name, "claim": claim, "outcome": "NOT RUN", "detail": f"missing {A if A not in runs else B_}"}
    pa, pb = pts[A], pts[B_]
    key = {"index": "index", "shift": "shift", "cf": "cf"}[metric]
    da, db = pa.get(key), pb.get(key)
    res = {"id": name, "claim": claim, "a": _arm(A), "b": _arm(B_), "metric": metric,
           "a_value": da, "b_value": db, "difference": (da - db) if da is not None and db is not None else None,
           "p_seeds": _welch(pa.get(key + "_seeds", []), pb.get(key + "_seeds", []))}
    ba, bb = boots.get(A), boots.get(B_)
    bkey = {"index": "index", "shift": "shift"}.get(metric)
    if ba is not None and bb is not None and bkey and np.isfinite(ba[bkey]).any() and np.isfinite(bb[bkey]).any():
        d = ba[bkey] - bb[bkey]
        lo, hi = _ci(d)
        res["ci95"] = [lo, hi]
        # point estimates from the same statistic as the interval (bootstrap mean), so they always agree
        res["a_value"], res["b_value"] = float(np.nanmean(ba[bkey])), float(np.nanmean(bb[bkey]))
        res["difference"] = float(np.nanmean(d))
        res["outcome"] = ("SUPPORTED" if lo is not None and lo > 0 else
                          "CONTRADICTED" if hi is not None and hi < 0 else "INCONCLUSIVE")
    elif res["difference"] is not None:
        res["outcome"] = "SUPPORTED (point estimate)" if res["difference"] > 0 else "NOT SUPPORTED (point estimate)"
    else:
        res["outcome"] = "NOT RUN"
    return res


def analyse(exp_dir, seed=0):
    runs = load_runs(exp_dir)
    if not runs:
        return {"note": f"no results in {exp_dir}"}
    rng = np.random.default_rng(seed)
    boots = {t: bootstrap(ms, rng, _avg(ms), (ms[0].get("config") or {}).get("min_domain_queries", 20),
                          (ms[0].get("config") or {}).get("min_domains", 3)) for t, ms in runs.items()}
    average = _avg(next(iter(runs.values())))
    pts = {t: point(ms) for t, ms in runs.items()}
    by_arm = {_arm(t): t for t in runs}
    fam = {t: ms[0]["_family"] for t, ms in runs.items()}
    best = lambda tags: max(tags, key=lambda t: pts[t]["index"] if pts[t]["index"] is not None else -1) if tags else None
    def current(t):                                   # evaluated with the current protocol (topic pools +
        c = runs[t][0].get("config") or {}             # counterfactuals); old-protocol arms are ablations only
        return c.get("pool_mode", "topic") == "topic" and c.get("minimal_pairs", True)
    is_llm = lambda t: _arm(t).startswith("LLM-Reasoning")   # one arm, or one per model (LLM-Reasoning@<model>)
    jepa = [t for t in runs if fam[t] in ("jepa", "gnn") and current(t)]
    comp = [t for t in runs if fam[t] == "comparison" and not is_llm(t) and current(t)]
    llms = [t for t in runs if is_llm(t) and current(t)]
    gnn = [t for t in runs if fam[t] == "gnn" and current(t)]
    bj, bc, bg, bl = best(jepa), best(comp), best(gnn), best(llms)
    H = [compare("H1", "the best JEPA model beats the strongest comparison system (reasoning index)", bj, bc, runs, boots, pts),
         compare("H2", "reading the KU graph with a GNN beats the graph transformer (best JEPA + GNN vs KU-Graph JEPA)",
                 bg, by_arm.get("reference"), runs, boots, pts),
         compare("H3a", "action training improves intervention (Action-Conditioned vs KU-Graph JEPA, intervention shift)",
                 by_arm.get("world_model"), by_arm.get("reference"), runs, boots, pts, metric="shift"),
         compare("H3b", "action training improves intervention (Motif + Action vs Motif-Target JEPA, intervention shift)",
                 by_arm.get("motif+world"), by_arm.get("motif"), runs, boots, pts, metric="shift"),
         compare("H4", "the best JEPA model resists counterfactuals better than the strongest comparison system",
                 bj, bc, runs, boots, pts, metric="cf"),
         compare("H5", "the best JEPA model beats " + ("the strongest LLM" if len(llms) > 1 else "LLM-Reasoning")
                 + " (reasoning index)", bj, bl, runs, boots, pts)]
    if len(llms) > 1:                                 # several LLMs: the best JEPA model against each of them
        H += [compare(f"H5-{_arm(t).split('@', 1)[-1]}", f"the best JEPA model beats {_arm(t)} (reasoning index)",
                      bj, t, runs, boots, pts) for t in sorted(llms, key=_arm)]
    # contamination check
    from sci_jepa.evaluation.validity import contamination
    cont = contamination(exp_dir)
    drops = {_arm(t): max(v for k, v in r.items() if k.endswith("drop_after_cutoff") and not k.startswith("consistency"))
             for t, r in cont.items()
             if any(k.endswith("drop_after_cutoff") and not k.startswith("consistency") for k in r)}
    n_after = max([v for r in cont.values() for k, v in r.items() if k.endswith(":n_after")] or [0])
    worst = max(drops.items(), key=lambda kv: kv[1]) if drops else (None, 0.0)
    C = {"id": "C", "claim": "no system loses more than 0.5x chance on test papers after the encoders' cutoff",
         "outcome": ("PASSED" if worst[1] <= 0.5 else "FAILED") if drops else
                    (f"NOT ENOUGH DATA ({n_after} queries after the cutoff; >= 50 per task needed)" if cont else "NOT AVAILABLE"),
         "largest_drop": {"system": worst[0], "drop_x_chance": worst[1]} if drops else None}
    out = {"experiment": os.path.basename(exp_dir.rstrip("/")), "n_systems": len(runs), "average": average,
           "seeds": max(len(ms) for ms in runs.values()),
           "best": {"jepa": _arm(bj) if bj else None, "comparison": _arm(bc) if bc else None, "gnn": _arm(bg) if bg else None,
                    "llm": _arm(bl) if bl else None},
           "index": {_arm(t): {"value": float(np.nanmean(boots[t]["index"])) if boots[t] is not None else pts[t]["index"],
                               "ci95": _ci(boots[t]["index"]) if boots[t] is not None else None,
                               "family": fam[t]} for t in runs},
           "hypotheses": H, "contamination": C}
    out["conclusion"] = conclude(out)
    return out


def conclude(f):
    """Plain-language conclusion from the hypothesis outcomes (decision rules fixed in advance)."""
    H = {h["id"]: h for h in f["hypotheses"]}
    ok = lambda k: H.get(k, {}).get("outcome", "").startswith("SUPPORTED")
    bad = lambda k: H.get(k, {}).get("outcome", "").startswith(("CONTRADICTED", "NOT SUPPORTED"))
    lines = []
    if ok("H1"):
        lines.append(f"Main result: a JEPA world model over the knowledge-unit graph ({H['H1']['a']}) reasons better than "
                     f"every comparison system, including the strongest ({H['H1']['b']}).")
        frame = "method"
    elif bad("H1"):
        lines.append(f"Main result: no JEPA model beats the strongest comparison system ({H['H1']['b']}) on the reasoning "
                     f"index; the benchmark exposes a gap that current self-supervised graph models do not close.")
        frame = "benchmark"
    else:
        lines.append("Main result: the best JEPA model and the strongest comparison system are not separable on the "
                     "reasoning index at this scale (the 95% interval of the difference contains 0).")
        frame = "benchmark"
    if ok("H2"):
        lines.append(f"Reading the KUs as a graph with a GNN helps: {H['H2']['a']} beats the graph-transformer JEPA.")
    elif bad("H2"):
        lines.append("A GNN encoder does not beat the graph transformer: full attention over the papers is enough.")
    if ok("H3a") or ok("H3b"):
        lines.append("Action-conditioned (world-model) training makes predictions respond to interventions, a property "
                     "no comparison system has by construction.")
    elif bad("H3a") and bad("H3b"):
        lines.append("Action-conditioned training does not improve the intervention response.")
    if ok("H4"):
        lines.append("JEPA predictions resist counterfactual edits better than text similarity: they depend on content, "
                     "not only on topic and wording.")
    elif bad("H4"):
        lines.append("Counterfactual edits fool the JEPA models at least as often as the comparison systems.")
    if H.get("H5", {}).get("outcome") not in (None, "NOT RUN"):
        lines.append("Against LLM-Reasoning: " + ("the JEPA model is better." if ok("H5") else
                                                  "the LLM is better." if bad("H5") else "no separable difference."))
    c = f.get("contamination", {})
    if c.get("outcome") == "FAILED":
        lines.append(f"Caution: {c['largest_drop']['system']} loses {c['largest_drop']['drop_x_chance']:.2f}x chance after "
                     f"the encoders' cutoff; report the after-cutoff scores as the main numbers.")
    elif c.get("outcome") == "PASSED":
        lines.append("No sign of test-set contamination: scores after the encoders' pretraining cutoff match those before.")
    venue = ("ICML / KDD research track: a method paper (H1 supported)." if frame == "method" else
             "NeurIPS Datasets & Benchmarks / KDD datasets track: a benchmark paper (H1 not supported).")
    lines.append("Suggested framing: " + venue)
    if f.get("seeds", 1) < 3:
        lines.append(f"Only {f.get('seeds')} seed(s): rerun with 3-5 seeds before stating these conclusions.")
    return " ".join(lines)


def write(exp_dir, dashboard=None):
    f = analyse(exp_dir)
    json.dump(f, open(os.path.join(exp_dir, "findings.json"), "w"), indent=1, default=str)
    if dashboard:
        json.dump(f, open(os.path.join(dashboard, f"findings_{os.path.basename(exp_dir.rstrip('/'))}.json"), "w"),
                  indent=1, default=str)
    return f


if __name__ == "__main__":
    r = write(sys.argv[1])
    for h in r.get("hypotheses", []):
        ci = h.get("ci95")
        print(f"{h['id']:<4s} {h['outcome']:<28s} {h['claim']}"
              + (f"  [{h.get('a')} vs {h.get('b')}: {h['difference']:+.3f}" if h.get("difference") is not None else "")
              + (f", 95% CI {ci[0]:+.3f} .. {ci[1]:+.3f}]" if ci and ci[0] is not None else ("]" if h.get("difference") is not None else "")))
    print("C   ", r.get("contamination", {}).get("outcome"))
    print("\nConclusion:", r.get("conclusion"))
