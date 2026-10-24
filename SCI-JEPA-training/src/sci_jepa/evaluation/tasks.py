"""
Reasoning-task evaluation matrix: the same trained model, queried in five ways.

The normalized parquet has no separate Problem / Material / Condition fields, so
the plan's slots are mapped onto KU types:

  Problem              context + question
  Method               method
  Material, Condition  procedure (setups, architectures, data, conditions)
  Result               result
  Claim / Hypothesis   claim (+ question, which holds the hypothesis)

  forward        mask result               given {problem, method, material/condition}
  inverse        mask method / procedure   given {problem, result}
  abduction      mask claim / question     given {result, condition}
  consistency    hide claim-evidence edges inside a paper; does the inverse-relation
                 head tell supported_by from challenged_by? (balanced accuracy, chance
                 0.5). Inter-paper supports / contradicts edges do not exist in the
                 data (papers are linked only by citations / field / proxy edges), so
                 the inter-paper variant is reported as not available.
  intervention   forward query with change(Condition := c') or change(Method := m'):
                 the paper's procedure / method KU is replaced by a donor paper's
                 (same subfield). The model has no z_do input, so this measures INPUT
                 interventions: does the predicted result move (rel_shift: change caused
                 by the swap / typical change between two unrelated queries; 0 = ignores
                 the condition, 1 = as large as switching paper), and toward the donor's
                 actual result (direction after removing the mean result vector)?

"given" means: in the target's own paper only these KU types stay visible (all
others are removed); neighbouring papers are unchanged. Retrieval is ranked like
the main balanced metric: the true KU among K same-type KUs of related papers.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.evaluation.diagnostics import _chance, _ranks, centred_cos, encode_papers, hard_pool
from sci_jepa.data.instances import build_view
from sci_jepa.data.graph_index import COMP_TYPES, REL_ID, REL_NONE, REL_SAME_PAPER
from sci_jepa.data.instances import CTX_TYPE, PROBE_NONE, Sampler, collate
from sci_jepa.data.ku_schema import TEMPLATES as _TPL
from sci_jepa.models.model import to_device

TASKS = {                              # given / targets come from ku_schema.TEMPLATES (shared with
    "forward": {"label": "Forward prediction",          # ku_mode="paper_masked", which encodes
                "query": "mask Result | given Problem, Method, Material/Condition"},   # these texts)
    "inverse": {"label": "Experiment design (inverse)",
                "query": "mask Method / Material | given Problem, Result"},
    "abduction": {"label": "Abduction",
                  "query": "mask Claim / Hypothesis | given Results, Conditions"},
}
for _k in TASKS:
    TASKS[_k]["given"], TASKS[_k]["targets"] = _TPL[_k]
TASK_ORDER = ["forward", "inverse", "abduction", "consistency", "intervention"]
TASK_LABEL = {**{k: v["label"] for k, v in TASKS.items()},
              "consistency": "Consistency checking", "intervention": "Intervention conditioning"}
SUP, CON = REL_ID["supported_by"] - 3, REL_ID["challenged_by"] - 3     # inverse-head classes


def _sampler(gi, cfg, split, seed, target, given):
    s = Sampler(gi, cfg.override(target_types=(target,)), split, seed)
    s.given = set(given) if given is not None else None
    return s


def _retrieval(model, gi, cfg, dev, split, target, given, n, bank, rng, seed):
    """Balanced MRR of `target` KUs queried with only `given` visible in their paper."""
    ids, H_t, pos_of, ctype, bank_paper = bank
    t = COMP_TYPES.index(target)
    same = np.nonzero(ctype == t)[0]
    if not len(same):
        return None
    s = _sampler(gi, cfg, split, seed, target, given)
    if not s.n_anchors():
        return None
    ranks, pools, left, mp_pair, mp_all, mp_k, years, doms, sims, dis = [], [], n, [], [], [], [], [], [], []
    mu = H_t.float().mean(0, keepdim=True).to(dev)     # similarity rule: centred cosine (diagnostics.centred_cos)
    while left > 0:
        insts = [i for i in s.sample(min(cfg.batch_size, left)) if i["target_comp"] in pos_of]
        left -= cfg.batch_size
        if not insts:
            continue
        pred, _ = model.predict(to_device(collate(insts), dev))
        for b, inst in enumerate(insts):
            g = int(inst["target_comp"])
            others = hard_pool(gi, int(inst["papers"][0]), same, bank_paper, cfg.balanced_k, rng, mode=cfg.pool_mode)
            q = F.normalize(pred[b:b + 1], dim=-1)
            s_oth = (q @ F.normalize(H_t[others].to(dev), dim=-1).T)[0] if others else torch.zeros(0, device=dev)
            var = gi.mp_of.get(g, []) if cfg.minimal_pairs else []
            if var:                                    # true KU + its counterfactuals, same paper view
                K_raw = variant_keys(model, gi, cfg, dev, g, var)
                K, key = F.normalize(K_raw, dim=-1), K_raw[:1]
                s_g, s_v = (q @ K[:1].T)[0, 0], (q @ K[1:].T)[0]
                mp_pair += (s_g > s_v).float().tolist(); mp_all.append(float((s_g > s_v).all()))
                mp_k += [int(gi.mp_kind[r]) for r in var]
            else:
                key = H_t[pos_of[g]:pos_of[g] + 1].to(dev)
                s_g, s_v = (q @ F.normalize(key, dim=-1).T)[0, 0], torch.zeros(0, device=dev)
            ranks.append(1 + int((s_oth > s_g).sum()) + int((s_v > s_g).sum()))
            sims.append(float(centred_cos(pred[b:b + 1], key, mu)[0]))   # similarity rule: to the true KU ...
            dis.append(float(centred_cos(pred[b:b + 1], H_t[others].to(dev), mu).max())   # ... and the best wrong KU
                       if others else float("nan"))
            pools.append(1 + len(others) + len(var))
            years.append(int(gi.paper_year[int(inst["papers"][0])])); doms.append(_dom(gi, int(inst["papers"][0])))
    if not ranks:
        return None
    r = np.asarray(ranks)
    ch = float(np.mean([_chance(k) for k in pools]))
    mrr = float((1 / r).mean())
    thr = getattr(cfg, "sim_threshold", 0.8)
    return {"MRR": mrr, "chance": ch, "ratio": mrr / max(ch, 1e-9), "Hits@1": float((r <= 1).mean()),
            "Hits@10": float((r <= 10).mean()), "n": int(len(r)),
            "acc_sim": float((np.asarray(sims) >= thr).mean()), "sim_mean": float(np.mean(sims)), "sim_threshold": thr,
            "acc_sim_top1": float(((np.asarray(sims) >= thr) & (r == 1)).mean()),
            "false_accept": float(np.nanmean(np.asarray(dis) >= thr)) if np.isfinite(dis).any() else None,
            "mp_pair": float(np.mean(mp_pair)) if mp_pair else None,     # true > each variant (chance 0.5)
            "mp_all": float(np.mean(mp_all)) if mp_all else None, "mp_n": len(mp_all),
            "mp_pairs_by_kind": list(zip(mp_k, mp_pair)),
            "q": [[float(1 / a), float(_chance(b)), int(y), int(d)] for a, b, y, d in zip(r, pools, years, doms)]}   # per query


@torch.no_grad()
def variant_keys(model, gi, cfg, dev, comp, rows):
    """Target-encoder keys of a KU and of its minimal-pair variants, each encoded inside the
    KU's own paper (the paper alone, as for the candidate bank) -> (1 + len(rows), d)."""
    p0 = int(gi.comp_paper[comp])
    comps = gi.paper_comps[p0]
    if len(comps) > cfg.max_comp_per_paper:
        comps = np.concatenate([[comp], [c for c in comps if c != comp][:cfg.max_comp_per_paper - 1]])
    view = build_view(gi, [p0], [np.asarray(comps)], cfg)
    k = int(np.nonzero(view["comp"] == comp)[0][0])
    views = [view]
    for r in rows:
        v = {**view, "x": view["x"].copy()}
        v["x"][k] = gi.mp_x[r].astype(np.float32)
        views.append(v)
    from sci_jepa.data.instances import collate_views
    z = model.target(to_device(collate_views(views), dev))
    return z[:, k]


def _by_kind(pairs):
    """(kind id, true > variant) pairs -> share per kind of edit (direction / negation / number / substitution)."""
    from sci_jepa.data.ku_schema import MP_KINDS
    out = {}
    for k, v in pairs:
        out.setdefault(MP_KINDS[k], []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def _merge(parts):
    parts = {k: v for k, v in parts.items() if v}
    if not parts:
        return {"MRR": None, "n": 0, "by_target": {}}
    n = sum(v["n"] for v in parts.values())
    avg = lambda k: float(sum(v[k] * v["n"] for v in parts.values()) / n)
    mrr, ch = avg("MRR"), avg("chance")
    mp = [(v["mp_pair"], v["mp_all"], v["mp_n"]) for v in parts.values() if v.get("mp_pair") is not None]
    sim = {k: avg(k) for k in ("acc_sim", "sim_mean", "acc_sim_top1", "false_accept")
           if all(v.get(k) is not None for v in parts.values())}
    mpn = sum(x[2] for x in mp)
    return {"MRR": mrr, "chance": ch, "ratio": mrr / max(ch, 1e-9), "Hits@1": avg("Hits@1"),
            "Hits@10": avg("Hits@10"), "n": n, "by_target": parts, **sim,
            "mp_pair": float(sum(a * w for a, _, w in mp) / mpn) if mpn else None,
            "mp_all": float(sum(b * w for _, b, w in mp) / mpn) if mpn else None, "mp_n": mpn,
            "mp_by_kind": _by_kind([kv for v in parts.values() for kv in v.get("mp_pairs_by_kind", [])]),
            "q": [x for v in parts.values() for x in v.get("q", [])]}


def _consistency(model, gi, cfg, dev, split, n, seed):
    """Hide every claim-evidence edge of the target paper; 2-way support vs contradict
    from the inverse-relation head."""
    s = _sampler(gi, cfg, split, seed, "context", None)   # masking context keeps every claim visible
    s.given = None
    rel_s, rel_c = REL_ID["supported_by"], REL_ID["challenged_by"]
    y, yhat, yr, dm, left = [], [], [], [], n
    while left > 0:
        insts = s.sample(min(cfg.batch_size, left))
        left -= cfg.batch_size
        probes = []
        for b, inst in enumerate(insts):
            ctx = inst["ctx"]
            ii, jj = np.nonzero(np.triu(np.isin(ctx["rel"], [rel_s, rel_c])))
            for i, j in zip(ii.tolist(), jj.tolist()):
                lab = int(ctx["rel"][i, j] == rel_c)
                base = REL_SAME_PAPER if ctx["slot"][i] == ctx["slot"][j] else REL_NONE
                ctx["rel"][i, j] = ctx["rel"][j, i] = base
                probes.append((b, i, j, lab))
            inst["probes"], inst["probe_labels"] = [], []
        if not probes:
            continue
        batch = to_device(collate(insts), dev)
        _, h = model.predict(batch)
        b_, i_, j_, lab = (torch.tensor(v, device=dev) for v in zip(*probes))
        logit = model.inverse(h[b_, i_], h[b_, j_])
        yhat += (logit[:, CON] > logit[:, SUP]).long().tolist()
        y += lab.tolist()
        yr += [int(gi.paper_year[int(insts[b]["papers"][0])]) for b, _, _, _ in probes]
        dm += [_dom(gi, int(insts[b]["papers"][0])) for b, _, _, _ in probes]
    y, yhat = np.asarray(y), np.asarray(yhat)
    if not len(y):
        return {"bal_acc": None, "n": 0}
    rec = [float((yhat[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return {"bal_acc": float(np.mean(rec)), "acc": float((y == yhat).mean()), "chance": 0.5,
            "n": int(len(y)), "n_supports": int((y == 0).sum()), "n_contradicts": int((y == 1).sum()),
            "recall_supports": rec[0] if (y == 0).any() else None,
            "recall_contradicts": rec[-1] if (y == 1).any() else None,
            "q": [[int(a), int(b), int(c), int(d)] for a, b, c, d in zip(y, yhat, yr, dm)],
            "inter_paper": "not available: the data has no supports / contradicts edges BETWEEN papers "
                           "(papers are linked only by citations, field and proxy edges)"}


def _intervention(model, gi, cfg, dev, split, n, bank, rng, seed):
    """change(Condition := c') / change(Method := m') on forward queries."""
    ids, H_t, pos_of, ctype, bank_paper = bank
    t_res = COMP_TYPES.index("result")
    res_of = {int(gi.comp_paper[int(ids[k])]): k for k in np.nonzero(ctype == t_res)[0]}
    allowed = set(int(p) for p in gi.papers(split))
    centre = H_t[np.nonzero(ctype == t_res)[0]].mean(0)          # remove the shared "result" direction
    split_res = [q for q in res_of if q in allowed]

    def donor_for(p0, papers):
        for pool in (gi.subfield_papers[int(gi.paper_subfield[p0])], gi.field_papers[int(gi.paper_field[p0])],
                     split_res):
            c = [q for q in pool if q != p0 and q in allowed and q in res_of and q not in papers]
            if c:
                return int(rng.choice(c))
        return None
    out = {}
    for name, swap in (("condition", "procedure"), ("method", "method")):
        t_sw = COMP_TYPES.index(swap)
        s = _sampler(gi, cfg, split, seed + 1, "result", TASKS["forward"]["given"])
        shift, base, toward, away, iyr, idm, left = [], [], [], [], [], [], n
        while left > 0:
            insts = s.sample(min(cfg.batch_size, left))
            left -= cfg.batch_size
            cf, keep = [], []
            for b, inst in enumerate(insts):
                p0 = int(inst["papers"][0]); ctx = inst["ctx"]
                if cfg.ku_mode == "paper_masked":     # re-encoded text with the donor's section
                    slot0 = np.nonzero((ctx["slot"] == 0) & (ctx["type"] == CTX_TYPE))[0]
                    hit = next((h for h in gi.cf_of.get((p0, name), ())
                                if h[1] in res_of and h[1] not in inst["papers"]), None)
                    if not len(slot0) or hit is None:
                        continue
                    d = hit[1]
                    x = ctx["x"].copy(); x[slot0] = gi.cf_x[hit[0]].astype(np.float32)
                    cf.append({**inst, "ctx": {**ctx, "x": x}}); keep.append((b, d, p0))
                    continue
                slot0 = np.nonzero((ctx["slot"] == 0) & (ctx["type"] == t_sw))[0]
                d = donor_for(p0, inst["papers"])
                if not len(slot0) or d is None:
                    continue
                dc = [int(c) for c in gi.paper_comps[d] if gi.comp_type[c] == t_sw]
                if not dc:
                    continue
                x = ctx["x"].copy(); x[slot0] = gi.comp_x[dc[0]].astype(np.float32)
                cf.append({**inst, "ctx": {**ctx, "x": x}}); keep.append((b, d, p0))
            if not cf:
                continue
            p1, _ = model.predict(to_device(collate([insts[b] for b, _, _ in keep]), dev))
            p2, _ = model.predict(to_device(collate(cf), dev))
            z1, z2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)
            c = centre.to(dev)
            donor = H_t[[res_of[d] for _, d, _ in keep]].to(dev) - c
            own = [res_of.get(p0) for _, _, p0 in keep]
            shift += (1 - (z1 * z2).sum(-1)).tolist()
            if len(z1) > 1:                                       # typical gap between unrelated queries
                off = ~torch.eye(len(z1), dtype=torch.bool, device=dev)
                base.append(float((1 - z1 @ z1.T)[off].mean()))
            toward += (((z2 - z1) * donor).sum(-1) > 0).float().tolist()
            iyr += [int(gi.paper_year[p0]) for _, _, p0 in keep]
            idm += [_dom(gi, p0) for _, _, p0 in keep]
            for k, o in enumerate(own):
                if o is not None:
                    away.append(float(((z2[k] - z1[k]) * (H_t[o].to(dev) - c)).sum() < 0))
        rel = float(np.mean(shift) / max(np.mean(base), 1e-12)) if shift and base else None
        out[name] = {"swap": swap, "shift": float(np.mean(shift)) if shift else None, "rel_shift": rel,
                     "toward_donor": float(np.mean(toward)) if toward else None,
                     "away_from_own": float(np.mean(away)) if away else None, "chance": 0.5,
                     "n": int(len(shift)),
                     "q": [[float(t), float(sh), int(y), int(d)] for t, sh, y, d in zip(toward, shift, iyr, idm)],
                     "base_gap": float(np.mean(base)) if base else None}
    out["note"] = "input intervention (the model has no z_do conditioning yet)"
    return out


@torch.no_grad()
def task_eval(model, gi, cfg, dev, split="test", n=None, seed=4321):
    """-> {task: metrics} for the five reasoning tasks (section mode only)."""
    if cfg.ku_mode == "paper":
        return {"note": "not applicable with ku_mode='paper': the tasks mask one KU type given other "
                        "KU types OF THE SAME PAPER, and a paper is a single KU"}
    n = n or cfg.task_eval_instances
    model.eval()
    papers = np.asarray(gi.papers(split))
    ids, H_t = encode_papers(model, gi, papers, cfg, dev, "target")
    pos_of = {int(c): k for k, c in enumerate(ids.tolist())}
    ctype = gi.comp_type[ids.numpy()]
    bank = (ids, H_t, pos_of, ctype, gi.comp_paper[ids.numpy()])
    rng = np.random.default_rng(seed)
    out = {}
    for key, t in TASKS.items():
        per = {tg: _retrieval(model, gi, cfg, dev, split, tg, t["given"], max(n // len(t["targets"]), 16),
                              bank, rng, seed) for tg in t["targets"]}
        out[key] = {**_merge(per), "label": t["label"], "query": t["query"]}
    out["consistency"] = {**_consistency(model, gi, cfg, dev, split, n, seed), "label": TASK_LABEL["consistency"],
                          "query": "hide supports / contradicts edges (claim-evidence) | classify them"}
    out["intervention"] = {**_intervention(model, gi, cfg, dev, split, max(n // 2, 16), bank, rng, seed),
                           "label": TASK_LABEL["intervention"],
                           "query": "forward query with change(Condition := c') / change(Method := m')"}
    add_periods(out, cfg.encoder_cutoff_year)
    add_domains(out, gi, cfg.min_domain_queries, cfg.min_domains)
    model.train()
    return out


def _dom(gi, p):
    return int(gi.paper_domain[p]) if hasattr(gi, "paper_domain") else -1


def add_domains(tasks, gi, min_n=20, min_domains=3):
    """Every task scored per canonical domain (q[3]); macro = mean over domains with >= min_n queries (each
    domain weighs the same), worst = the weakest such domain. tasks["macro"] holds the macro scores and the
    macro reasoning index."""
    names = list(getattr(gi, "domain_names", []) or [])
    if not names:
        return tasks
    macro = {}
    for k in ("forward", "inverse", "abduction"):
        q = [x for x in ((tasks.get(k) or {}).get("q") or []) if len(x) > 3]
        by = {}
        for d in sorted({x[3] for x in q}):
            g = [x for x in q if x[3] == d]
            rr, ch = np.mean([x[0] for x in g]), np.mean([x[1] for x in g])
            by[names[d] if 0 <= d < len(names) else str(d)] = {"ratio": float(rr / max(ch, 1e-9)), "n": len(g)}
        ok = {d: v for d, v in by.items() if v["n"] >= min_n}
        if by:
            tasks[k]["by_domain"] = by
        if len(ok) >= min_domains:
            tasks[k]["macro_ratio"] = float(np.mean([v["ratio"] for v in ok.values()]))
            w = min(ok, key=lambda d: ok[d]["ratio"])
            tasks[k]["worst_domain"] = {"domain": w, **ok[w]}
            macro[k] = tasks[k]["macro_ratio"]
        elif tasks[k].get("ratio") is not None:          # too few domains with enough queries: micro score
            macro[k] = tasks[k]["ratio"]; tasks[k]["macro_fallback"] = "micro (fewer than min_domains domains)"
    q = [x for x in ((tasks.get("consistency") or {}).get("q") or []) if len(x) > 3]
    by = {}
    for d in sorted({x[3] for x in q}):
        g = np.asarray([x[:2] for x in q if x[3] == d])
        rec = [float((g[g[:, 0] == c, 1] == c).mean()) for c in (0, 1) if (g[:, 0] == c).any()]
        by[names[d] if 0 <= d < len(names) else str(d)] = {"bal_acc": float(np.mean(rec)), "n": int(len(g))}
    ok = {d: v for d, v in by.items() if v["n"] >= min_n}
    if by:
        tasks["consistency"]["by_domain"] = by
    if len(ok) >= min_domains:
        macro["consistency"] = tasks["consistency"]["macro_bal_acc"] = float(np.mean([v["bal_acc"] for v in ok.values()]))
    elif (tasks.get("consistency") or {}).get("bal_acc") is not None:
        macro["consistency"] = tasks["consistency"]["bal_acc"]
    iv = (tasks.get("intervention") or {}).get("condition") or {}
    q = [x for x in (iv.get("q") or []) if len(x) > 3]
    by = {}
    for d in sorted({x[3] for x in q}):
        g = [x[0] for x in q if x[3] == d]
        by[names[d] if 0 <= d < len(names) else str(d)] = {"toward_donor": float(np.mean(g)), "n": len(g)}
    ok = {d: v for d, v in by.items() if v["n"] >= max(min_n // 2, 5)}
    if by:
        iv["by_domain"] = by
    if len(ok) >= min_domains:
        macro["toward"] = iv["macro_toward"] = float(np.mean([v["toward_donor"] for v in ok.values()]))
    elif iv.get("toward_donor") is not None:
        macro["toward"] = iv["toward_donor"]
    if macro:
        v = [macro.get("forward"), macro.get("inverse"), macro.get("abduction"),
             macro["consistency"] / 0.5 if "consistency" in macro else None,
             macro["toward"] / 0.5 if "toward" in macro else None]
        v = [x for x in v if x is not None]
        macro["index"] = float(np.mean(v)) if v else None
        macro["min_queries"] = min_n
        tasks["macro"] = macro
    tasks["domain_names"] = names
    return tasks


def add_periods(tasks, cutoff):
    """Every task scored separately for test papers up to / after the text encoder's pretraining cutoff
    (contamination check): tasks[k]["by_period"] = {"<= cutoff": {...}, "> cutoff": {...}}."""
    if not cutoff:
        return tasks
    groups = (("until_cutoff", lambda y: y <= cutoff), ("after_cutoff", lambda y: y > cutoff))
    for k in ("forward", "inverse", "abduction"):
        q = (tasks.get(k) or {}).get("q") or []
        out = {}
        for name, f in groups:
            g = [x for x in q if f(x[2])]
            if g:
                rr, ch = np.mean([x[0] for x in g]), np.mean([x[1] for x in g])
                out[name] = {"MRR": float(rr), "chance": float(ch), "ratio": float(rr / max(ch, 1e-9)), "n": len(g)}
        if out:
            tasks[k]["by_period"] = out
    q = (tasks.get("consistency") or {}).get("q") or []
    out = {}
    for name, f in groups:
        g = np.asarray([x[:2] for x in q if f(x[2])])
        if len(g):
            rec = [float((g[g[:, 0] == c, 1] == c).mean()) for c in (0, 1) if (g[:, 0] == c).any()]
            out[name] = {"bal_acc": float(np.mean(rec)), "n": int(len(g))}
    if out:
        tasks["consistency"]["by_period"] = out
    iv = ((tasks.get("intervention") or {}).get("condition") or {})
    q = iv.get("q") or []
    out = {}
    for name, f in groups:
        g = [x for x in q if f(x[2])]
        if g:
            out[name] = {"toward_donor": float(np.mean([x[0] for x in g])), "n": len(g)}
    if out:
        iv["by_period"] = out
    tasks["cutoff_year"] = cutoff
    return tasks


def reasoning_index(tasks, average="micro"):
    """Mean of the five task scores on one x-chance scale (consistency and intervention / 0.5), as
    in the dashboard's Benchmark tab and the report. average="macro": the same over domain-macro scores
    (each canonical domain weighs the same; falls back to micro without domain data). None if nothing scored."""
    if not tasks or ("note" in tasks and len(tasks) == 1):
        return None
    if average == "macro" and (tasks.get("macro") or {}).get("index") is not None:
        return tasks["macro"]["index"]
    v = [tasks.get(k, {}).get("ratio") for k in ("forward", "inverse", "abduction")]
    c = (tasks.get("consistency") or {}).get("bal_acc")
    i = ((tasks.get("intervention") or {}).get("condition") or {}).get("toward_donor")
    v += [c / 0.5 if c is not None else None, i / 0.5 if i is not None else None]
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else None


def task_summary(tasks):
    """One number per task for tables: retrieval ratio to chance, consistency balanced
    accuracy, intervention share moving toward the donor's result."""
    if not tasks or "note" in tasks and len(tasks) == 1:
        return {}
    s = {k: tasks[k].get("ratio") for k in TASKS if k in tasks}
    s["consistency"] = tasks.get("consistency", {}).get("bal_acc")
    iv = tasks.get("intervention", {})
    s["intervention"] = iv.get("condition", {}).get("toward_donor")
    return s
