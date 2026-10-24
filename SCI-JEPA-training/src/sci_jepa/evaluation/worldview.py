"""
The model seen as a JEPA world model: how much it relies on graph structure, whether
it predicts the effect of actions, and per-example explanations for the dashboard.

  structure_reliance  balanced MRR / chance with the normal query graphs vs with every
                      typed edge removed (role, proxy, cites, field links). A gap of ~0
                      means the accuracy comes from KU text + paper grouping alone.
  action_eval         world-model test: apply an action (swap a KU for another paper's /
                      flip a supports-contradicts edge), compare the CHANGE of the
                      prediction with the CHANGE of the true target latent.
                        direction_cos  cos(pred_after - pred_before, target_after - target_before);
                                       0 = no idea how the action matters, 1 = perfect
                        direction_hit  share with direction_cos > 0 (chance 0.5)
                        magnitude      |pred change| / |target change| (0 = ignores the action)
  examples            a few test instances explained: which tokens the predictor attended
                      to, how much the prediction changes when each typed edge is removed,
                      where the true KU ranks, and one action with its predicted effect.
"""
import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.evaluation.diagnostics import encode_papers, evaluate
from sci_jepa.data.graph_index import COMP_TYPES, HUB, REL_MEMBER, REL_NONE, REL_SAME_PAPER
from sci_jepa.data.instances import Sampler, add_action, ablate_structure, collate
from sci_jepa.data.ku_schema import REL_DO
from sci_jepa.models.model import to_device
from sci_jepa.dashboard.telemetry import REL_NAMES

_PLAIN = (REL_NONE, REL_SAME_PAPER, REL_MEMBER, REL_DO)
TYPE_NAMES = COMP_TYPES + ["paper hub"]


def _ratio(m):
    rb = m["retrieval_balanced"]
    return rb["MRR"] / max(rb.get("chance_MRR") or 1e-9, 1e-9)


def structure_reliance(model, gi, cfg, dev, split="test", base=None):
    """-> {with, without, gap, share}: ×chance with / without typed edges."""
    base = base or evaluate(model, gi, cfg, dev, None, split=split)
    abl = evaluate(model, gi, cfg, dev, None, split=split, ablate="typed")
    w, wo = _ratio(base), _ratio(abl)
    return {"with": w, "without": wo, "gap": w - wo, "share": (w - wo) / max(w - 1.0, 1e-9)}


@torch.no_grad()
def action_eval(model, gi, cfg, dev, split="val", n=200, seed=11):
    model.eval()
    out = {}
    for kind in ("swap", "flip"):
        s = Sampler(gi, cfg, split, seed)
        cos, mag, left = [], [], n
        while left > 0:
            insts = s.sample(min(cfg.batch_size, left))
            left -= cfg.batch_size
            pairs = [(i, add_action(gi, i, s.rng, s.papers, kind=kind)) for i in insts]
            pairs = [(a, b) for a, b in pairs if b is not None]
            if not pairs:
                continue
            b0 = to_device(collate([a for a, _ in pairs]), dev)
            b1 = to_device(collate([b for _, b in pairs]), dev)
            p0, _ = model.predict(b0)
            p1, _ = model.predict(b1)
            dz = model.encode_target(b1["tgt"], b1["target_pos"]) - model.encode_target(b0["tgt"], b0["target_pos"])
            dp = p1 - p0
            ok = dz.norm(dim=-1) > 1e-6
            cos += F.cosine_similarity(dp[ok], dz[ok], dim=-1).tolist()
            mag += (dp[ok].norm(dim=-1) / dz[ok].norm(dim=-1)).tolist()
        out[kind] = {"direction_cos": float(np.mean(cos)) if cos else None,
                     "direction_hit": float(np.mean(np.asarray(cos) > 0)) if cos else None,
                     "magnitude": float(np.median(mag)) if mag else None, "n": len(cos)}
    model.train()
    return out


# ---------------------------------------------------------------- explanations
def _capture(model, on):
    for blk in list(model.context.blocks) + list(model.predictor.blocks):
        if getattr(blk, "att", None) is None:          # GINE layers have no attention to capture
            continue
        blk.att.capture = on
        if not on:
            blk.att.last_att = None


def _predict(model, insts, dev):
    return model.predict(to_device(collate(insts), dev))[0]


def _snip(gi, c, n=160):
    s = getattr(gi, "comp_snippet", None)
    return str(s[int(c)])[:n] if s is not None and c >= 0 else ""


@torch.no_grad()
def _explain(model, gi, cfg, dev, inst, sampler, rng, n_cand=60):
    ctx, m = inst["ctx"], int(inst["mask_pos"])
    papers = [int(p) for p in inst["papers"]]
    # 1. what the predictor looked at (attention of the mask position, mean over heads and predictor layers)
    _capture(model, True)
    pred = _predict(model, [inst], dev)
    T = len(ctx["type"])
    pa = [blk.att.last_att[0, :, m, :].mean(0) for blk in model.predictor.blocks if getattr(blk, "att", None) is not None]
    att = torch.stack(pa).mean(0) if pa else torch.zeros(T)          # no attention (GINE): shown as zero
    last = model.context.blocks[-1]
    enc_att = last.att.last_att[0, :, m, :].mean(0) if getattr(last, "att", None) is not None else torch.zeros(T)
    _capture(model, False)
    att, enc_att = att.float().cpu().numpy(), enc_att.float().cpu().numpy()
    # 2. edge importance: remove one typed edge at a time
    rel = ctx["rel"]
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(~np.isin(rel, _PLAIN))))][:150]
    imp = []
    if edges:
        variants = []
        for i, j in edges:
            c2 = {**ctx, "rel": ctx["rel"].copy()}
            c2["rel"][i, j] = c2["rel"][j, i] = REL_SAME_PAPER if ctx["slot"][i] == ctx["slot"][j] else REL_NONE
            variants.append({**inst, "ctx": c2})
        P = torch.cat([_predict(model, variants[k:k + 64], dev) for k in range(0, len(variants), 64)])
        imp = (1 - F.cosine_similarity(P, pred.expand_as(P), dim=-1)).clamp_min(0).cpu().numpy()
    no_struct = float(1 - F.cosine_similarity(_predict(model, [ablate_structure(inst)], dev), pred).item())
    # 3. ranking among same-type KUs of related papers
    t = int(inst["target_type"]); p0 = papers[0]
    same_sub = [q for q in gi.subfield_papers[int(gi.paper_subfield[p0])] if q != p0 and q in set(sampler.papers.tolist())]
    others = list(rng.permutation(same_sub))[:n_cand] if same_sub else []
    if len(others) < n_cand:
        rest = [int(q) for q in rng.permutation(sampler.papers) if int(q) != p0 and int(q) not in others]
        others += rest[:n_cand - len(others)]
    ids, H = encode_papers(model, gi, np.asarray([p0] + [int(q) for q in others]), cfg, dev, "target")
    ids = ids.numpy()
    keep = (gi.comp_type[ids] == t) & ((gi.comp_paper[ids] != p0) | (ids == inst["target_comp"]))
    ids, H = ids[keep], H[torch.as_tensor(keep)]
    sc = (F.normalize(pred.float().cpu(), dim=-1) @ F.normalize(H, dim=-1).T)[0].numpy()
    order = np.argsort(-sc)
    gold = int(np.nonzero(ids == inst["target_comp"])[0][0]) if (ids == inst["target_comp"]).any() else None
    rank = int(np.nonzero(order == gold)[0][0]) + 1 if gold is not None else None
    top = [{"text": _snip(gi, ids[k]), "same_paper": bool(gi.comp_paper[ids[k]] == p0), "score": float(sc[k]),
            "gold": bool(k == gold)} for k in order[:3]]
    # 4. one action and its predicted effect
    act = None
    for kind in ("swap", "flip"):
        a = add_action(gi, inst, rng, sampler.papers, kind=kind)
        if a is None:
            continue
        b0, b1 = to_device(collate([inst]), dev), to_device(collate([a]), dev)
        dp = model.predict(b1)[0] - pred
        dz = model.encode_target(b1["tgt"], b1["target_pos"]) - model.encode_target(b0["tgt"], b0["target_pos"])
        info = dict(a["action"])
        if kind == "swap":
            info["old_text"], info["new_text"] = _snip(gi, info["node"], 120), _snip(gi, info["donor"], 120)
        act = {**info, "direction_cos": float(F.cosine_similarity(dp, dz).item()),
               "magnitude": float(dp.norm() / dz.norm().clamp_min(1e-8))}
        break
    # 5. nodes and edges for the graph view (plus the KUs hidden from the model, as ghosts)
    nodes, slot_paper = [], {k: papers[k] for k in range(len(papers))}
    for k in range(len(ctx["type"])):
        c, ty = int(ctx["comp"][k]), int(ctx["type"][k])
        p = slot_paper.get(int(ctx["slot"][k]), -1)
        nodes.append({"id": k, "type": TYPE_NAMES[ty] if ty < len(TYPE_NAMES) else str(ty),
                      "paper": str(gi.paper_ids[p]) if p >= 0 else "", "slot": int(ctx["slot"][k]),
                      "text": (str(gi.field_names[gi.paper_field[p]]) if ty == HUB else _snip(gi, c)) if k != m
                      else _snip(gi, c),
                      "masked": k == m, "att": float(att[k]), "enc_att": float(enc_att[k])})
    edge_rows = [{"from": i, "to": j, "rel": REL_NAMES[int(rel[i, j])], "importance": float(v),
                  "inter": bool(ctx["slot"][i] != ctx["slot"][j])} for (i, j), v in zip(edges, imp)]
    member = [{"from": i, "to": j} for i, j in zip(*np.nonzero(np.triu(rel == REL_MEMBER)))]
    tv = inst["target_view"]
    in_ctx = set(int(c) for c in ctx["comp"] if c >= 0)
    ghosts, gedges = [], []
    pos_tv = {int(c): k for k, c in enumerate(tv["comp"]) if c >= 0}
    anchor_tv = pos_tv.get(int(inst["target_comp"]))
    gid = {}
    for c, k in pos_tv.items():
        if c not in in_ctx:
            gid[c] = f"g{c}"
            ghosts.append({"id": gid[c], "type": COMP_TYPES[int(tv["type"][k])], "text": _snip(gi, c), "hidden": True})
    if anchor_tv is not None:                      # the hidden KU's true role edges (removed from the context)
        for c, k in pos_tv.items():
            r = int(tv["rel"][anchor_tv, k])
            if c != int(inst["target_comp"]) and r not in _PLAIN:
                src = gid.get(c) if c in gid else int(np.nonzero(ctx["comp"] == c)[0][0])
                gedges.append({"from": m, "to": src, "rel": REL_NAMES[r]})
    order_att = np.argsort(-att)
    top_att = [{"id": int(k), "type": nodes[k]["type"], "own_paper": nodes[k]["slot"] == 0, "share": float(att[k])}
               for k in order_att[:5] if k != m]
    return {"type": COMP_TYPES[t], "paper": str(gi.paper_ids[p0]), "rank": rank, "pool": int(len(ids)),
            "top": top, "nodes": nodes, "edges": edge_rows, "member": member, "ghosts": ghosts,
            "hidden_edges": gedges, "top_attention": top_att, "no_structure_shift": no_struct,
            "max_edge_shift": float(max(imp)) if len(imp) else 0.0, "action": act}


@torch.no_grad()
def examples(model, gi, cfg, dev, split="test", n=6, seed=5):
    model.eval()
    rng = np.random.default_rng(seed)
    want = [t for t in ("result", "method", "claim", "evidence", "question", "limitation", "procedure",
                        "context", "interpretation", "implication") if t in cfg.target_types][:n]
    out = []
    for t in want:
        s = Sampler(gi, cfg.override(target_types=(t,)), split, seed)
        insts = s.sample(1)
        if insts:
            out.append(_explain(model, gi, cfg, dev, insts[0], s, rng))
    model.train()
    return out
