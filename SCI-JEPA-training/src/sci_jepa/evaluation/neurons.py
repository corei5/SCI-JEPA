"""
Neuron-level interpretation, in plain language.

For every neuron (one dimension of a layer's output) of every context-encoder
layer, and of the predictor's output, measure WHAT its activation varies with:

  omega_type   share of its variance explained by the KU type of the token
  omega_paper  ... by WHICH PAPER the token comes from (type effect removed)
  omega_field  ... by the research field (type effect removed)
  r_leak       at the masked token: correlation with the paraphrase-leak flag
               (a visible KU >= audit_tau similar to the hidden target)
  r_target     predictor only: correlation between this output dimension and
               the same dimension of the true target, WITHIN each KU type
               (high = this dimension learned the target's content, not just its type)

omega^2 is the bias-corrected effect size (SSB - (k-1) MSW) / (SST + MSW). Unlike
plain eta^2 it stays near 0 under chance even with thousands of small groups
(papers), so "explains 60% of the variance" means what it says.

Each neuron gets ONE human-readable label (first rule that fires):

  dead          its activation never changes -> carries nothing
  identity      >= 50% of its variance tells papers apart -> the A1 shortcut:
                the model can satisfy the loss by recognising the paper
  type          >= 50% tells KU types apart -> knows "this is a claim/result/..."
  field         >= 30% tells research fields apart
  leak          |r| >= 0.3 with the paraphrase-leak flag at the masked token
  content       none of the above: a distributed content / relational feature
"""
import numpy as np
import torch

from sci_jepa.data.graph_index import HUB
from sci_jepa.data.instances import Sampler, collate
from sci_jepa.data.ku_schema import COMP_TYPES
from sci_jepa.models.model import to_device

TH = {"dead_rel_std": 1e-3, "identity": 0.5, "type": 0.5, "field": 0.3, "leak": 0.3, "learned": 0.5}
CATS = ["dead", "identity", "type", "field", "leak", "content"]
CAT_TEXT = {
    "dead": "dead: never changes, carries no information",
    "identity": "paper-identity neuron: tells papers apart (shortcut / leak risk)",
    "type": "KU-type detector",
    "field": "research-field detector",
    "leak": "tracks paraphrase leaks at the masked token",
    "content": "content / relational feature (distributed)",
}


def omega2(X, g):
    """X (n, d), g (n,) int group ids -> (d,) bias-corrected variance share."""
    n, d = X.shape
    _, g = torch.unique(g, return_inverse=True)
    k = int(g.max()) + 1
    if k < 2 or n <= k:
        return torch.zeros(d)
    cnt = torch.zeros(k).index_add_(0, g, torch.ones(n))
    sums = torch.zeros(k, d).index_add_(0, g, X)
    mu = X.mean(0)
    means = sums / cnt[:, None]
    ssb = (cnt[:, None] * (means - mu) ** 2).sum(0)
    sst = ((X - mu) ** 2).sum(0)
    msw = (sst - ssb) / (n - k)
    return ((ssb - (k - 1) * msw) / (sst + msw).clamp_min(1e-12)).clamp(0, 1)


def _type_centre(X, t):
    Xc = X.clone()
    for u in t.unique():
        Xc[t == u] -= X[t == u].mean(0)
    return Xc


def _corr(a, b):
    """columnwise Pearson r between (n, d) a and (n,) or (n, d) b."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    if b.dim() == 1:
        b = b[:, None]
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)


def _top_group(X, g, names):
    """for each neuron, the group with the highest mean z-scored activation."""
    z = (X - X.mean(0)) / X.std(0).clamp_min(1e-8)
    _, inv = torch.unique(g, return_inverse=True)
    keys = torch.unique(g)
    k = len(keys)
    m = torch.zeros(k, X.shape[1]).index_add_(0, inv, z) / torch.bincount(inv, minlength=k)[:, None].float()
    best = m.argmax(0)
    return [names(int(keys[i])) for i in best]


def _label(cat, top_type, top_field, extra=""):
    if cat == "type":
        return f"KU-type detector: strongest for '{top_type}'"
    if cat == "field":
        return f"research-field detector: strongest for '{top_field}'"
    return CAT_TEXT[cat] + extra


@torch.no_grad()
def neuron_report(model, gi, cfg, dev, split="val", n_instances=128, max_tokens=12000, seed=7, rows=True):
    model.eval()
    sampler = Sampler(gi, cfg, split, seed)
    insts = []
    while len(insts) < n_instances:
        got = sampler.sample(min(64, n_instances - len(insts)))
        if not got:
            break
        insts += got
    if not insts:
        model.train()
        return None

    L = len(model.context.blocks)
    acts = [[] for _ in range(L + 1)]                    # blocks + final norm
    mask_acts = [[] for _ in range(L + 1)]
    tok_type, tok_paper, tok_field = [], [], []
    flagged, ttype, preds, targets = [], [], [], []
    for s in range(0, len(insts), 32):
        chunk = insts[s:s + 32]
        batch = to_device(collate(chunk), dev)
        c = batch["ctx"]
        out, layers = model.context(c, return_layers=True)
        layers = layers + [out]
        pred = model.predictor(out, c, batch["mask_pos"], batch["target_type"],
                               target_pos_enc=model._target_pos_enc(batch))
        tgt = model.encode_target(batch["tgt"], batch["target_pos"])
        vis = (~c["pad"]) & (c["type"] != HUB) & (~c["is_mask"])
        b_idx, t_idx = vis.nonzero(as_tuple=True)
        papers = torch.tensor([[inst["papers"][int(sl)] if sl >= 0 else -1 for sl in c["slot"][b].tolist()]
                               for b, inst in enumerate(chunk)])
        for li, h in enumerate(layers):
            acts[li].append(h[b_idx, t_idx].float().cpu())
            mask_acts[li].append(h[torch.arange(len(chunk), device=h.device), batch["mask_pos"]].float().cpu())
        tok_type.append(c["type"][b_idx, t_idx].cpu())
        tp = papers[b_idx.cpu(), t_idx.cpu()]
        tok_paper.append(tp)
        tok_field.append(torch.as_tensor(gi.paper_field[tp.numpy()]))
        flagged += [float(i["audit"]["flagged"]) for i in chunk]
        ttype.append(batch["target_type"].cpu())
        preds.append(pred.float().cpu()); targets.append(tgt.float().cpu())

    tok_type, tok_paper, tok_field = torch.cat(tok_type), torch.cat(tok_paper), torch.cat(tok_field)
    rng = np.random.default_rng(seed)
    keep = torch.as_tensor(np.sort(rng.choice(len(tok_type), min(max_tokens, len(tok_type)), replace=False)))
    tok_type, tok_paper, tok_field = tok_type[keep], tok_paper[keep], tok_field[keep]
    flagged = torch.tensor(flagged)
    has_leak_var = 0 < flagged.sum() < len(flagged)
    fname = lambda f: str(gi.field_names[f])
    tname = lambda t: COMP_TYPES[t] if t < len(COMP_TYPES) else "hub"

    layer_out, neuron_rows = [], []
    for li in range(L + 1):
        name = f"encoder block {li + 1}" if li < L else "encoder output"
        X = torch.cat(acts[li])[keep]
        M = torch.cat(mask_acts[li])
        std = X.std(0)
        dead = std < TH["dead_rel_std"] * std.mean().clamp_min(1e-8)
        o_type = omega2(X, tok_type)
        Xc = _type_centre(X, tok_type)
        o_paper = omega2(Xc, tok_paper)
        if cfg.ku_mode == "paper":           # one KU per paper: "which paper" = "which KU", every
            o_paper = np.zeros_like(o_paper)  # varying neuron would look like identity -> not defined
        o_field = omega2(Xc, tok_field)
        tt_all = torch.cat(ttype)
        r_leak = (_corr(_type_centre(M, tt_all), _type_centre(flagged[:, None], tt_all)[:, 0])
                  if has_leak_var else torch.zeros(X.shape[1]))
        top_t = _top_group(X, tok_type, tname)
        top_f = _top_group(Xc, tok_field, fname)
        cats = []
        for j in range(X.shape[1]):
            if dead[j]:
                c = "dead"
            elif o_paper[j] >= TH["identity"]:
                c = "identity"
            elif o_type[j] >= TH["type"]:
                c = "type"
            elif o_field[j] >= TH["field"]:
                c = "field"
            elif abs(r_leak[j]) >= TH["leak"]:
                c = "leak"
            else:
                c = "content"
            cats.append(c)
            if rows:
                neuron_rows.append({"layer": name, "neuron": j, "category": c,
                                    "label": _label(c, top_t[j], top_f[j]),
                                    "std": round(float(std[j]), 4), "o_type": round(float(o_type[j]), 3),
                                    "o_paper": round(float(o_paper[j]), 3), "o_field": round(float(o_field[j]), 3),
                                    "r_leak": round(float(r_leak[j]), 3), "top_type": top_t[j], "top_field": top_f[j]})
        counts = {k: cats.count(k) for k in CATS}
        d = X.shape[1]
        layer_out.append({"name": name, "n": d, "counts": counts,
                          "mean_o_type": float(o_type.mean()), "mean_o_paper": float(o_paper.mean()),
                          "mean_o_field": float(o_field.mean()),
                          "share_paper_var": float(o_paper.mean()),
                          "verdict": _layer_verdict(name, counts, d, float(o_paper.mean()), float(o_type.mean()),
                                                    cfg.ku_mode == "paper")})

    # ---- predictor output: which dimensions learned the target? ----------------------
    P, T = torch.cat(preds), torch.cat(targets)
    tt = torch.cat(ttype)
    # WITHIN-type: the target type alone moves both pred and target, so a raw
    # correlation would credit "knows it is a claim" as learning the claim
    Pc, Tc = _type_centre(P, tt), _type_centre(T, tt)
    r_t = (Pc * Tc).sum(0) / (Pc.norm(dim=0) * Tc.norm(dim=0)).clamp_min(1e-12)
    o_tt = omega2(P, tt)
    r_l = _corr(Pc, _type_centre(flagged[:, None], tt)[:, 0]) if has_leak_var else torch.zeros(P.shape[1])
    learned = (r_t >= TH["learned"])
    pstd = P.std(0)
    pred_rows = []
    for j in range(P.shape[1]):
        if pstd[j] < TH["dead_rel_std"] * pstd.mean().clamp_min(1e-8):
            lab = "dead: same output for every query"
        elif r_t[j] >= TH["learned"]:
            lab = f"learned: follows the true target's content within its type (r = {r_t[j]:.2f})"
        elif o_tt[j] >= 0.5:
            lab = "only encodes WHICH KU type is asked for, not its content"
        elif abs(r_l[j]) >= TH["leak"]:
            lab = "moves with paraphrase leaks (copying from visible text)"
        else:
            lab = "not learned: unrelated to the target"
        if rows:
            pred_rows.append({"neuron": j, "r_target": round(float(r_t[j]), 3), "o_ktype": round(float(o_tt[j]), 3),
                              "r_leak": round(float(r_l[j]), 3), "std": round(float(pstd[j]), 4), "label": lab})
    n_learned = int(learned.sum())
    pred = {"n": int(P.shape[1]), "learned": n_learned, "mean_r_target": float(r_t.mean()),
            "type_only": int((o_tt >= 0.5).sum()), "leak": int((r_l.abs() >= TH["leak"]).sum()),
            "verdict": (f"{n_learned} of {P.shape[1]} predictor outputs follow the true target (r >= {TH['learned']}); "
                        f"mean r = {float(r_t.mean()):.2f}. "
                        + ("The predictor has learned a real part of the target." if n_learned > 0.2 * P.shape[1] else
                           "Most outputs do not track the target: the prediction carries little of the hidden KU's content.")
                        + (f" {int((o_tt >= 0.5).sum())} outputs only encode which KU type was asked for."
                           if int((o_tt >= 0.5).sum()) else ""))}
    model.train()
    return {"layers": layer_out, "predictor": pred, "neurons": neuron_rows, "predictor_neurons": pred_rows,
            "n_instances": len(insts), "n_tokens": int(len(keep)), "thresholds": TH}


def _layer_verdict(name, c, d, mean_paper, mean_type, paper_mode=False):
    pct = lambda k: 100 * c[k] / max(d, 1)
    parts = []
    if pct("dead") > 20:
        parts.append(f"{pct('dead'):.0f}% of neurons are dead")
    if pct("identity") > 10:
        parts.append(f"{pct('identity'):.0f}% mainly encode WHICH PAPER a KU comes from: this layer supports the "
                     f"identity shortcut")
    if pct("type") > 30:
        parts.append(f"{pct('type'):.0f}% mainly encode the KU type")
    if pct("leak") > 5:
        parts.append(f"{pct('leak'):.0f}% track paraphrase leaks at the masked token")
    parts.append(f"{pct('content'):.0f}% are distributed content features")
    if paper_mode:
        parts.append("paper identity is not measured (one KU per paper: the paper IS the KU)")
    else:
        parts.append(f"on average {100 * mean_paper:.0f}% of a neuron's within-type variance is paper identity")
    return f"{name}: " + "; ".join(parts) + "."
