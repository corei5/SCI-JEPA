"""
Evaluation for Paper A. Always reported as a set: the prior result is that
marginal diagnostics miss the failure.

  chi                 within-subgraph variance share of component states
                      (H-JEPA App. L, Def. 1 scale: collapsed 0.008-0.016,
                      healthy 0.12). Reported overall and within type.
  partition_probe     linear probe accuracy for subgraph identity, and the
                      between-subgraph variance share (G1: below 30%).
  retrieval           masked-component MRR / Hits@1 / Hits@10 against the
                      full same-type bank (standard) and K hard same-type
                      distractors from the same subfield / proxy-citation
                      neighbours / field, independent of N (balanced); shortcut
                      margin = standard - balanced with a bootstrap CI.
  relation_sensitivity  relative change of the prediction when one true role
                      edge is rewired to another subgraph.
  inverse_acc         accuracy of the inverse-relation head on hidden edges.
  detector            online collapse detector: constraint satisfaction on the
                      true target minus satisfaction against a same-type
                      impostor. ~0 means the prediction no longer depends on
                      the content (fires before variance diagnostics should).
  audit               nuisance-audit summary of the evaluation instances.
"""
import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.data.graph_index import COMP_TYPES, HUB
from sci_jepa.data.instances import (Sampler, build_view, collate, collate_views,
                                     minimal_pair, perturb_relation)
from sci_jepa.models.model import to_device


def effective_rank(X):
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X.double())
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def encode_papers(model, gi, papers, cfg, dev, which="target", bs=64):
    """Each paper encoded alone and unmasked (N=1). Returns (comp ids, states)."""
    enc = model.target if which == "target" else model.context
    rng = np.random.default_rng(0)
    ids, reps = [], []
    for s in range(0, len(papers), bs):
        views = []
        for p in papers[s:s + bs]:
            comps = gi.paper_comps[p]
            if len(comps) > cfg.max_comp_per_paper:
                comps = rng.choice(comps, cfg.max_comp_per_paper, replace=False)
            views.append(build_view(gi, [int(p)], [comps], cfg))
        v = to_device(collate_views(views), dev)
        h = enc(v)
        keep = (~v["pad"]) & (v["type"] != HUB)
        ids.append(v["comp"][keep].cpu()); reps.append(h[keep].float().cpu())
    return torch.cat(ids), torch.cat(reps)


def variance_share(H, groups):
    """Fraction of total variance that is WITHIN groups."""
    # one pass instead of a loop over groups (50k papers at 2M): within = total - sum_g n_g |mean_g|^2, in float64
    H = H.double()
    H = H - H.mean(0)
    tot = H.pow(2).sum()
    _, g = groups.unique(return_inverse=True)
    sums = torch.zeros(int(g.max()) + 1, H.shape[1], dtype=H.dtype, device=H.device).index_add_(0, g, H)
    n = torch.bincount(g, minlength=sums.shape[0]).to(H.dtype)
    within = tot - (sums.pow(2).sum(1) / n.clamp_min(1)).sum()
    return float(within.clamp_min(0) / tot.clamp_min(1e-12))


def type_centre(H, typ):
    """Remove each component type's mean: the aspect centroid is the shortcut,
    so subgraph-identity statistics are computed on what is left."""
    Hc = H.clone()
    for t in typ.unique():
        Hc[typ == t] -= H[typ == t].mean(0)
    return Hc


def chi_stats(ids, H, gi):
    """chi = within-subgraph variance share (H-JEPA Def. 1 scale).
      chi_raw          all components, type centroids included (dominated by type)
      chi              type-centred: the number to compare with H-JEPA / G1
      chi_per_type     within one type, papers with >= 2 components of it
                       (types with one per paper, e.g. method/result, have none)"""
    paper = torch.as_tensor(gi.comp_paper[ids.numpy()])
    typ = torch.as_tensor(gi.comp_type[ids.numpy()])
    per_type = {}
    for t in typ.unique():
        sel = typ == t
        p, c = paper[sel].unique(return_counts=True)
        multi = torch.isin(paper, p[c >= 2]) & sel
        if multi.sum() > 10:
            per_type[COMP_TYPES[int(t)]] = variance_share(H[multi], paper[multi])
    return {"chi_raw": variance_share(H, paper),
            "chi": variance_share(type_centre(H, typ), paper),
            "chi_per_type": per_type,
            "eff_rank": effective_rank(H[:8000]),
            "dim": int(H.shape[1])}


def partition_probe(ids, H, gi, n_papers, seed=0):
    """Linear probe for subgraph identity on type-centred states. Train/test are
    split WITHIN each type, so the probe cannot learn type -> paper."""
    from sklearn.linear_model import LogisticRegression
    paper = gi.comp_paper[ids.numpy()]
    typ = gi.comp_type[ids.numpy()]
    vals, counts = np.unique(paper, return_counts=True)
    elig = vals[counts >= 4]
    if len(elig) < 3:
        return {"probe_acc": float("nan")}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(elig, size=min(n_papers, len(elig)), replace=False)
    tr, te = [], []
    for p in chosen:
        for t in np.unique(typ[paper == p]):
            rows = rng.permutation(np.nonzero((paper == p) & (typ == t))[0])
            if len(rows) >= 2:
                tr += list(rows[:len(rows) // 2]); te += list(rows[len(rows) // 2:])
    if not tr or not te:
        return {"probe_acc": float("nan")}
    Hc = type_centre(H, torch.as_tensor(typ))
    X = F.normalize(Hc, dim=-1).numpy()
    clf = LogisticRegression(max_iter=2000).fit(X[tr], paper[tr])
    acc = float((clf.predict(X[te]) == paper[te]).mean())
    sel = np.isin(paper, chosen)
    between = 1 - variance_share(Hc[sel], torch.as_tensor(paper[sel]))
    return {"probe_acc": acc, "probe_chance": 1.0 / len(chosen), "probe_papers": int(len(chosen)),
            "probe_test_n": len(te), "between_subgraph_var_share": between}


def _ranks(pred, bank, gold_pos):
    s = F.normalize(pred, dim=-1) @ F.normalize(bank, dim=-1).T
    g = s.gather(1, gold_pos.view(-1, 1))
    return (s > g).sum(1).float() + 1


def _mrr(r):
    return {"MRR": float((1 / r).mean()), "Hits@1": float((r <= 1).float().mean()),
            "Hits@10": float((r <= 10).float().mean())}


def centred_cos(pred, keys, mu):
    """Cosine of a prediction to target vectors after removing mu, the mean target vector of the candidate bank.
    The JEPA latent space has a large direction every KU shares (raw cosines between ANY prediction and ANY KU are
    ~0.85-1.0, even before training), so the similarity rule is applied to the centred vectors."""
    return F.cosine_similarity((pred.float() - mu), (keys.float() - mu))


def similarity_acc(sims, types=None, thr=0.8, distractor=None, top1=None):
    """Accuracy under the similarity rule: a prediction is correct when its (centred, centred_cos) cosine to the
    true KU's target vector is >= thr. -> {threshold, acc, mean cosine, n, by_type} plus, with the best distractor's cosine and whether the
    true KU ranked first: false_accept (share of queries whose best WRONG candidate is also >= thr: how much the
    rule alone over-counts) and acc_top1 (true KU ranked first AND >= thr)."""
    s = np.asarray(sims, dtype=float)
    if not len(s):
        return {"threshold": thr, "acc": None, "mean": None, "n": 0, "by_type": {}}
    out = {"threshold": thr, "acc": float((s >= thr).mean()), "mean": float(s.mean()), "n": int(len(s)), "by_type": {}}
    if distractor is not None and len(distractor) == len(s):
        dd = np.asarray(distractor, dtype=float)
        out["false_accept"] = float(np.nanmean(dd >= thr)) if np.isfinite(dd).any() else None
        out["distractor_mean"] = float(np.nanmean(dd)) if np.isfinite(dd).any() else None
    if top1 is not None and len(top1) == len(s):
        out["acc_top1"] = float(((s >= thr) & np.asarray(top1, dtype=bool)).mean())
    if types is not None:
        t = np.asarray(types)
        out["by_type"] = {k: {"acc": float((s[t == k] >= thr).mean()), "mean": float(s[t == k].mean()),
                              "n": int((t == k).sum())} for k in sorted(set(types))}
    return out


def _chance(n):
    return float(sum(1.0 / k for k in range(1, n + 1)) / n)


def hard_pool(gi, p0, same, bank_paper, K, rng, mode="topic"):
    """K same-type candidates from papers related to p0 but never p0 itself. Independent
    of the sampled neighbourhood.
      topic     the K candidates whose papers' PROBLEM (context + question) is most similar
                to p0's: every distractor is on-topic, so topic matching cannot pick the answer
      subfield  same subfield, then citation / proxy neighbours, then same field (old protocol)"""
    if mode == "topic":
        cand = same[bank_paper[same] != p0]
        if not len(cand):
            return []
        V = gi.problem_vec()
        sim = V[bank_paper[cand]] @ V[p0]
        tie = rng.random(len(cand))                     # drawn in full: the rng stream stays as before
        if len(cand) > 4 * K:                           # top K by (-sim, tie) without sorting every candidate
            kth = np.partition(-sim, K - 1)[K - 1]
            sub = np.flatnonzero(-sim <= kth)           # every candidate tied with the K-th stays in
            order = sub[np.lexsort((tie[sub], -sim[sub]))]
        else:
            order = np.lexsort((tie, -sim))
        return [int(c) for c in cand[order[:K]]]
    tiers = [gi.subfield_papers[int(gi.paper_subfield[p0])],
             list(gi.paper_nbrs.get(p0, {}).keys()),
             gi.field_papers[int(gi.paper_field[p0])]]
    chosen, seen = [], set()
    for papers in tiers:
        cand = same[np.isin(bank_paper[same], [q for q in papers if q != p0])]
        cand = [int(c) for c in rng.permutation(cand) if int(c) not in seen]
        take = cand[:K - len(chosen)]
        chosen += take; seen.update(take)
        if len(chosen) >= K:
            return chosen
    rest = [int(c) for c in rng.permutation(same[bank_paper[same] != p0]) if int(c) not in seen]
    return chosen + rest[:K - len(chosen)]


def _breakdowns(records, inv_lab, inv_pred, max_records=1500, audit_tau=0.75):
    """Where the model learns and where it does not: per KU type, per field, per
    instance property, rank histograms, and inverse-head accuracy per relation."""
    from sci_jepa.data.ku_schema import ROLE_RELS
    out = {}
    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        df["rr_bal"] = 1 / df["rank_bal"]; df["rr_std"] = 1 / df["rank_std"]
        df["chance_bal"] = [_chance(k) for k in df["pool"]]
        agg = lambda g: {"n": int(len(g)), "MRR_bal": float(g.rr_bal.mean()), "chance_bal": float(g.chance_bal.mean()),
                         "Hits1_bal": float((g.rank_bal <= 1).mean()), "MRR_std": float(g.rr_std.mean()),
                         "top1_same_paper": float(g.top1_same_paper.mean())}
        out["per_type"] = {k: agg(g) for k, g in df.groupby("type")}
        top_fields = df.field.value_counts().index[:15]
        out["per_field"] = {k: agg(g) for k, g in df[df.field.isin(top_fields)].groupby("field")}
        out["by_flagged"] = {str(k): agg(g) for k, g in df.groupby("flagged")}
        out["by_singleton"] = {str(k): agg(g) for k, g in df.groupby("singleton")}
        t = audit_tau                                # relative to the encoder's leak threshold
        bins = [0, t - .20, t - .10, t - .05, t, 1.01]
        df["sim_bin"] = pd.cut(df.max_ctx_sim, bins).astype(str)
        out["by_ctx_sim"] = {k: agg(g) for k, g in df.groupby("sim_bin")}
        edges = [1, 2, 3, 6, 11, 21, 51, 101, 10 ** 9]
        out["rank_hist_bal"] = np.histogram(df.rank_bal, bins=edges)[0].tolist()
        out["rank_hist_edges"] = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"]
        out["instances"] = df.drop(columns=["sim_bin"]).sort_values("rank_bal").to_dict("records")[:max_records]
    if inv_lab:
        names = ROLE_RELS + ["no edge"]
        lab, pr = np.asarray(inv_lab), np.asarray(inv_pred)
        out["inverse_per_rel"] = {names[c]: {"n": int((lab == c).sum()), "acc": float((pr[lab == c] == c).mean())}
                                  for c in np.unique(lab)}
    return out


def _boot_diff(a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


@torch.no_grad()
def evaluate(model, gi, cfg, dev, shape_bank=None, split="test", seed=1234, ablate=None):
    """ablate="typed": every typed edge removed from the query graphs (structure reliance)."""
    model.eval()
    papers = np.asarray(gi.papers(split))
    bank_papers = papers if not cfg.eval_bank_papers else \
        np.random.default_rng(seed).choice(papers, min(cfg.eval_bank_papers, len(papers)), replace=False)
    ids, H_t = encode_papers(model, gi, bank_papers, cfg, dev, "target")
    _, H_c = encode_papers(model, gi, bank_papers, cfg, dev, "context")
    out = {"target_space": chi_stats(ids, H_t, gi), "context_space": chi_stats(ids, H_c, gi)}
    out["partition_probe"] = partition_probe(ids, H_c, gi, cfg.probe_papers)

    # ---- instances drawn from the bank papers only ------------------------------
    sampler = Sampler(gi, cfg, split, seed)
    sampler.ablate = ablate
    in_bank = np.zeros(gi.n_papers, bool); in_bank[bank_papers] = True
    sampler.anchors = sampler.anchors[in_bank[gi.comp_paper[sampler.anchors]]]
    pos_of = {int(c): k for k, c in enumerate(ids.tolist())}
    ctype = gi.comp_type[ids.numpy()]
    bank_paper = gi.comp_paper[ids.numpy()]
    rng = np.random.default_rng(seed)

    preds, golds, types, std_r, bal_r, sens, inv_hit, sat_t, sat_o = [], [], [], [], [], [], [], [], []
    sims, sim_types, dis, top1s = [], [], [], []      # similarity rule: cosine to the true KU / best distractor
    mu = H_t.float().mean(0, keepdim=True).to(dev)     # ... centred on the bank's mean vector (centred_cos)
    pool_sizes = []
    audits = []
    records, inv_lab, inv_pred = [], [], []           # per-query detail for the dashboard
    snip = getattr(gi, "comp_snippet", None)
    n_left = cfg.eval_instances
    while n_left > 0:
        insts = [i for i in sampler.sample(min(cfg.batch_size, n_left)) if i["target_comp"] in pos_of]
        n_left -= cfg.batch_size
        if not insts:
            continue
        batch = to_device(collate(insts), dev)
        pred, h = model.predict(batch)
        audits += batch["audit"]
        for b, inst in enumerate(insts):
            t = inst["target_type"]
            same = np.nonzero(ctype == t)[0]
            gold = pos_of[inst["target_comp"]]
            # standard: full same-type bank
            bank = H_t[same].to(dev)
            gp = torch.tensor([int(np.nonzero(same == gold)[0][0])], device=dev)
            std_r.append(_ranks(pred[b:b + 1], bank, gp).item())
            top1 = int(same[int((F.normalize(pred[b:b + 1], dim=-1) @ F.normalize(bank, dim=-1).T).argmax())])
            # balanced: exactly K same-type distractors, chosen WITHOUT the instance's
            # neighbourhood (so pool difficulty does not change with N): same subfield,
            # then proxy / citation neighbours, then same field, then anywhere
            pool = hard_pool(gi, int(inst["papers"][0]), same, bank_paper, cfg.balanced_k, rng, mode=cfg.pool_mode)
            pool_sizes.append(len(pool))
            pool = [gold] + pool
            bal_r.append(_ranks(pred[b:b + 1], H_t[pool].to(dev),
                                torch.tensor([0], device=dev)).item())
            cp = centred_cos(pred[b:b + 1], H_t[pool].to(dev), mu)
            sims.append(float(cp[0])); sim_types.append(COMP_TYPES[t]); top1s.append(bal_r[-1] == 1)
            dis.append(float(cp[1:].max()) if len(pool) > 1 else float("nan"))
            p0 = int(inst["papers"][0]); a = inst["audit"]
            records.append({
                "type": COMP_TYPES[t], "rank_std": std_r[-1], "rank_bal": bal_r[-1], "sim": round(sims[-1], 4),
                "pool": len(pool), "bank": int(len(same)),
                "field": str(gi.field_names[gi.paper_field[p0]]), "year": int(gi.paper_year[p0]),
                "paper": str(gi.paper_ids[p0]), "n_papers": int(a["n_papers"]),
                "max_ctx_sim": round(float(a["max_ctx_sim"]), 4), "flagged": bool(a["flagged"]),
                "singleton": bool(a["singleton_type"]),
                "top1_same_paper": bool(gi.comp_paper[int(ids[top1])] == p0),
                "target": str(snip[int(inst["target_comp"])]) if snip is not None else "",
                "top1": str(snip[int(ids[top1])]) if snip is not None else ""})
        # relation sensitivity
        pert = [perturb_relation(i, rng) for i in insts]
        keep = [k for k, p in enumerate(pert) if p is not None]
        if keep:
            pb = to_device(collate([pert[k] for k in keep]), dev)
            p2, _ = model.predict(pb)
            p1 = pred[keep]
            sens += ((p2 - p1).norm(dim=-1) / p1.norm(dim=-1).clamp_min(1e-8)).tolist()
        # inverse head
        b_, i_, j_, lab = batch["probe"]
        if len(b_):
            ip = model.inverse(h[b_, i_], h[b_, j_]).argmax(-1)
            inv_hit += (ip == lab).float().tolist()
            inv_lab += lab.tolist(); inv_pred += ip.tolist()
        # online detector
        if shape_bank is not None:
            comps, tt = batch["target_comp"], batch["target_type"]
            impostor = comps.clone()
            for k in range(len(comps)):
                same_t = sampler.anchors[gi.comp_type[sampler.anchors] == int(tt[k])]
                impostor[k] = int(rng.choice(same_t))
            sat_t += shape_bank.satisfaction(pred, comps, tt).tolist()
            sat_o += shape_bank.satisfaction(pred, impostor, tt).tolist()

    std_r, bal_r = torch.tensor(std_r), torch.tensor(bal_r)
    out["retrieval_standard"] = _mrr(std_r)
    out["retrieval_balanced"] = {**_mrr(bal_r), "mean_pool": float(np.mean(pool_sizes)) + 1 if pool_sizes else 0.0,
                                 "chance_MRR": float(np.mean([_chance(k + 1) for k in pool_sizes])) if pool_sizes else 0.0}
    out["similarity"] = similarity_acc(sims, sim_types, cfg.sim_threshold, dis, top1s)
    out["retrieval_balanced"]["acc_sim"] = out["similarity"]["acc"]
    out["shortcut_margin_MRR"] = _boot_diff((1 / std_r).numpy(), (1 / bal_r).numpy())
    n_bank = {COMP_TYPES[t]: int((ctype == t).sum()) for t in np.unique(ctype)}
    out["bank_sizes"] = n_bank
    out["chance_MRR_standard"] = float(np.mean([(np.log(n) + 0.5772) / n for n in n_bank.values() if n > 1]))
    out["relation_sensitivity"] = float(np.mean(sens)) if sens else float("nan")
    out["inverse_acc"] = float(np.mean(inv_hit)) if inv_hit else float("nan")
    if sat_t:
        out["detector"] = {"sat_true": float(np.mean(sat_t)), "sat_impostor": float(np.mean(sat_o)),
                           "discrimination": float(np.mean(sat_t) - np.mean(sat_o))}
    out["audit"] = {"n": len(audits),
                    "flagged_frac": float(np.mean([a["flagged"] for a in audits])) if audits else 0.0,
                    "singleton_type_frac": float(np.mean([a["singleton_type"] for a in audits])) if audits else 0.0,
                    "mean_max_ctx_sim": float(np.mean([a["max_ctx_sim"] for a in audits])) if audits else 0.0,
                    "mean_papers": float(np.mean([a["n_papers"] for a in audits])) if audits else 0.0}
    out["n_eval"] = int(len(std_r))
    out.update(_breakdowns(records, inv_lab, inv_pred, audit_tau=cfg.audit_tau))
    model.train()
    return out
