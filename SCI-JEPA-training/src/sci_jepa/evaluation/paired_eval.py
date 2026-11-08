"""
Paired benchmark, protocol v2 (sci-jepa paired-eval): the fixes from the evaluation audit of 25 Oct 2026.
Every saved model and every comparison system is re-scored, without retraining, on ONE frozen question set.

  1 shared questions     the same anchors, neighbourhoods, 100-candidate pools, counterfactual edits, consistency
                         pairs and intervention donors for every system (the stored results used different seeds)
  2 mid-rank ties        rank = 1 + #higher + 0.5 * #equal (the expected rank under random tie-breaking); a tie with
                         a counterfactual edit counts half
  3 one candidate code   JEPA: every candidate, the truth and its edits included, encoded by the target encoder
                         inside its own paper, so no question is dropped and the truth is not encoded differently
  4 consistency          the same fixed test pairs; JEPA: inverse-relation head on the hidden claim-evidence edge;
                         embedding systems: linear probe on the system's OWN vectors (supervised: its network's
                         output) trained on train pairs; LLMs: SUMMED log-probability of " supports" / " contradicts"
                         (calibrated by a content-free prompt), threshold tuned on train pairs. Balanced accuracy + AUC
  5 intervention         swap the paper's procedure for a topic donor's; rank of the DONOR's result among 100 results
                         topic-matched to the donor, after vs before the swap (gain in reciprocal rank). Reported, not
                         in the headline: the old "toward donor" share was ~100 % for every system by construction
  6 statistics           headline = average of forward, design, abduction (balanced MRR) and consistency (balanced
                         accuracy), in %; per question, seeds are averaged first; 95 % intervals from a paired
                         bootstrap over the shared questions (consistency pairs resampled by paper); the best JEPA is
                         chosen on VALIDATION; single-seed systems are marked
Writes <exp>/paired_v2.json (with per-question records) and dashboard/paired_v2.json (summary).
    sci-jepa paired-eval <results folder> [--n 300] [--n-llm 150] [--pairs 3000] [--only a,b] [--no-supervised]
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np

RANK_TASKS = ("forward", "inverse", "abduction")
B_BOOT = 2000


# ---------------------------------------------------------------- metrics
def mid_rank(s_truth, s_wrong):
    s_wrong = np.asarray(s_wrong, float)
    return 1.0 + float((s_wrong > s_truth).sum()) + 0.5 * float((s_wrong == s_truth).sum())


def pair_win(s_truth, s_var):
    """1 if the truth beats the edit, 0.5 on a tie, 0 if the edit wins."""
    return [1.0 if s_truth > v else (0.5 if s_truth == v else 0.0) for v in np.asarray(s_var, float)]


def chance_rr(n):
    return float(sum(1.0 / k for k in range(1, n + 1)) / n)


def auc(y, s):
    """Area under the ROC curve (Mann-Whitney, ties half); None for one class."""
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv)); i = 0
    while i < len(allv):                                 # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    r = np.empty(len(allv)); r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bal_acc(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rec = [float((yhat[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return float(np.mean(rec)) if rec else None


def best_threshold(y, s):
    """Threshold on s (predict 1 when s > t) that maximises balanced accuracy on (y, s)."""
    s = np.asarray(s, float); y = np.asarray(y)
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 199)))
    best = max(cand, key=lambda t: bal_acc(y, (s > t).astype(int)) or 0)
    return float(best)


# ---------------------------------------------------------------- the frozen question set
def build_questions(gi, cfg, n=300, n_pairs=3000, n_train_pairs=2000, n_int=150, seed=4321, k_pool=100):
    from sci_jepa.data.graph_index import COMP_TYPES
    from sci_jepa.evaluation.baselines import _pairs
    from sci_jepa.evaluation.diagnostics import hard_pool
    from sci_jepa.evaluation.tasks import TASKS, _dom, _sampler
    test = np.asarray(gi.papers("test"))
    bank = np.nonzero(np.isin(gi.comp_paper, test))[0]
    bank_paper = gi.comp_paper[bank]
    Q = []
    for task in RANK_TASKS:
        given, targets = TASKS[task]["given"], TASKS[task]["targets"]
        for tg in targets:
            s = _sampler(gi, cfg, "test", seed, tg, given)
            same = np.nonzero(gi.comp_type[bank] == COMP_TYPES.index(tg))[0]
            for a in s.draw(max(n // len(targets), 16)):
                a = int(a); p0 = int(gi.comp_paper[a]); i = len(Q)
                rng = np.random.default_rng(seed + 7919 * i)
                Q.append({"i": i, "task": task, "target": tg, "anchor": a, "papers": [int(p) for p in s.neighbourhood(p0)],
                          "pool": [int(bank[j]) for j in hard_pool(gi, p0, same, bank_paper, k_pool, rng)],
                          "var": [int(r) for r in gi.mp_of.get(a, [])] if cfg.minimal_pairs else [],
                          "paper": p0, "year": int(gi.paper_year[p0]), "dom": int(_dom(gi, p0))})
    rng = np.random.default_rng(seed)
    te, tr = _pairs(gi, test), _pairs(gi, gi.papers("train"))
    pick = rng.choice(len(te), size=min(n_pairs, len(te)), replace=False) if te else []
    s_ctx = _sampler(gi, cfg, "test", seed + 3, "context", None)
    P = []
    for k in sorted(pick):
        a, b, y = te[int(k)]; p = int(gi.comp_paper[a])
        P.append({"a": int(a), "b": int(b), "y": int(y), "paper": p, "papers": [int(q) for q in s_ctx.neighbourhood(p)],
                  "year": int(gi.paper_year[p]), "dom": int(_dom(gi, p))})
    PT = [{"a": int(tr[int(k)][0]), "b": int(tr[int(k)][1]), "y": int(tr[int(k)][2])}
          for k in rng.choice(len(tr), size=min(n_train_pairs, len(tr)), replace=False)] if tr else []
    # intervention: forward questions whose paper has a procedure and a precomputed condition donor in the test split
    t_res, t_proc = COMP_TYPES.index("result"), COMP_TYPES.index("procedure")
    same_res = np.nonzero(gi.comp_type[bank] == t_res)[0]
    test_set = set(test.tolist())
    I = []
    for q in [q for q in Q if q["task"] == "forward"]:
        if len(I) >= n_int:
            break
        p0 = q["paper"]
        if not any(gi.comp_type[c] == t_proc for c in gi.paper_comps[p0]):
            continue
        rng_d = np.random.default_rng(seed + 31 * q["i"])
        sub = []                                         # same subfield, then same field, then any test paper
        for grp in (gi.subfield_papers[int(gi.paper_subfield[p0])], gi.field_papers[int(gi.paper_field[p0])], test):
            sub += [int(x) for x in rng_d.permutation(grp) if int(x) != p0 and int(x) in test_set and int(x) not in sub]
            if len(sub) >= 50:
                break
        sub = sub[:50]
        cands = [(int(row), int(d)) for row, d in gi.cf_of.get((p0, "condition"), ())] + [(-1, d) for d in sub]
        for row, d in cands:                             # precomputed condition donors first (paper_masked can use them)
            dres = [int(c) for c in gi.paper_comps[d] if gi.comp_type[c] == t_res]
            dproc = [int(c) for c in gi.paper_comps[d] if gi.comp_type[c] == t_proc]
            if d in test_set and d not in q["papers"] and dres and dproc:
                rng = np.random.default_rng(seed + 104729 + len(I))
                pool = [int(bank[j]) for j in hard_pool(gi, d, same_res, bank_paper, k_pool, rng)]
                own = [int(c) for c in gi.paper_comps[p0] if gi.comp_type[c] == t_res]
                pool = [c for c in pool if c not in own] + own[:1]
                I.append({"q": q["i"], "donor": d, "cf_row": int(row), "donor_result": dres[0], "donor_proc": dproc,
                          "pool": pool})
                break
    return {"questions": Q, "pairs": P, "train_pairs": PT, "interventions": I}


# ---------------------------------------------------------------- scorers
class JepaPaired:
    """A saved JEPA model. Ranking: predictor output vs target-encoder keys of every candidate inside its own paper.
    Consistency: inverse-relation head on the claim-evidence edge, hidden, in the paper's context view.
    Intervention: the paper's procedure tokens replaced by the donor's (paper_masked: the donor-condition text)."""

    def __init__(self, model_dir, gi, dev):
        from sci_jepa.evaluation.science_safety import JepaScorer
        self.js = JepaScorer(model_dir, gi, dev)
        self.cfg, self.gi, self.dev, self.model = self.js.cfg, gi, self.js.dev, self.js.model

    def _pred(self, qs, given, swap=None):
        import torch.nn.functional as F
        from sci_jepa.data.graph_index import COMP_TYPES
        from sci_jepa.data.instances import build_instance, collate
        from sci_jepa.evaluation.tasks import CTX_TYPE
        from sci_jepa.models.model import to_device
        gi, cfg = self.gi, self.cfg
        insts = [build_instance(gi, q["anchor"], q["papers"], cfg, np.random.default_rng(q["i"]), given=set(given),
                                train=False) for q in qs]
        if swap is not None:
            t_proc = COMP_TYPES.index("procedure")
            for inst, it in zip(insts, swap):
                c = {**inst["ctx"], "x": inst["ctx"]["x"].copy()}
                if cfg.ku_mode == "paper_masked":
                    k = np.nonzero((c["slot"] == 0) & (c["type"] == CTX_TYPE))[0]
                    c["x"][k] = gi.cf_x[it["cf_row"]].astype(np.float32)
                else:
                    k = np.nonzero((c["slot"] == 0) & (c["type"] == t_proc))[0]
                    for j, kk in enumerate(k):
                        c["x"][kk] = gi.comp_x[it["donor_proc"][j % len(it["donor_proc"])]].astype(np.float32)
                inst["ctx"] = c
        pred, _ = self.model.predict(to_device(collate(insts), self.dev))
        return F.normalize(pred.float(), dim=-1)

    def rank(self, Q, log=print):
        import torch
        from sci_jepa.evaluation.tasks import TASKS
        out = []
        with torch.no_grad():
            for task in RANK_TASKS:
                qs = [q for q in Q if q["task"] == task]
                for s in range(0, len(qs), 32):
                    part = qs[s:s + 32]
                    pred = self._pred(part, TASKS[task]["given"])
                    for b, q in enumerate(part):
                        kt, kp = self.js.keys([q["anchor"]])[0], self.js.keys(q["pool"])
                        st = float(pred[b] @ kt)
                        sv = []
                        if q["var"]:
                            kv = self.js._views([q["anchor"]] * len(q["var"]),
                                                self.gi.mp_x[np.asarray(q["var"])].astype(np.float32))
                            sv = (kv @ pred[b]).cpu().numpy()
                        out.append(_rank_rec(q, st, (kp @ pred[b]).cpu().numpy(), sv))
        return out

    def consistency(self, P, PT=None):
        import torch
        from sci_jepa.data.graph_index import REL_NONE, REL_SAME_PAPER
        from sci_jepa.data.instances import build_view, collate_views
        from sci_jepa.evaluation.tasks import CON, SUP
        from sci_jepa.models.model import to_device
        gi, cfg = self.gi, self.cfg
        cap = cfg.max_comp_per_paper
        out = []
        with torch.no_grad():
            for s in range(0, len(P), 64):
                part, views, pos = P[s:s + 64], [], []
                for p in part:
                    comps = []
                    for j, q in enumerate(p["papers"]):
                        cs = [int(c) for c in gi.paper_comps[q]]
                        if j == 0:
                            cs = [p["a"], p["b"]] + [c for c in cs if c not in (p["a"], p["b"])]
                        comps.append(np.asarray(cs[:cap]))
                    v = build_view(gi, p["papers"], comps, cfg)
                    ia, ib = int(np.nonzero(v["comp"] == p["a"])[0][0]), int(np.nonzero(v["comp"] == p["b"])[0][0])
                    v = {**v, "rel": v["rel"].copy()}
                    v["rel"][ia, ib] = v["rel"][ib, ia] = REL_SAME_PAPER if v["slot"][ia] == v["slot"][ib] else REL_NONE
                    views.append(v); pos.append((ia, ib))
                h = self.model.context(to_device(collate_views(views), self.dev))
                i_ = torch.tensor([a for a, _ in pos], device=self.dev); j_ = torch.tensor([b for _, b in pos], device=self.dev)
                b_ = torch.arange(len(views), device=self.dev)
                lg = self.model.inverse(h[b_, i_], h[b_, j_])[:, [SUP, CON]].float()
                pc = torch.softmax(lg, -1)[:, 1].cpu().numpy()
                out += [{"score": float(x), "yhat": int(x > 0.5)} for x in pc]
        return out

    def intervention(self, Q, I):
        import torch
        from sci_jepa.evaluation.tasks import TASKS
        q_of = {q["i"]: q for q in Q}
        todo = [it for it in I if it["cf_row"] >= 0] if self.cfg.ku_mode == "paper_masked" else I
        done = {}                                        # paper_masked: only donors with a donor-condition text
        with torch.no_grad():
            for s in range(0, len(todo), 32):
                part = todo[s:s + 32]
                qs = [q_of[it["q"]] for it in part]
                p1 = self._pred(qs, TASKS["forward"]["given"])
                p2 = self._pred(qs, TASKS["forward"]["given"], swap=part)
                for b, it in enumerate(part):
                    kd, kp = self.js.keys([it["donor_result"]])[0], self.js.keys(it["pool"])
                    done[it["q"]] = _int_rec(it, float(p1[b] @ kd), (kp @ p1[b]).cpu().numpy(),
                                             float(p2[b] @ kd), (kp @ p2[b]).cpu().numpy())
        return [done.get(it["q"]) for it in I]


class BaselinePaired:
    """A comparison system from baselines.py on the same questions (query from the paper's visible KUs)."""

    def __init__(self, cfg, gi, seed=0):
        from sci_jepa.evaluation import baselines as B
        self.B, self.cfg, self.gi = B, cfg, gi
        self.sys = B.make_system(cfg, gi, "test", seed)
        self.name = cfg.baseline

    def _vis(self, q, base_cfg, given):
        from sci_jepa.data.instances import build_instance
        inst = build_instance(self.gi, q["anchor"], q["papers"], base_cfg, np.random.default_rng(q["i"]),
                              given=set(given), train=False)
        return [int(c) for c in self.B._visible(inst)]

    def rank(self, Q, base_cfg, log=print):
        from sci_jepa.evaluation.tasks import TASKS
        out, t0 = [], time.time()
        for j, q in enumerate(Q):
            vis = self._vis(q, base_cfg, TASKS[q["task"]]["given"])
            if not vis:
                out.append(None); continue
            qv = self.sys.query(vis, int(self.gi.comp_type[q["anchor"]]))
            s = np.asarray(self.sys.score(qv, [self.sys.pos_of[c] for c in [q["anchor"]] + q["pool"]]), float)
            sv = np.asarray(self.sys.score_mp(qv, q["var"]), float) if q["var"] else []
            out.append(_rank_rec(q, float(s[0]), s[1:], sv))
            if (j + 1) % 100 == 0:
                log(f"[paired] {self.name}: ranking {j + 1}/{len(Q)} ({time.time() - t0:.0f}s)")
        return out

    def _feats(self, ids):
        """The system's own vector of each KU: probe features (supervised: its network's output for the KU)."""
        sys = self.sys
        if self.name == "supervised":
            import torch
            x = self.B._unit(self.gi.comp_x[np.asarray(ids)].astype(np.float32) - sys.mu)
            t = torch.as_tensor(self.gi.comp_type[np.asarray(ids)], dtype=torch.long, device=sys.dev)
            with torch.no_grad():
                return sys.net(torch.as_tensor(x, device=sys.dev), t).float().cpu().numpy()
        return sys.probe_feats(ids)

    def consistency(self, P, PT, log=print):
        if self.name == "llm":
            return self._llm_consistency(P, PT, log)
        from sklearn.linear_model import LogisticRegression
        tr = self.B._pairs(self.gi, self.gi.papers("train"))
        def feats(A, Bv):
            c, e = self.B._unit(self._feats(A)), self.B._unit(self._feats(Bv))
            return np.concatenate([c, e, c * e, np.abs(c - e)], 1)
        X = np.concatenate([feats([a for a, _, _ in tr[s:s + 50000]], [b for _, b, _ in tr[s:s + 50000]])
                            for s in range(0, len(tr), 50000)])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X, np.asarray([l for *_, l in tr]))
        pr = clf.predict_proba(feats([p["a"] for p in P], [p["b"] for p in P]))[:, 1]
        return [{"score": float(x), "yhat": int(x > 0.5)} for x in pr]

    def _llm_consistency(self, P, PT, log=print):
        """Summed log-probability of each answer (length-fair), minus the same after a content-free prompt; the
        decision threshold on [contradicts - supports] is tuned on train pairs."""
        sys = self.sys
        n = max(32, min(300, sys.ctx_tokens // 3))
        fmt = lambda c, e: (f"Claim: {sys._cut(c, n)}\nEvidence: {sys._cut(e, n)}\n"
                            f"Does the evidence support or contradict the claim? Answer:")
        ans = ["supports", "contradicts"]
        ntok = np.asarray([len(sys.tok(" " + a, add_special_tokens=False)["input_ids"]) for a in ans], float)
        null = sys._lp([fmt("N/A", "N/A")] * 2, ans) * ntok
        def score(pairs, label):
            out, t0 = [], time.time()
            for k, p in enumerate(pairs):
                lp = sys._lp_shared(fmt(sys.texts[p["a"]], sys.texts[p["b"]]), ans) * ntok - null
                out.append(float(lp[1] - lp[0]))
                if (k + 1) % 500 == 0:
                    log(f"[paired] llm {label}: {k + 1}/{len(pairs)} ({time.time() - t0:.0f}s)")
            return np.asarray(out)
        s_tr = score(PT, "train pairs (threshold)")
        thr = best_threshold([p["y"] for p in PT], s_tr) if len(PT) else 0.0
        s = score(P, "test pairs")
        self.threshold = thr
        return [{"score": float(x), "yhat": int(x > thr)} for x in s]

    def intervention(self, Q, I, base_cfg):
        from sci_jepa.data.graph_index import COMP_TYPES
        from sci_jepa.evaluation.tasks import TASKS
        q_of = {q["i"]: q for q in Q}
        t_proc, t_res = COMP_TYPES.index("procedure"), COMP_TYPES.index("result")
        out = []
        for it in I:
            q = q_of[it["q"]]
            vis = self._vis(q, base_cfg, TASKS["forward"]["given"])
            if not vis:
                out.append(None); continue
            vis2 = [c for c in vis if self.gi.comp_type[c] != t_proc] + list(it["donor_proc"])
            pos = [self.sys.pos_of[c] for c in [it["donor_result"]] + it["pool"]]
            q1, q2 = self.sys.query(vis, t_res), self.sys.query(vis2, t_res)
            s1, s2 = np.asarray(self.sys.score(q1, pos), float), np.asarray(self.sys.score(q2, pos), float)
            out.append(_int_rec(it, float(s1[0]), s1[1:], float(s2[0]), s2[1:]))
        return out

    def close(self):
        if hasattr(self.sys, "close"):
            self.sys.close()


def _rank_rec(q, s_truth, s_pool, s_var):
    s_var = np.asarray(s_var, float)
    wrong = np.concatenate([np.asarray(s_pool, float), s_var])
    w = pair_win(s_truth, s_var)
    return {"i": q["i"], "rr": 1.0 / mid_rank(s_truth, wrong), "chance": chance_rr(1 + len(wrong)),
            "mp": float(np.mean(w)) if w else None, "mp_all": float(all(x == 1.0 for x in w)) if w else None}


def _int_rec(it, s1d, s1p, s2d, s2p):
    r1, r2 = mid_rank(s1d, s1p), mid_rank(s2d, s2p)
    return {"q": it["q"], "rr_before": 1.0 / r1, "rr_after": 1.0 / r2,
            "toward": 1.0 if r2 < r1 else (0.5 if r2 == r1 else 0.0)}


# ---------------------------------------------------------------- aggregation with paired bootstrap
def _per_question(seed_recs, key, n):
    """Mean over seeds for each question index (None where no seed scored it)."""
    M = np.full((len(seed_recs), n), np.nan)
    for s, recs in enumerate(seed_recs):
        for j, r in enumerate(recs):
            if r is not None and r.get(key) is not None:
                M[s, j] = r[key]
    with np.errstate(all="ignore"):
        return np.nanmean(M, 0)


def summarise(data, systems, val_index=None):
    """systems: {tag: {"seeds": [...], "rank": [[rec...] per seed], "cons": [[...]], "int": [[...]]}}.
    -> per-system metrics with 95 % paired-bootstrap intervals and the H1-style comparison."""
    Q, P, I = data["questions"], data["pairs"], data["interventions"]
    task_idx = {t: np.asarray([j for j, q in enumerate(Q) if q["task"] == t]) for t in RANK_TASKS}
    y = np.asarray([p["y"] for p in P])
    pair_paper = np.asarray([p["paper"] for p in P])
    papers_u = np.unique(pair_paper)
    by_paper = [np.nonzero(pair_paper == u)[0] for u in papers_u]
    rng = np.random.default_rng(0)
    draws = []                                          # shared draws: the comparison between systems is paired
    for _ in range(B_BOOT):
        d = {t: rng.choice(ix, size=len(ix), replace=True) for t, ix in task_idx.items() if len(ix)}
        pk = rng.integers(0, len(by_paper), len(by_paper)) if by_paper else []
        d["pairs"] = np.concatenate([by_paper[k] for k in pk]) if len(pk) else np.zeros(0, int)
        draws.append(d)
    out = {}
    for tag, S in systems.items():
        rr = _per_question(S["rank"], "rr", len(Q)); mp = _per_question(S["rank"], "mp", len(Q))
        mpa = _per_question(S["rank"], "mp_all", len(Q))
        cons = S.get("cons") or []
        yhat = None
        if cons and any(c for c in cons):
            sc = np.nanmean(np.asarray([[np.nan if r is None else r["score"] for r in c] for c in cons], float), 0)
            yh = np.nanmean(np.asarray([[np.nan if r is None else r["yhat"] for r in c] for c in cons], float), 0)
            yhat = (yh >= 0.5).astype(int)
        def stat(d=None):
            m = {}
            for t, ix in task_idx.items():
                ix = ix if d is None else d[t]
                v = rr[ix]; m[t] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            if yhat is not None:
                ix = np.arange(len(P)) if d is None else d["pairs"]
                m["consistency"] = bal_acc(y[ix], yhat[ix]) if len(ix) else np.nan
            else:
                m["consistency"] = np.nan
            parts = [m[t] for t in RANK_TASKS] + [m["consistency"]]
            m["average"] = float(np.mean(parts)) if all(np.isfinite(parts)) else np.nan
            ixr = np.concatenate(list(task_idx.values())) if d is None else np.concatenate([d[t] for t in RANK_TASKS])
            v = mp[ixr]; m["robustness"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            return m
        point = stat()
        boots = [stat(d) for d in draws]
        ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))]
              for k in point if np.isfinite(point[k])}
        res = {"seeds": S["seeds"], "n_seeds": len(S["seeds"]), "point": point, "ci": ci, "family": S.get("family"),
               "config": S.get("config"), "_boots": boots,
               "robust_all": float(np.nanmean(mpa)) if np.isfinite(mpa).any() else None}
        if yhat is not None:
            sc_ok = np.isfinite(sc)
            res["consistency_auc"] = auc(y[sc_ok], sc[sc_ok])
            res["recall_supports"] = float((yhat[y == 0] == 0).mean()) if (y == 0).any() else None
            res["recall_contradicts"] = float((yhat[y == 1] == 1).mean()) if (y == 1).any() else None
        ints = S.get("int") or []
        if ints and any(c for c in ints):
            rb = _per_question(ints, "rr_before", len(I)); ra = _per_question(ints, "rr_after", len(I))
            tw = _per_question(ints, "toward", len(I))
            ok = np.isfinite(rb) & np.isfinite(ra)
            if ok.any():
                g = ra[ok] - rb[ok]
                gb = [float(np.mean(g[rng.integers(0, len(g), len(g))])) for _ in range(B_BOOT)]
                res["intervention"] = {"rr_before": float(rb[ok].mean()), "rr_after": float(ra[ok].mean()),
                                       "gain": float(g.mean()), "gain_ci": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
                                       "toward": float(np.nanmean(tw[ok])), "n": int(ok.sum())}
        if val_index and tag in val_index:
            res["val_index"] = val_index[tag]
        out[tag] = res
    # comparison: best JEPA (chosen on validation) vs the strongest comparison system (on test, the hardest opponent)
    jepa = [t for t, r in out.items() if r["family"] != "comparison" and np.isfinite(r["point"]["average"])]
    comp = [t for t, r in out.items() if r["family"] == "comparison" and np.isfinite(r["point"]["average"])]
    cmp = {}
    if jepa and comp:
        bj = max(jepa, key=lambda t: (out[t].get("val_index") is not None, out[t].get("val_index") or -1,
                                      out[t]["point"]["average"]))
        bc = max(comp, key=lambda t: out[t]["point"]["average"])
        for key in ("average", "forward", "inverse", "abduction", "consistency", "robustness"):
            diff = [a[key] - b[key] for a, b in zip(out[bj]["_boots"], out[bc]["_boots"])
                    if np.isfinite(a[key]) and np.isfinite(b[key])]
            if not diff:
                continue
            lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            cmp[key] = {"diff": out[bj]["point"][key] - out[bc]["point"][key], "ci": [lo, hi],
                        "verdict": "JEPA better" if lo > 0 else ("JEPA worse" if hi < 0 else "inconclusive")}
        cmp = {"best_jepa": bj, "selected_on": "validation reasoning index" if out[bj].get("val_index") is not None else "test",
               "strongest_comparison": bc, "metrics": cmp}
    for r in out.values():
        r.pop("_boots", None)
    return out, cmp


# ---------------------------------------------------------------- driver
def _val_index(m):
    """Best validation reasoning index over the run's epochs (the epoch whose weights are saved as best)."""
    best = None
    for e in m.get("epoch_results") or []:
        v = ((e.get("tasks") or {}).get("macro") or {}).get("index")
        if v is not None and (best is None or v > best):
            best = v
    return best


def run(out_dir, exp="BENCH", n=300, n_llm=150, n_pairs=3000, n_int=150, only=None, supervised=True, device="auto",
        log=print):
    import torch
    from sci_jepa.config import PaperAConfig
    from sci_jepa.data.graph_index import get_index
    exp_dir = os.path.join(out_dir, exp)
    path = os.path.join(exp_dir, "paired_v2.json")
    rep = json.load(open(path)) if os.path.exists(path) else {}
    rep.setdefault("runs", {})
    base = PaperAConfig()
    over = lambda c: base.override(**{k: (tuple(v) if isinstance(getattr(base, k), tuple) and isinstance(v, list) else v)
                                      for k, v in c.items() if hasattr(base, k) and k not in ("embedder", "device")}
                                   ).override(device=device)
    files = sorted(glob.glob(os.path.join(exp_dir, "*_seed*.json")))
    runs = []
    for f in files:
        m = json.load(open(f))
        tag = re.sub(r"_seed\d+\.json$", "", os.path.basename(f)); seed = int(re.search(r"_seed(\d+)\.json$", f).group(1))
        if only and tag not in only:
            continue
        b = (m.get("config") or {}).get("baseline") or ""
        if b and seed != 0:                              # comparison systems do not depend on the seed any more
            if not (b == "supervised" and supervised):
                continue
        if b == "supervised" and (not supervised or seed != 0):
            continue                                     # retrains a regressor (hours): seed 0 only
        if not b and not m.get("model_dir"):
            continue
        runs.append((tag, seed, m))
    ref = next((m for _, _, m in runs if m.get("model_dir")), runs[0][2])
    cfg = over(ref["config"])
    gi = get_index(cfg); gi.texts()
    if cfg.minimal_pairs and cfg.mp_filter:
        from sci_jepa.evaluation.validity import apply_filter
        apply_filter(gi, cfg)
    if not rep.get("data"):
        t0 = time.time()
        rep["data"] = build_questions(gi, cfg, n, n_pairs, 2000, n_int)
        D = rep["data"]
        log(f"[paired] questions: {len(D['questions'])} ranking, {len(D['pairs'])} consistency pairs, "
            f"{len(D['interventions'])} interventions ({time.time() - t0:.0f}s)")
        json.dump(rep, open(path, "w"), default=str)
    D = rep["data"]
    for tag, seed, m in sorted(runs, key=lambda r: ((r[2].get("config") or {}).get("baseline") == "llm",
                                                    (r[2].get("config") or {}).get("baseline") == "supervised")):
        key = f"{tag}_seed{seed}"
        if key in rep["runs"]:
            continue
        t1 = time.time()
        try:
            if m.get("model_dir"):
                sc = JepaPaired(m["model_dir"], gi, cfg.device if cfg.device != "auto" else device)
                r = {"rank": sc.rank(D["questions"]), "cons": sc.consistency(D["pairs"]),
                     "int": sc.intervention(D["questions"], D["interventions"])}
            else:
                c = over(m["config"])
                sc = BaselinePaired(c, gi, seed)
                Qs = D["questions"]
                if c.baseline == "llm":                 # LLM scoring is costly: the first n_llm per task
                    keep, cnt = set(), {}
                    for q in Qs:
                        cnt[q["task"]] = cnt.get(q["task"], 0) + 1
                        if cnt[q["task"]] <= n_llm:
                            keep.add(q["i"])
                    sub = [q for q in Qs if q["i"] in keep]
                    rk = [None] * len(Qs)
                    for q, rr in zip(sub, sc.rank(sub, cfg, log)):
                        rk[q["i"]] = rr
                    I = [it for it in D["interventions"] if it["q"] in keep]
                    iv = sc.intervention(Qs, I, cfg); ivs = {it["q"]: v for it, v in zip(I, iv)}
                    r = {"rank": rk, "cons": sc.consistency(D["pairs"], D["train_pairs"], log),
                         "int": [ivs.get(it["q"]) for it in D["interventions"]]}
                else:
                    r = {"rank": sc.rank(Qs, cfg, log), "cons": sc.consistency(D["pairs"], D["train_pairs"], log),
                         "int": sc.intervention(Qs, D["interventions"], cfg)}
                if getattr(sc, "threshold", None) is not None:
                    r["threshold"] = sc.threshold
                sc.close()
        except Exception as e:                           # noqa: BLE001  (one system failing must not stop the rest)
            import traceback
            log(f"[paired] {key}: FAILED {e!r}\n{traceback.format_exc()}")
            continue
        r.update({"tag": tag, "seed": seed, "family": m.get("family"), "val_index": _val_index(m),
                  "config": {k: (m.get("config") or {}).get(k) for k in
                             ("baseline", "encoder", "ku_mode", "motif_p", "wm_p_action", "edit_p", "ku_subgraph", "llm_model")}})
        rep["runs"][key] = r
        json.dump(rep, open(path, "w"), default=str)
        log(f"[paired] {key}: done ({time.time() - t1:.0f}s)")
        del sc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return finish(out_dir, exp, rep)


def finish(out_dir, exp="BENCH", rep=None):
    exp_dir = os.path.join(out_dir, exp)
    path = os.path.join(exp_dir, "paired_v2.json")
    rep = rep or json.load(open(path))
    systems, vals = {}, {}
    for key, r in rep["runs"].items():
        S = systems.setdefault(r["tag"], {"seeds": [], "rank": [], "cons": [], "int": [], "family": r.get("family"),
                                          "config": r.get("config")})
        S["seeds"].append(r["seed"]); S["rank"].append(r["rank"]); S["cons"].append(r.get("cons"))
        S["int"].append(r.get("int"))
        if r.get("val_index") is not None:
            vals.setdefault(r["tag"], []).append(r["val_index"])
    val_index = {t: float(np.mean(v)) for t, v in vals.items()}
    summ, cmp = summarise(rep["data"], systems, val_index)
    D = rep["data"]
    head = {"protocol": "v2 (evaluation audit fixes, 25 Oct 2026)",
            "fixes": ["one shared question set for every system", "mid-rank ties",
                      "same candidate encoding for truth, edits and distractors",
                      "consistency on the same pairs: JEPA inverse head; embeddings: probe on own vectors; LLM: summed "
                      "log-probability, threshold tuned on train pairs; balanced accuracy + AUC",
                      "intervention: donor-result rank gain against a topic pool around the donor (not in the headline)",
                      "headline = average of 4 tasks; paired bootstrap over shared questions (pairs by paper); best JEPA "
                      "chosen on validation; seed counts shown"],
            "n": {"ranking": {t: sum(q["task"] == t for q in D["questions"]) for t in RANK_TASKS},
                  "pairs": len(D["pairs"]), "pairs_contradicts": int(sum(p["y"] for p in D["pairs"])),
                  "train_pairs": len(D["train_pairs"]), "interventions": len(D["interventions"]), "bootstrap": B_BOOT},
            "systems": summ, "comparison": cmp}
    rep["summary"] = head
    json.dump(rep, open(path, "w"), default=str)
    dash = os.path.join(out_dir, "dashboard")
    if os.path.isdir(dash):
        json.dump(head, open(os.path.join(dash, "paired_v2.json"), "w"), default=str)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", help="results folder (PAPER_A_OUT), e.g. $STORE/SCI-JEPA/arxiv60k_S")
    p.add_argument("--exp", default="BENCH")
    p.add_argument("--n", type=int, default=300, help="ranking questions per task")
    p.add_argument("--n-llm", type=int, default=150, help="ranking questions per task for the LLM systems")
    p.add_argument("--pairs", type=int, default=3000, help="test claim-evidence pairs")
    p.add_argument("--interventions", type=int, default=150)
    p.add_argument("--only", default="", help="comma-separated arms")
    p.add_argument("--no-supervised", action="store_true", help="skip the supervised regressor (it retrains)")
    p.add_argument("--summary-only", action="store_true", help="recompute the summary from stored records")
    p.add_argument("--device", default="auto")
    A = p.parse_args(argv)
    if A.summary_only:
        rep = finish(A.out, A.exp)
    else:
        rep = run(A.out, A.exp, A.n, A.n_llm, A.pairs, A.interventions, [x for x in A.only.split(",") if x] or None,
                  not A.no_supervised, A.device)
    c = rep["summary"]["comparison"]
    print(f"[paired] {len(rep['summary']['systems'])} systems; best JEPA {c.get('best_jepa')} vs "
          f"{c.get('strongest_comparison')}: " + ", ".join(f"{k} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] "
                                                           f"{v['verdict']}" for k, v in (c.get("metrics") or {}).items()))


if __name__ == "__main__":
    main()
