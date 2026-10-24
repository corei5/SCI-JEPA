"""
Training instances for Paper A: a masked component inside a neighbourhood of
N contribution subgraphs.

Construction (plan, "How a training instance is built"):
  1. anchor on a component (claim / method / result);
  2. expand k hops over the paper graph, capped at N subgraphs, preferring
     proxy-grounded neighbours, then citations, then shared field;
  3. mask the anchor according to `mask_policy`;
  4. run the nuisance audit on the sampled neighbourhood.

Tokens: one hub per paper (field embedding only, never paper text) followed by
that paper's components. Global ids never reach the model: a token knows its
type, its features and its typed relations to other tokens, nothing else.

Leak-safe masking. With mask_policy="token+edges+children" (default) the
masked token keeps its type and its membership in its paper, but loses its
features, every role edge, and -- for a claim -- its evidence and implication
children, which paraphrase it. This is the masking the older Part-3 code did
not do: there the encoder saw the full graph and only the mixer was masked.
"""
import numpy as np
import torch

from sci_jepa.data.graph_index import (COMP_TYPES, HUB, PARAPHRASE_SIBLINGS, REL_CITES, REL_ID,
                                       REL_MEMBER, REL_NONE, REL_SAME_FIELD, REL_SAME_PAPER,
                                       REL_SAME_SUBFIELD, ROLE_RELS)

from sci_jepa.data.ku_schema import CTX_KU, REL_DO, SECTION_KUS, TEMPLATES, template_for

_ROLE_IDS = set(REL_ID.values())
CTX_TYPE = COMP_TYPES.index(CTX_KU)            # paper_masked: the joint-text token
DO_SWAP, DO_FLIP = COMP_TYPES.index("do_swap"), COMP_TYPES.index("do_flip")
_SEC_IDS = {COMP_TYPES.index(t) for t in SECTION_KUS}
_POLARITY = (REL_ID["supported_by"], REL_ID["challenged_by"])
# Graph-JEPA-style subgraph targets: a KU's motif = the KUs that form one unit with it
_FAMILY = {"context": "framing", "question": "framing", "method": "approach", "procedure": "approach",
           "result": "outcome", "interpretation": "outcome", "limitation": "outcome"}


def motif_of(gi, anchor):
    """The KUs of anchor's paper that form one sub-graph unit with it: a claim with its
    evidence and implications, or a section family (framing / approach / outcome)."""
    t = COMP_TYPES[gi.comp_type[anchor]]
    p0 = int(gi.comp_paper[anchor])
    if t in ("claim", "evidence", "implication"):
        claim = anchor if t == "claim" else (gi.parents.get(anchor) or [anchor])[0]
        return {int(claim)} | {int(c) for c in gi.children.get(claim, ())}
    fam = _FAMILY.get(t)
    if fam is None:
        return {int(anchor)}
    return {int(c) for c in gi.paper_comps[p0] if _FAMILY.get(COMP_TYPES[gi.comp_type[c]]) == fam}
N_PROBE_CLASSES = len(ROLE_RELS) + 1          # role relations + "no edge"
PROBE_NONE = len(ROLE_RELS)


class Sampler:
    """Draws anchors and neighbourhoods for one split."""

    def __init__(self, gi, cfg, split, seed):
        self.gi, self.cfg = gi, cfg
        self.papers = np.asarray(gi.papers(split))
        self.allowed = np.zeros(gi.n_papers, dtype=bool)
        self.allowed[self.papers] = True
        tt = [COMP_TYPES.index(t) for t in cfg.target_types]
        cand = np.isin(gi.comp_type, tt) & self.allowed[gi.comp_paper]
        self.anchors = np.nonzero(cand)[0]
        self.rng = np.random.default_rng(seed)
        self.frozen = {}                      # E4: anchor -> neighbourhood, when not resampling
        self.given = None                     # evaluation tasks: KU types left visible in the target's paper
        self.train = split == "train"         # paper_masked: random template (train) vs leave-one-out (eval)
        self.p_action = cfg.wm_p_action if self.train else 0.0   # world model: share of instances with an action
        self.ablate = None                    # "typed": evaluate with every typed edge removed (structure reliance)
        # neighbour scores and subfield / field pools depend only on the index, the split and cfg.hops: shared by
        # every sampler of this split (evaluation creates a new sampler per call), one copy per process
        caches = gi.__dict__.setdefault("_sampler_caches", {})
        self._score_cache = caches.setdefault(("scores", split, cfg.hops), {})
        self._group_cache = caches.setdefault(("groups", split), {})

    # ------------------------------------------------------------------
    def neighbourhood(self, p0):
        cfg, gi, rng = self.cfg, self.gi, self.rng
        if not cfg.resample_each_epoch and p0 in self.frozen:
            return self.frozen[p0]
        f0, sf0 = gi.paper_field[p0], gi.paper_subfield[p0]
        ids, sc = self._scores(p0)
        perm = rng.permutation(len(ids))                # random tie-breaking, then best score first (stable)
        order = perm[np.argsort(-sc[perm], kind="stable")]
        chosen = [p0] + [int(q) for q in ids[order[:cfg.n_subgraphs - 1]]]
        for policy, groups, key in (("same_subfield", gi.subfield_papers, sf0),
                                    ("same_field", gi.field_papers, f0)):
            if len(chosen) < cfg.n_subgraphs and cfg.fill_policy in (policy, "same_subfield"):
                # a uniformly random subset of the group's other papers, in random order. Drawing need + len(chosen)
                # and dropping the chosen ones gives the same distribution as shuffling the whole group (the
                # earlier code), in time independent of the group size (a subfield holds thousands at 60k papers)
                arr, need = self._group(policy, key, groups), cfg.n_subgraphs - len(chosen)
                pick = rng.choice(arr, size=min(len(arr), need + len(chosen)), replace=False) if len(arr) else arr
                chosen += [int(q) for q in pick if q not in chosen][:need]
        if len(chosen) < cfg.n_subgraphs and cfg.fill_policy != "none":
            while len(chosen) < cfg.n_subgraphs and len(chosen) < len(self.papers):
                q = int(rng.choice(self.papers))
                if q not in chosen:
                    chosen.append(q)
        if not cfg.resample_each_epoch:
            self.frozen[p0] = chosen
        return chosen

    def _scores(self, p0):
        """k-hop citation neighbours of p0 in this split and their scores (citation weight / hop + subfield / field
        bonus), as two arrays. Deterministic, so computed once per paper and cached (compact: a 60k-paper corpus
        stays within tens of MB per process); only the tie-breaking is random."""
        if p0 in self._score_cache:
            return self._score_cache[p0]
        cfg, gi = self.cfg, self.gi
        f0, sf0 = gi.paper_field[p0], gi.paper_subfield[p0]
        score, frontier = {}, [p0]
        for hop in range(cfg.hops):
            nxt = []
            for p in frontier:
                for q, w in gi.paper_nbrs.get(p, {}).items():
                    if q == p0 or not self.allowed[q]:
                        continue
                    s = w / (hop + 1) + (0.5 if gi.paper_subfield[q] == sf0 else
                                         0.25 if gi.paper_field[q] == f0 else 0.0)
                    if s > score.get(q, -1):
                        score[q] = s; nxt.append(q)
            frontier = nxt
        out = (np.fromiter(score.keys(), dtype=np.int32, count=len(score)),
               np.fromiter(score.values(), dtype=np.float64, count=len(score)))
        self._score_cache[p0] = out
        return out

    def _group(self, policy, key, groups):
        """The papers of one subfield / field that are in this split, as an array (cached)."""
        k = (policy, key)
        if k not in self._group_cache:
            self._group_cache[k] = np.asarray([q for q in groups[key] if self.allowed[q]], dtype=np.int64)
        return self._group_cache[k]

    # ---- anchor pools: drawn WITHOUT replacement, so one pass masks every KU once ----
    def _pool(self, key):
        """key: a KU type name, or "all". Pools are built lazily from self.anchors
        (evaluation narrows self.anchors before the first draw)."""
        if not hasattr(self, "_pools"):
            self._pools, self.passes = {}, {}
        if key not in self._pools:
            ids = self.anchors if key == "all" else \
                self.anchors[self.gi.comp_type[self.anchors] == COMP_TYPES.index(key)]
            w = None
            a = getattr(self.cfg, "train_domain_alpha", 1.0)
            if self.train and a != 1.0 and hasattr(self.gi, "paper_domain") and len(ids):
                dom = self.gi.paper_domain[self.gi.comp_paper[ids]]     # anchors of small domains drawn more often
                cnt = np.bincount(dom, minlength=int(dom.max()) + 1).astype(float)
                w = cnt[dom] ** (a - 1.0); w /= w.sum()
            self._pools[key] = {"ids": ids, "queue": [], "pos": 0, "w": w}
            self.passes[key] = 0
        return self._pools[key]

    def n_anchors(self, key="all"):
        return int(len(self._pool(key)["ids"]))

    # ---- resume: the sampler's position, so a resumed run draws exactly the batches it would have drawn ----
    def get_state(self):
        """Random stream + every pool's shuffled queue and position + passes (+ frozen neighbourhoods, E4).
        The pool ids and weights are rebuilt from the index, so they are not stored."""
        pools = getattr(self, "_pools", {})
        return {"rng": self.rng.bit_generator.state,
                "pools": {k: {"queue": np.asarray(v["queue"]).copy(), "pos": int(v["pos"])} for k, v in pools.items()},
                "passes": dict(getattr(self, "passes", {})),
                "frozen": {int(k): list(v) for k, v in self.frozen.items()}}

    def set_state(self, state):
        self.rng.bit_generator.state = state["rng"]
        for key, p in state["pools"].items():
            pool = self._pool(key)
            pool["queue"], pool["pos"] = np.asarray(p["queue"]), int(p["pos"])
        self.passes.update(state.get("passes", {}))
        self.frozen = {int(k): list(v) for k, v in state.get("frozen", {}).items()}

    def _draw(self, key, k):
        pool, out = self._pool(key), []
        if not len(pool["ids"]):
            return out
        while len(out) < k:
            if pool["pos"] >= len(pool["queue"]):            # pass finished: reshuffle
                if len(pool["queue"]):
                    self.passes[key] += 1
                pool["queue"] = (self.rng.choice(pool["ids"], len(pool["ids"]), p=pool["w"]) if pool.get("w") is not None
                                 else self.rng.permutation(pool["ids"]))
                pool["pos"] = 0
            take = pool["queue"][pool["pos"]:pool["pos"] + k - len(out)]
            pool["pos"] += len(take); out += [int(a) for a in take]
        return out

    def sample(self, batch_size, plan=None):
        """plan: None = every target type mixed in proportion to its KU count, or
        {type: count} for scheduled masking (balanced / round-robin / sequential)."""
        return self.build(self.draw(batch_size, plan))

    def draw(self, batch_size, plan=None):
        """The anchors (hidden KUs) of one batch: cheap, and the only part that moves the epoch pools."""
        if plan is None:
            return self._draw("all", batch_size)
        return [a for t, k in plan.items() if k > 0 for a in self._draw(t, k)]

    def build(self, anchors, rng=None):
        """Training instances for drawn anchors. rng: a separate random stream for this batch (parallel batch
        building gives every step its own, so the batches do not depend on the number of worker processes);
        None = the sampler's own stream."""
        if rng is None:
            return self._build(anchors)
        own, self.rng = self.rng, rng
        try:
            return self._build(anchors)
        finally:
            self.rng = own

    def _build(self, anchors):
        out = []
        k = self.cfg.targets_per_instance if self.train else 1
        for a in anchors:
            inst = build_instance(self.gi, int(a), self.neighbourhood(int(self.gi.comp_paper[a])),
                                  self.cfg, self.rng, given=self.given, train=self.train, n_targets=k)
            if self.cfg.audit_drop and inst["audit"]["flagged"]:
                continue
            if self.p_action and self.rng.random() < self.p_action:
                inst = add_action(self.gi, inst, self.rng, self.papers) or inst
            if self.ablate:
                inst = ablate_structure(inst)
            out.append(inst)
        return out


# ----------------------------------------------------------------------
def _paper_tokens(gi, p, cfg, rng, keep):
    comps = gi.paper_comps[p]
    if len(comps) > cfg.max_comp_per_paper:
        rest = comps[~np.isin(comps, keep)]
        pick = rng.choice(rest, size=cfg.max_comp_per_paper - int(np.isin(comps, keep).sum()),
                          replace=False)
        comps = np.concatenate([comps[np.isin(comps, keep)], pick])
    return comps


def build_view(gi, papers, comps_per_paper, cfg):
    """Tokens + symmetric relation matrix for a set of papers (no masking)."""
    xs, types, comp_ids, slots, deg = [], [], [], [], []
    for slot, (p, comps) in enumerate(zip(papers, comps_per_paper)):
        xs.append(gi.field_x[gi.paper_field[p]][None].astype(np.float32))
        types.append([HUB]); comp_ids.append([-1]); slots.append([slot]); deg.append([len(comps)])
        xs.append(gi.comp_x[comps].astype(np.float32))
        types.append(gi.comp_type[comps]); comp_ids.append(comps)
        slots.append(np.full(len(comps), slot)); deg.append(np.full(len(comps), len(comps)))
    x = np.concatenate(xs, 0)
    typ = np.concatenate([np.asarray(t) for t in types]).astype(np.int64)
    cid = np.concatenate([np.asarray(c) for c in comp_ids]).astype(np.int64)
    slot = np.concatenate([np.asarray(s) for s in slots]).astype(np.int64)
    deg = np.concatenate([np.asarray(d) for d in deg]).astype(np.int64)
    T = len(typ)

    rel = np.zeros((T, T), dtype=np.int64)
    same = slot[:, None] == slot[None, :]
    rel[same] = REL_SAME_PAPER
    hub = typ == HUB
    rel[np.ix_(hub, ~hub)] = np.where(same[np.ix_(hub, ~hub)], REL_MEMBER, REL_NONE)
    rel[np.ix_(~hub, hub)] = rel[np.ix_(hub, ~hub)].T
    hub_pos = np.nonzero(hub)[0]
    for i in hub_pos:                                   # paper-level relations
        for j in hub_pos:
            if i == j:
                continue
            pi, pj = papers[slot[i]], papers[slot[j]]
            if (pi, pj) in gi.cites_pairs:
                rel[i, j] = REL_CITES
            elif gi.paper_subfield[pi] == gi.paper_subfield[pj]:
                rel[i, j] = REL_SAME_SUBFIELD
            elif gi.paper_field[pi] == gi.paper_field[pj]:
                rel[i, j] = REL_SAME_FIELD
    pos = {int(c): k for k, c in enumerate(cid) if c >= 0}
    for c, k in pos.items():                            # role + proxy edges present in the view
        for d, r in gi.comp_edges.get(c, ()):
            m = pos.get(d)
            if m is not None:
                rel[k, m] = rel[m, k] = r
    np.fill_diagonal(rel, REL_NONE)
    return {"x": x, "type": typ, "comp": cid, "slot": slot, "deg": deg, "rel": rel}


def _drop_tokens(view, drop):
    keep = ~np.isin(np.arange(len(view["type"])), drop)
    out = {k: (v[keep] if k != "rel" else v[np.ix_(keep, keep)]) for k, v in view.items()}
    return out, np.cumsum(keep) - 1


def _pick_template(gi, anchor, p0, given, train, rng):
    """paper_masked: which visible-text template the target's paper is shown with."""
    t = COMP_TYPES[gi.comp_type[anchor]]
    if given is not None:
        name = template_for(given)
        if name is None:
            raise KeyError(f"no paper_masked template has given = {sorted(given)}")
        return name
    ok = [n for n, (_, tg) in TEMPLATES.items() if t in tg and (p0, n) in gi.ctx_of]
    if not ok:
        return None
    if train:
        return ok[int(rng.integers(len(ok)))]
    return f"all-{t}" if f"all-{t}" in ok else ok[0]


def _add_text_token(view, x, slot):
    """Append ONE context-text token (the visible part of paper `slot`)."""
    T = len(view["type"])
    rel = np.zeros((T + 1, T + 1), dtype=np.int64)
    rel[:T, :T] = view["rel"]
    same = view["slot"] == slot
    row = np.where(same, np.where(view["type"] == HUB, REL_MEMBER, REL_SAME_PAPER), REL_NONE)
    rel[T, :T] = rel[:T, T] = row
    out = {"x": np.concatenate([view["x"], x[None].astype(np.float32)]),
           "type": np.append(view["type"], CTX_TYPE), "comp": np.append(view["comp"], -2),
           "slot": np.append(view["slot"], slot), "deg": np.append(view["deg"], 1), "rel": rel}
    return out


def _leak_family(gi, c, cfg):
    """KUs that restate c and are hidden with it (mask_policy): its claim / children and,
    with "siblings", the paraphrase siblings in its paper (method <-> procedure)."""
    out = set()
    if "children" in cfg.mask_policy:
        out |= set(gi.children.get(c, ())) | set(gi.parents.get(c, ()))
        if "siblings" in cfg.mask_policy:
            sib = {COMP_TYPES.index(t) for t in PARAPHRASE_SIBLINGS.get(COMP_TYPES[gi.comp_type[c]], ())}
            out |= {int(d) for d in gi.paper_comps[int(gi.comp_paper[c])] if gi.comp_type[d] in sib}
    return out


def _extra_anchors(gi, view, papers, cfg, rng, k):
    """Multi-target training (targets_per_instance = k): one more hidden KU in each of
    k - 1 neighbouring papers, as I-JEPA predicts several target blocks per image."""
    tt = np.asarray([COMP_TYPES.index(t) for t in cfg.target_types])
    out = []
    for s in rng.permutation(np.arange(1, len(papers))):
        if len(out) >= k - 1:
            break
        cand = view["comp"][(view["slot"] == s) & (view["comp"] >= 0) & np.isin(view["type"], tt)]
        if len(cand):
            out.append((int(s), int(rng.choice(cand))))
    return out


def build_instance(gi, anchor, papers, cfg, rng, given=None, train=True, n_targets=1):
    """given: optional set of KU type names. Only these stay visible in the target's
    own paper (evaluation tasks, e.g. forward prediction: result | problem, method,
    conditions); neighbouring papers are unchanged.
    n_targets > 1 (training only): also hide one KU in each of n_targets - 1 neighbouring
    papers, leak-safe masked like the anchor, each with its own target view."""
    p0 = int(gi.comp_paper[anchor])
    assert papers[0] == p0
    comps = [_paper_tokens(gi, p, cfg, rng, keep=np.array([anchor]) if p == p0 else np.array([], int))
             for p in papers]
    view = build_view(gi, papers, comps, cfg)
    t_view = view if cfg.target_scope == "instance" else build_view(gi, [p0], comps[:1], cfg)
    t_pos = int(np.nonzero(t_view["comp"] == anchor)[0][0])
    template = None
    if cfg.ku_mode == "paper_masked":       # same target + neighbours; the target's own paper is
        template = _pick_template(gi, anchor, p0, given, train, rng)   # shown as ONE joint text
        view = build_view(gi, papers, [np.array([anchor])] + comps[1:], cfg)
        if template is not None:
            view = _add_text_token(view, gi.ctx_x[gi.ctx_of[(p0, template)]], 0)
        given = None                         # already applied: the text holds exactly the given KUs

    # ---- masking ------------------------------------------------------------
    ctx = {k: v.copy() for k, v in view.items()}
    m = int(np.nonzero(ctx["comp"] == anchor)[0][0])
    drop, motif = set(), set()
    if train and cfg.task_mask_p and cfg.ku_mode == "section" and given is None and rng.random() < cfg.task_mask_p:
        t_name = COMP_TYPES[gi.comp_type[anchor]]       # task-aware masking: show only a reasoning template's
        fit = [g for n, (g, tg) in TEMPLATES.items()     # given parts, as the five reasoning queries do
               if not n.startswith("all-") and t_name in tg]
        if fit:
            given = set(fit[rng.integers(len(fit))])
    elif train and cfg.motif_p and cfg.ku_mode == "section" and given is None and rng.random() < cfg.motif_p:
        motif = motif_of(gi, anchor) - {anchor}          # predicted together with the anchor (training only)
    if given is not None:
        keep_t = {COMP_TYPES.index(t) for t in given if t in COMP_TYPES}
        drop |= {int(c) for c in gi.paper_comps[p0] if int(c) != anchor and gi.comp_type[c] not in keep_t}
    drop |= _leak_family(gi, anchor, cfg)               # e.g. masked claim hides its evidence, method its procedure
    others = []                                         # multi-target: (slot, KU) hidden in neighbouring papers
    if train and n_targets > 1 and cfg.ku_mode == "section":
        others = _extra_anchors(gi, ctx, papers, cfg, rng, n_targets)
        for _, c in others:
            drop |= _leak_family(gi, c, cfg)
        drop -= {c for _, c in others}
    drop -= motif                                       # motif members are predicted, not removed
    kids = np.nonzero(np.isin(ctx["comp"], list(drop)))[0]
    if len(kids):
        ctx, remap = _drop_tokens(ctx, kids)
        m = int(remap[m])
    hidden = [m] + [int(k) for k in np.nonzero(np.isin(ctx["comp"], list(motif)))[0]]
    n_motif = len(hidden)
    hidden += [int(np.nonzero(ctx["comp"] == c)[0][0]) for _, c in others]
    is_mask = np.zeros(len(ctx["type"]), dtype=bool); is_mask[hidden] = True
    raw_target = ctx["x"][m].copy()
    ctx["x"][hidden] = 0.0
    if "edges" in cfg.mask_policy:                      # hidden nodes keep the links AMONG themselves
        group = np.zeros_like(is_mask); group[hidden[:n_motif]] = True   # (the motif's shape), lose those
        for k in hidden:                                # to visible KUs; multi-target nodes keep none
            row = ctx["rel"][k]
            role = np.isin(row, list(_ROLE_IDS)) & ~(group if group[k] else np.zeros_like(is_mask))
            same = ctx["slot"] == ctx["slot"][k]
            row[role] = np.where(same[role], REL_SAME_PAPER, REL_NONE)
            ctx["rel"][:, k] = row
    # extra hidden nodes: (ctx position, position in its target view, type, target view: 0 = the
    # anchor's, j = extra_views[j - 1])
    tpos = {int(c): k for k, c in enumerate(t_view["comp"]) if c >= 0}
    extra = [(k, tpos[int(ctx["comp"][k])], int(ctx["type"][k]), 0) for k in hidden[1:n_motif]
             if int(ctx["comp"][k]) in tpos]
    extra_views = []
    for (s, c), k in zip(others, hidden[n_motif:]):
        tv = build_view(gi, [papers[s]], [comps[s]], cfg)
        extra_views.append(tv)
        extra.append((k, int(np.nonzero(tv["comp"] == c)[0][0]), int(ctx["type"][k]), len(extra_views)))

    # ---- inverse-relation probes: hide some role edges from the encoder ---------
    probes, labels = [], []
    role_edges = _role_edges(ctx["rel"], exclude=m)
    if role_edges is not None and len(role_edges) and is_mask.sum() > 1:
        role_edges = role_edges[~(is_mask[role_edges[:, 0]] | is_mask[role_edges[:, 1]])]
    if role_edges is not None and len(role_edges) and cfg.n_probe_edges:
        pick = rng.choice(len(role_edges), size=min(cfg.n_probe_edges, len(role_edges)), replace=False)
        for k in pick:
            i, j = (int(v) for v in role_edges[k])
            labels.append(int(ctx["rel"][i, j]) - 3)
            base = REL_SAME_PAPER if ctx["slot"][i] == ctx["slot"][j] else REL_NONE
            ctx["rel"][i, j] = ctx["rel"][j, i] = base
            probes.append((i, j))
            neg = _hard_negative(ctx, i, j, m, rng)
            if neg is not None:
                probes.append(neg); labels.append(PROBE_NONE)

    # ---- nuisance audit ----------------------------------------------------------
    vis = (~is_mask) & (ctx["type"] != HUB)
    z = raw_target.astype(np.float64) / (np.linalg.norm(raw_target) + 1e-8)
    if vis.any():
        V = ctx["x"][vis].astype(np.float64)
        sims = V @ z / (np.linalg.norm(V, axis=1) + 1e-8)
        max_sim = float(sims.max())
    else:
        max_sim = 0.0
    same_type_left = int(((ctx["type"] == ctx["type"][m]) & vis & (ctx["slot"] == 0)).sum())
    audit = {"max_ctx_sim": max_sim,
             "singleton_type": same_type_left == 0,     # type + subgraph identity fix the target
             "n_papers": len(papers),
             "flagged": max_sim >= cfg.audit_tau}

    return {"ctx": ctx, "is_mask": is_mask, "mask_pos": m,
            "target_view": t_view, "target_pos": t_pos,
            "target_comp": anchor, "target_type": int(gi.comp_type[anchor]),
            "probes": probes, "probe_labels": labels, "audit": audit, "papers": papers, "template": template,
            "extra": extra, "extra_views": extra_views}


# ---------------------------------------------------------------- world model: actions
def add_action(gi, inst, rng, pool, kind=None, donor=None):
    """JEPA world model: (state, action) -> latent of the NEXT state.

    The context keeps the ORIGINAL graph and gets one action token; the target view
    is the hidden KU's paper AFTER the action, so the target moves and the predictor
    must learn how the action changes the hidden KU's latent.
      swap  change(KU := another paper's KU of the same type): content intervention.
            Action token = the new KU's text feature, linked by "do" to the KU it replaces.
      flip  supported_by <-> challenged_by on one claim-evidence edge: a purely
            structural intervention (no text changes). Action token = zeros, linked by
            "do" to both ends of the edge.
    Returns None when the paper offers no valid action."""
    ctx, tv = inst["ctx"], inst["target_view"]
    p0, anchor = int(inst["papers"][0]), int(inst["target_comp"])
    own = (ctx["slot"] == 0) & (ctx["comp"] >= 0)
    flips = [(i, j) for i, j in zip(*np.nonzero(np.triu(np.isin(ctx["rel"], _POLARITY))))
             if own[i] and own[j]]
    swaps = [i for i in np.nonzero(own)[0] if ctx["comp"][i] != anchor and ctx["type"][i] in _SEC_IDS
             and ctx["type"][i] != inst["target_type"]]
    kind = kind or ("flip" if flips and (not swaps or rng.random() < 0.5) else "swap")
    tv2 = {k: v.copy() for k, v in tv.items()}
    if kind == "swap":
        if not swaps:
            return None
        i = int(swaps[rng.integers(len(swaps))]); c = int(ctx["comp"][i]); t = int(ctx["type"][i])
        if donor is None:
            others = [q for q in rng.permutation(pool)[:50] if int(q) != p0 and int(q) not in inst["papers"]]
            cand = [int(d) for q in others for d in gi.paper_comps[int(q)] if gi.comp_type[d] == t]
            if not cand:
                return None
            donor = cand[0]
        k = np.nonzero(tv2["comp"] == c)[0]
        if not len(k):
            return None
        tv2["x"][k] = gi.comp_x[donor].astype(np.float32)
        x_act, typ, linked, info = gi.comp_x[donor].astype(np.float32), DO_SWAP, [i], \
            {"kind": "swap", "node": c, "type": COMP_TYPES[t], "donor": int(donor)}
    else:
        if not flips:
            return None
        i, j = (int(v) for v in flips[rng.integers(len(flips))])
        ci, cj = int(ctx["comp"][i]), int(ctx["comp"][j])
        a, b = np.nonzero(tv2["comp"] == ci)[0], np.nonzero(tv2["comp"] == cj)[0]
        if not len(a) or not len(b):
            return None
        r = tv2["rel"][a[0], b[0]]
        tv2["rel"][a[0], b[0]] = tv2["rel"][b[0], a[0]] = _POLARITY[1] if r == _POLARITY[0] else _POLARITY[0]
        x_act, typ, linked, info = np.zeros(ctx["x"].shape[1], np.float32), DO_FLIP, [i, j], \
            {"kind": "flip", "nodes": [ci, cj], "to": "challenged_by" if r == _POLARITY[0] else "supported_by"}
    T = len(ctx["type"])
    rel = np.zeros((T + 1, T + 1), dtype=np.int64); rel[:T, :T] = ctx["rel"]
    for i in linked:
        rel[T, i] = rel[i, T] = REL_DO
    ctx2 = {"x": np.concatenate([ctx["x"], x_act[None]]), "type": np.append(ctx["type"], typ),
            "comp": np.append(ctx["comp"], -3), "slot": np.append(ctx["slot"], 0),
            "deg": np.append(ctx["deg"], 1), "rel": rel}
    return {**inst, "ctx": ctx2, "is_mask": np.append(inst["is_mask"], False), "target_view": tv2, "action": info}


def ablate_structure(inst):
    """Remove every typed edge from the context (role, proxy, cites, field links): KUs of
    one paper stay 'same paper', different papers 'none'. Used to measure how much
    accuracy depends on graph structure."""
    c = {k: v.copy() for k, v in inst["ctx"].items()}
    same = c["slot"][:, None] == c["slot"][None, :]
    keep = np.isin(c["rel"], [REL_NONE, REL_SAME_PAPER, REL_MEMBER, REL_DO])
    c["rel"] = np.where(keep, c["rel"], np.where(same, REL_SAME_PAPER, REL_NONE))
    return {**inst, "ctx": c}


def _role_edges(rel, exclude=-1):
    """(E, 2) upper-triangle token pairs joined by a role relation."""
    hit = np.isin(rel, list(_ROLE_IDS))
    hit = np.triu(hit, 1)
    if exclude >= 0:
        hit[exclude, :] = False; hit[:, exclude] = False
    return np.argwhere(hit)


def _hard_negative(ctx, i, j, m, rng, tries=30):
    """Same type pair as (i, j) with no edge, preferably across subgraphs."""
    A = np.nonzero(ctx["type"] == ctx["type"][i])[0]
    B = np.nonzero(ctx["type"] == ctx["type"][j])[0]
    fallback = None
    for _ in range(tries):
        a, b = int(rng.choice(A)), int(rng.choice(B))
        if a == b or m in (a, b) or (a, b) == (i, j):
            continue
        if ctx["rel"][a, b] not in (REL_NONE, REL_SAME_PAPER):
            continue
        if ctx["slot"][a] != ctx["slot"][b]:
            return (a, b)
        fallback = fallback or (a, b)
    return fallback


# ----------------------------------------------------------------------
def minimal_pair(gi, inst, rng, sampler_papers):
    """Same instance with ONE visible context component replaced by a same-type
    component from a paper outside the neighbourhood (relational contrast, E5)."""
    ctx = {k: v.copy() for k, v in inst["ctx"].items()}
    cand = np.nonzero((ctx["type"] != HUB) & ~inst["is_mask"])[0]
    if not len(cand):
        return None
    k = int(rng.choice(cand))
    t = ctx["type"][k]
    outside = np.setdiff1d(sampler_papers, inst["papers"])
    for _ in range(20):
        q = int(rng.choice(outside))
        pool = gi.paper_comps[q][gi.comp_type[gi.paper_comps[q]] == t]
        if len(pool):
            c = int(rng.choice(pool))
            ctx["x"][k] = gi.comp_x[c].astype(np.float32)
            ctx["comp"][k] = c
            return {**inst, "ctx": ctx}
    return None


def perturb_relation(inst, rng):
    """Swap one visible role edge (i, j) for (i, k), k a same-type token in another
    subgraph. Used by the relation-sensitivity diagnostic."""
    ctx = {k: v.copy() for k, v in inst["ctx"].items()}
    edges = _role_edges(ctx["rel"])
    rng.shuffle(edges)
    for i, j in edges:
        i, j = int(i), int(j)
        others = np.nonzero((ctx["type"] == ctx["type"][j]) & (ctx["slot"] != ctx["slot"][j]))[0]
        if len(others):
            k = int(rng.choice(others)); r = ctx["rel"][i, j]
            ctx["rel"][i, j] = ctx["rel"][j, i] = (REL_SAME_PAPER if ctx["slot"][i] == ctx["slot"][j]
                                                   else REL_NONE)
            ctx["rel"][i, k] = ctx["rel"][k, i] = r
            return {**inst, "ctx": ctx}
    return None


# ----------------------------------------------------------------------
def _pad(views, key, T, fill, dtype):
    out = np.full((len(views), T) + views[0][key].shape[1:], fill, dtype=dtype)
    for b, v in enumerate(views):
        out[b, :len(v[key])] = v[key]
    return torch.as_tensor(out)


def collate_views(views, is_mask=None):
    T = max(len(v["type"]) for v in views)
    batch = {"x": _pad(views, "x", T, 0.0, np.float32),
             "type": _pad(views, "type", T, HUB, np.int64),
             "slot": _pad(views, "slot", T, -1, np.int64),
             "deg": _pad(views, "deg", T, 0, np.int64),
             "comp": _pad(views, "comp", T, -1, np.int64),
             "pad": torch.as_tensor(np.stack([np.arange(T) >= len(v["type"]) for v in views]))}
    rel = np.zeros((len(views), T, T), dtype=np.uint8)   # ids < 256: 8x less to copy; to_device widens them
    for b, v in enumerate(views):
        n = len(v["type"]); rel[b, :n, :n] = v["rel"]
    batch["rel"] = torch.as_tensor(rel)
    mk = np.zeros((len(views), T), dtype=bool)
    if is_mask is not None:
        for b, m in enumerate(is_mask):
            mk[b, :len(m)] = m
    batch["is_mask"] = torch.as_tensor(mk)
    return batch


def collate(instances):
    ctx = collate_views([i["ctx"] for i in instances], [i["is_mask"] for i in instances])
    # target views: one per instance (rows 0..B-1), then the multi-target views of neighbouring papers
    views = [i["target_view"] for i in instances]
    eb, ec, et, er = [], [], [], []                      # hidden nodes predicted besides the anchor
    for b, inst in enumerate(instances):
        first = len(views)
        views += inst.get("extra_views", [])
        for cp, tp, _, j in inst.get("extra", ()):
            eb.append(b); ec.append(cp); et.append(tp); er.append(b if j == 0 else first + j - 1)
    tgt = collate_views(views)
    pb, pi, pj, pl = [], [], [], []
    for b, inst in enumerate(instances):
        for (i, j), l in zip(inst["probes"], inst["probe_labels"]):
            pb.append(b); pi.append(i); pj.append(j); pl.append(l)
    return {"ctx": ctx, "tgt": tgt,
            "mask_pos": torch.tensor([i["mask_pos"] for i in instances]),
            "target_pos": torch.tensor([i["target_pos"] for i in instances]),
            "target_type": torch.tensor([i["target_type"] for i in instances]),
            "target_comp": torch.tensor([i["target_comp"] for i in instances]),
            "probe": (torch.tensor(pb, dtype=torch.long), torch.tensor(pi, dtype=torch.long),
                      torch.tensor(pj, dtype=torch.long), torch.tensor(pl, dtype=torch.long)),
            "extra": (torch.tensor(eb, dtype=torch.long), torch.tensor(ec, dtype=torch.long),
                      torch.tensor(et, dtype=torch.long), torch.tensor(er, dtype=torch.long)),
            "audit": [i["audit"] for i in instances]}
