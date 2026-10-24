"""
Component index over knowledge units (KUs) that instance construction samples
from. Built by ku_graph.build_blob from the normalized parquet; types and
relations come from ku_schema.

Each paper is represented in an instance by a featureless hub carrying only
its coarse field embedding. Whole-paper paraphrases (executive summary, three
takeaways) never become tokens.

PROXY GROUNDED EDGES. Grounded edges need canonical Material / Method / Metric
identities that do not exist yet. Until they do, two KUs of a proxy type in
different papers are linked when their text embeddings have cosine >=
proxy_tau. Results produced with these edges must say "proxy grounded edges".
"""
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.data.ku_schema import (CHILD_RELS, COMP_TYPES, N_RELS, PARAPHRASE_SIBLINGS,  # noqa: F401
                                     REL_CITES, REL_ID, REL_MEMBER, REL_NONE, REL_SAME_FIELD,
                                     REL_SAME_PAPER, REL_SAME_SUBFIELD, REL_DO, ROLE_RELS)

HUB = len(COMP_TYPES)                  # token type id of a paper hub
N_TOKEN_TYPES = HUB + 1


class GraphIndex:
    """Numpy views of the KU graph. Built once, cached to disk."""

    def __init__(self, blob):
        for k, v in blob.items():
            setattr(self, k, v)
        self.n_papers = int(self.paper_field.shape[0])
        self.n_comps = int(self.comp_type.shape[0])
        self._build_lookups()

    def _build_lookups(self):
        P = self.n_papers
        self.paper_comps = [self.pc_idx[self.pc_ptr[p]:self.pc_ptr[p + 1]] for p in range(P)]
        self.comp_edges = defaultdict(list)          # KU -> [(other KU, rel id)]
        child_ids = {REL_ID[r] for r in CHILD_RELS}
        self.children = defaultdict(list)            # claim -> its evidence / implications
        self.parents = defaultdict(list)             # evidence / implication -> its claim
        for s, d, r in zip(self.role_src.tolist(), self.role_dst.tolist(), self.role_rel.tolist()):
            self.comp_edges[s].append((d, r))
            self.comp_edges[d].append((s, r))
            if r in child_ids:
                self.children[s].append(d)
                self.parents[d].append(s)
        self.paper_nbrs = defaultdict(dict)          # paper -> {paper: preference}
        self.cites_pairs = set()
        for a, b, w in zip(self.pp_src.tolist(), self.pp_dst.tolist(), self.pp_w.tolist()):
            self.paper_nbrs[a][b] = max(self.paper_nbrs[a].get(b, 0), w)
            self.paper_nbrs[b][a] = max(self.paper_nbrs[b].get(a, 0), w)
            if w == 1:
                self.cites_pairs.add((a, b)); self.cites_pairs.add((b, a))
        self.field_papers, self.subfield_papers = defaultdict(list), defaultdict(list)
        for p in range(P):
            self.field_papers[int(self.paper_field[p])].append(p)
            self.subfield_papers[int(self.paper_subfield[p])].append(p)
        self.ctx_of, self.cf_of = {}, {}                 # paper_masked: (paper, template) -> text row
        if hasattr(self, "ctx_x"):
            for p, t, r in zip(self.ctx_paper.tolist(), self.ctx_tpl.tolist(), self.ctx_row.tolist()):
                self.ctx_of[(p, self.template_names[t])] = r
            for k, (p, kind, d) in enumerate(zip(self.cf_paper.tolist(), self.cf_kind.tolist(),
                                                 self.cf_donor.tolist())):
                self.cf_of.setdefault((p, ("condition", "method")[kind]), []).append((k, d))
        self.mp_of = defaultdict(list)                   # result / claim -> rows of its minimal-pair variants
        if hasattr(self, "mp_comp"):
            for k, c in enumerate(self.mp_comp.tolist()):
                self.mp_of[c].append(k)
        self._problem = None
        self.split_of = np.full(P, -1, dtype=np.int8)   # 0 train, 1 val, 2 test, -1 none
        self.split_of[self.train_papers] = 0
        self.split_of[self.val_papers] = 1
        self.split_of[self.test_papers] = 2

    def texts(self):
        """-> (full text of every KU, full text of every minimal-pair variant), for the text
        baselines (BM25, other encoders). Kept in a side file next to the index (not in memory)."""
        if getattr(self, "comp_text", None) is None:
            import pickle
            path = getattr(self, "_text_path", None)
            if not path or not os.path.exists(path):
                raise FileNotFoundError("this KU index has no text file: rebuild it (delete the index file "
                                        "or bump INDEX_VERSION) to run the BM25 / encoder baselines")
            with open(path, "rb") as f:
                d = pickle.load(f)
            self.comp_text, self.mp_text = d["comp_text"], d["mp_text"]
        return self.comp_text, getattr(self, "mp_text", [])

    def feat_mean(self, chunk=1 << 20):
        """Mean KU feature, summed in chunks (a full fp32 copy is 34 GB at 500k papers)."""
        if getattr(self, "_mean", None) is None:
            s = np.zeros(self.comp_x.shape[1], np.float64)
            for i in range(0, self.n_comps, chunk):
                s += self.comp_x[i:i + chunk].astype(np.float64).sum(0)
            self._mean = (s / max(self.n_comps, 1)).astype(np.float32)
        return self._mean

    def problem_vec(self):
        """Per paper: the centred, normalised mean text embedding of its problem
        (context + question KUs; all KUs if it has neither). Used for topic-matched candidates."""
        if self._problem is None:
            mu = self.feat_mean()
            prob = [COMP_TYPES.index("context"), COMP_TYPES.index("question")]
            V = np.zeros((self.n_papers, self.comp_x.shape[1]), np.float32)
            for p in range(self.n_papers):
                cs = self.paper_comps[p]
                sel = cs[np.isin(self.comp_type[cs], prob)]
                V[p] = self.comp_x[sel if len(sel) else cs].astype(np.float32).mean(0) - mu if len(cs) else 0
            self._problem = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        return self._problem

    def papers(self, split):
        return {"train": self.train_papers, "val": self.val_papers, "test": self.test_papers}[split]

    def summary(self):
        t = np.bincount(self.comp_type, minlength=len(COMP_TYPES))
        rr = np.bincount(self.role_rel, minlength=N_RELS)
        return {"papers": self.n_papers, "kus": self.n_comps,
                "kus_per_paper": round(self.n_comps / max(self.n_papers, 1), 2),
                "per_type": {c: int(t[i]) for i, c in enumerate(COMP_TYPES) if t[i]},
                "role_edges": {r: int(rr[REL_ID[r]]) for r in ROLE_RELS if rr[REL_ID[r]]},
                "paper_edges_cites": int((self.pp_w == 1).sum()),
                "paper_edges_proxy": int((self.pp_w == 2).sum()),
                "fields": len(self.field_names), "subfields": int(self.paper_subfield.max()) + 1,
                "splits": {s: int(len(self.papers(s))) for s in ("train", "val", "test")},
                "build": self.build_stats,
                "proxy_note": "grounded edges are PROXIES (embedding near-duplicates)"}


# ----------------------------------------------------------------------
def build_proxy_edges(x, comp_paper, tau, topk, device):
    """Cross-paper near-duplicate pairs (i < j) among KUs of one type."""
    n = x.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    z = F.normalize(torch.as_tensor(x, dtype=torch.float32, device=device), dim=-1)
    owner = torch.as_tensor(comp_paper, device=device)
    k = min(topk + 1, n)
    pairs = []
    for s in range(0, n, 4096):
        sim = z[s:s + 4096] @ z.T
        rows = torch.arange(s, min(s + 4096, n), device=device)
        sim[owner[rows].unsqueeze(1) == owner.unsqueeze(0)] = -1.0   # cross-paper only (also kills self)
        val, idx = sim.topk(k, dim=1)
        keep = val >= tau
        r = rows.unsqueeze(1).expand_as(idx)[keep]
        c = idx[keep]
        pairs.append(torch.stack([torch.minimum(r, c), torch.maximum(r, c)], 1).cpu())
    out = torch.unique(torch.cat(pairs, 0), dim=0)
    return out.numpy().astype(np.int64)


def get_index(cfg):
    from sci_jepa.data.ku_graph import build_blob, data_fingerprint
    if cfg.synthetic:
        from sci_jepa.data.synthetic import synthetic_records
        return GraphIndex(build_blob(cfg.override(embedder="hash"), records=synthetic_records(seed=cfg.split_seed)))
    path = os.path.join(cfg.index_dir, f"paper_a_ku_index_{data_fingerprint(cfg)}.pt")
    text_path = path[:-3] + ".texts.pkl"
    if os.path.exists(path):
        print(f"[paper_a.index] loaded {path}")
        gi = GraphIndex(torch.load(path, weights_only=False))
        gi._text_path = text_path
        return gi
    blob = build_blob(cfg)
    os.makedirs(cfg.index_dir, exist_ok=True)
    import pickle
    with open(text_path, "wb") as f:                  # full KU texts: only the text baselines read them
        pickle.dump({"comp_text": blob.pop("comp_text"), "mp_text": blob.pop("mp_text", [])}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(blob, path)
    print(f"[paper_a.index] saved {path}")
    from sci_jepa.data.ku_graph import embed_cache_dir
    import shutil
    shutil.rmtree(embed_cache_dir(cfg), ignore_errors=True)   # resumable chunks are no longer needed
    gi = GraphIndex(blob)
    gi._text_path = text_path
    return gi
