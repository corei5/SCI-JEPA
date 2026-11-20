"""
The model behind the API: a trained SCI-JEPA folder (sci_jepa.inference.SciJEPA) plus, optionally,
the corpus KU index it was trained on. Everything here is synchronous and single-threaded; the app
holds one lock around it (one GPU, one model).

How answers are found (ranking tasks: forward, inverse, abduction)
  1. Shortlist papers. Every corpus paper has a vector: the centred mean of its KUs' frozen text
     embeddings (gi.comp_x), as SciJEPA uses to pick neighbouring papers. The user's parts are
     embedded with the same encoder; the `shortlist_papers` most similar papers (after the year and
     field filters) are kept. The vectors are computed once and cached next to the index.
  2. Score their KUs of the answer type(s):
       jepa  cosine between Sci-JEPA's predicted representation of the hidden part (predictor) and
             each candidate's target-encoder key, its paper encoded alone: the evaluation's rule
             (evaluation.tasks._retrieval with diagnostics.encode_papers)
       text  cosine between the user's text and the candidate's text (frozen encoder, centred)
       bm25  BM25 of the user's words in the candidate's text, over the shortlist only
     hybrid = 0.6 jepa + 0.3 text + 0.1 bm25, each min-max scaled within the request.
Scores are relative within one request (the specification's limitation), not probabilities.

Without an index the model still answers consistency, edit_check and intervention's shift;
the ranking endpoints return 503.
"""
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.api import textparts as T
from sci_jepa.data.ku_schema import COMP_TYPES, SECTION_KUS

ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_OLD = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")
PART_COLUMN = {"problem": "research_context", "method": "methodological_details",
               "conditions": "procedures_architectures", "result": "key_results"}
HYBRID = {"jepa": 0.6, "text": 0.3, "bm25": 0.1}
ATTRIBUTION = "Knowledge units from Scientific-Summaries (LAION, CC-BY-4.0); original papers on arXiv."


class NotReady(RuntimeError):
    """A resource (index, full texts) the request needs is not loaded."""


class NotFound(KeyError):
    pass


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def bm25(query, docs, k1=1.5, b=0.75):
    tok = lambda s: [w.lower() for w in T.words(s) if w.lower() not in T.STOP]
    D = [tok(d) for d in docs]
    if not D:
        return np.zeros(0)
    N, avg = len(D), max(1e-9, sum(map(len, D)) / len(D))
    df = Counter(w for d in D for w in set(d))
    q = set(tok(query))
    out = np.zeros(N)
    for i, d in enumerate(D):
        tf = Counter(d)
        for w in q:
            if tf[w]:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                out[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
    return out


def _minmax(x):
    x = np.asarray(x, np.float64)
    if not len(x):
        return x
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class Engine:
    def __init__(self, model_dir, weights="model", device="auto", index="auto", cache_dir=None,
                 full_texts=False, shortlist_papers=64, neighbours=True, version="0.1.0",
                 edit_threshold=0.02, warm_up=True, log=print):
        self.model_dir, self.weights, self.device = model_dir, weights, device
        self.index_spec, self.cache_dir, self.want_full_texts = index, cache_dir, full_texts
        self.shortlist_papers, self.neighbours = shortlist_papers, neighbours
        self.version, self.edit_threshold, self.log = version, edit_threshold, log
        self.warm = warm_up
        self.jepa = self.gi = None
        self.V = self.mu = None
        self.comp_text = None
        self.index_path = None
        self.state = {"model_loaded": False, "index_loaded": False, "search_loaded": False, "error": None}
        self.lock = threading.RLock()
        self._pid_of, self._arxiv_sorted = {}, []
        self.domain_names, self.field_names = [], []
        self._ref = None

    # ================================================================ loading
    def load(self):
        try:
            self._load_model()
            if self.index_spec not in (None, "", "none"):
                self._load_index()
                self._load_search()
        except Exception as e:                           # /v1/ready shows it; the process stays up
            self.state["error"] = f"{type(e).__name__}: {e}"
            self.log(f"[api] load failed: {self.state['error']}")
            raise
        return self

    def _load_model(self):
        from sci_jepa.inference import SciJEPA
        t0 = time.time()
        raw = json.load(open(os.path.join(self.model_dir, "config.json")))
        self.raw_cfg = raw
        self.jepa = SciJEPA.from_pretrained(self.model_dir, weights=self.weights, device=self.device)
        if raw.get("embedder") == "hash":                # synthetic smoke-test models
            from sci_jepa.data.ku_graph import HashEmbedder
            self.jepa._emb = HashEmbedder(self.jepa.cfg.feat_dim)
        self.cfg, self.dev = self.jepa.cfg, self.jepa.dev
        f = os.path.join(self.model_dir, f"{self.weights}.safetensors")
        self.weights_file = f if os.path.exists(f) else os.path.join(self.model_dir, "model.safetensors")
        self.weights_sha256 = _sha256(self.weights_file)
        sp = os.path.join(self.model_dir, "summary.json")
        self.summary = json.load(open(sp)) if os.path.exists(sp) else {}
        _ = self.jepa.text_encoder                       # load the frozen text encoder now, not on the first call
        self.state["model_loaded"] = True
        self.log(f"[api] model {self.model_dir} ({self.weights}) on {self.dev} in {time.time() - t0:.0f}s")

    def _load_index(self):
        from sci_jepa.data.graph_index import GraphIndex, _add_lookups, get_index, load_index
        t0 = time.time()
        if self.cfg.synthetic:
            self.gi = get_index(self.cfg)
            self.index_path = None
        else:
            path = self.index_spec
            if path == "auto":                           # the index the model was trained on: the fingerprint it
                d = os.environ.get("PAPER_A_INDEX_DIR", self.cfg.index_dir)   # recorded, else recomputed
                fp = self.summary.get("data_fingerprint")
                if not fp:
                    from sci_jepa.data.ku_graph import data_fingerprint
                    fp = data_fingerprint(self.cfg)
                path = os.path.join(d, f"paper_a_ku_index_{fp}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"KU index {path} not found: pass --index <paper_a_ku_index_*.pt> "
                                        "(the index the model was trained on) or --index none")
            self.gi = GraphIndex(_add_lookups(load_index(path), path))
            self.gi._text_path = path[:-3] + ".texts.pkl"
            self.index_path = path
        if self.gi.comp_x.shape[1] != self.cfg.feat_dim:
            raise ValueError(f"index features are {self.gi.comp_x.shape[1]}-d, the model expects {self.cfg.feat_dim}-d")
        self.field_names = [str(x) for x in self.gi.field_names]          # fine labels ("subfield")
        self.domain_names = ([str(x) for x in self.gi.domain_names]       # the 10 canonical domains ("field")
                             if hasattr(self.gi, "paper_domain") and getattr(self.gi, "domain_names", None) else [])
        if self.want_full_texts:
            self.comp_text, _ = self.gi.texts()
        self.state["index_loaded"] = True
        self.log(f"[api] index: {self.gi.n_papers} papers, {self.gi.n_comps} KUs in {time.time() - t0:.0f}s")

    def _vec_cache(self):
        if self.index_path is None:
            return None
        d = self.cache_dir or os.path.dirname(self.index_path)
        return os.path.join(d, "api_paper_vecs_" + os.path.basename(self.index_path)[:-3] + ".npz")

    def _load_search(self):
        t0, gi = time.time(), self.gi
        cache = self._vec_cache()
        if cache and os.path.exists(cache):
            z = np.load(cache)
            V, mu = z["V"], z["mu"]
        else:
            V, mu = self._paper_vectors()
            if cache:
                try:
                    os.makedirs(os.path.dirname(cache), exist_ok=True)
                    np.savez(cache + ".tmp.npz", V=V, mu=mu)
                    os.replace(cache + ".tmp.npz", cache)
                except OSError as e:
                    self.log(f"[api] paper vectors not cached ({e}); pass --cache-dir")
        self.V, self.mu = torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32)), mu.astype(np.float32)
        gi._mean = self.mu                               # feat_mean() without another pass over comp_x
        if self.neighbours and self.cfg.n_subgraphs > 1: # SciJEPA's neighbouring papers, from the same vectors
            train = np.asarray(gi.papers("train"))
            self.jepa.gi = gi
            self.jepa._pv = (train, V[train], self.mu)
        self._year = torch.from_numpy(np.asarray(gi.paper_year, dtype=np.int64))
        self._domain = (torch.from_numpy(np.asarray(gi.paper_domain, dtype=np.int64))
                        if self.domain_names else None)
        arx = [(str(x), i) for i, x in enumerate(gi.paper_ids) if self._is_arxiv(str(x))]
        arx.sort()
        self._arxiv_sorted = [a for a, _ in arx]          # for prefix search
        self._pid_of = dict(arx)
        if self.warm:
            self._warm_up()
        self.state["search_loaded"] = True
        self.log(f"[api] paper search ready in {time.time() - t0:.0f}s")

    def _warm_up(self):
        """One ranking and the intervention reference before /v1/ready says yes: the first user request
        would otherwise pay for CUDA kernels, the encoder's first batches and the reference papers (~10 s).
        It holds the lock: the model is already marked loaded, so requests may arrive meanwhile."""
        t0 = time.time()
        try:
            demo = {"problem": "Predicting reaction yields from written laboratory protocols.",
                    "method": "Fine-tuning a small language model on the protocol text.",
                    "conditions": "About 50,000 coupling protocols from patents, 80/20 split."}
            with self.lock:
                self.rank("forward", demo, top_k=1)
                self._reference_predictions()
            self.log(f"[api] warm-up in {time.time() - t0:.0f}s")
        except Exception as e:                           # never block start-up on it
            self.log(f"[api] warm-up skipped: {type(e).__name__}: {e}")

    def _paper_vectors(self, chunk=1 << 20):
        """Centred, normalised mean KU text embedding of every paper, one pass over comp_x."""
        gi = self.gi
        P, N, d = gi.n_papers, gi.n_comps, gi.comp_x.shape[1]
        S = torch.zeros(P, d, dtype=torch.float32)
        cp = np.asarray(gi.comp_paper, dtype=np.int64)
        for i in range(0, N, chunk):
            x = torch.from_numpy(np.asarray(gi.comp_x[i:i + chunk], dtype=np.float32))
            S.index_add_(0, torch.from_numpy(cp[i:i + chunk]), x)
            if (i // chunk) % 10 == 0:
                self.log(f"[api] paper vectors: {min(i + chunk, N)}/{N} KUs")
        n = torch.bincount(torch.from_numpy(cp), minlength=P).clamp_min(1).float()
        mu = S.sum(0) / max(N, 1)
        V = F.normalize(S / n[:, None] - mu, dim=1)
        return V.numpy(), mu.numpy()

    # ================================================================ facts for /v1/info
    @property
    def model_id(self):
        return f"sci-jepa-{self.version}"

    def model_ref(self):
        return {"id": self.model_id, "checkpoint": self.summary.get("best_epoch") if self.weights == "best" else "final",
                "weights": self.weights, "sha256": self.weights_sha256[:16],
                "data_version": os.path.basename(self.index_path)[:-3] if self.index_path else
                ("synthetic" if self.gi is not None else None)}

    def info(self):
        corpus = None
        if self.gi is not None:
            y = np.asarray(self.gi.paper_year)
            corpus = {"papers": int(self.gi.n_papers), "knowledge_units": int(self.gi.n_comps),
                      "years": f"{int(y.min())} to {int(y.max())}" if len(y) else None,
                      "fields": self._field_counts(),
                      "splits": {s: int(len(self.gi.papers(s))) for s in ("train", "val", "test")},
                      "full_texts": self.comp_text is not None}
        return {"name": "Sci-JEPA", "version": self.version, "model": self.model_ref(),
                "checkpoint": {"weights": self.weights, "best_epoch": self.summary.get("best_epoch"),
                               "sha256": self.weights_sha256},
                "architecture": {"encoder": self.cfg.encoder if hasattr(self.cfg, "encoder") else None,
                                 "text_encoder": self.raw_cfg.get("embed_model"), "feat_dim": self.cfg.feat_dim,
                                 "context_papers": self.cfg.n_subgraphs if self.jepa.gi is not None else 1},
                "test_metrics": self.summary.get("test"),
                "training_data": {"source": "Scientific-Summaries (LAION, CC-BY-4.0)"},
                "corpus": corpus, "language": ["en"], "tasks": list(T.TASKS),
                "limitations": [
                    "Ranks published findings by fit; it does not check facts.",
                    "Scores are relative within one request, not probabilities of truth.",
                    "English only; trained on English papers, so other languages are not supported.",
                    "The text splitter of /v1/ask and /v1/parse is a cue-word placeholder; check `understood`.",
                    "BM25 is computed over the shortlisted papers, not the whole corpus."],
                "attribution": ATTRIBUTION}

    # ================================================================ corpus helpers
    def _need_index(self):
        if self.gi is None or self.V is None:
            raise NotReady("the corpus index is not loaded on this server (started with --index none)")

    def ku_text(self, c):
        if self.comp_text is not None:
            return self.comp_text[int(c)], False
        s = self.gi.comp_snippet[int(c)]
        return s, len(s) >= getattr(self.cfg, "snippet_chars", 240)

    @staticmethod
    def _is_arxiv(pid):
        return bool(ARXIV_NEW.match(pid) or ARXIV_OLD.match(pid))

    def _field_counts(self):
        if self.domain_names:
            n = np.bincount(np.asarray(self.gi.paper_domain, dtype=np.int64), minlength=len(self.domain_names))
            return {d: int(k) for d, k in zip(self.domain_names, n)}
        return self.field_names

    def public_id(self, p):
        """The arXiv ID, or p_<n> for corpus papers without one (their stored ID is an internal file path)."""
        pid = str(self.gi.paper_ids[int(p)])
        return pid if self._is_arxiv(pid) else f"p_{int(p)}"

    def paper_meta(self, p):
        gi, p = self.gi, int(p)
        pid = self.public_id(p)
        split = {0: "train", 1: "val", 2: "test"}.get(int(gi.split_of[p]), None)
        sub = self.field_names[int(gi.paper_field[p])]
        field = self.domain_names[int(gi.paper_domain[p])] if self.domain_names else sub
        return {"paper_id": pid, "arxiv_id": pid if not pid.startswith("p_") else None, "title": None,
                "year": int(gi.paper_year[p]), "field": field, "subfield": sub, "split": split}

    def paper_index(self, paper_id):
        self._need_index()
        m = re.fullmatch(r"p_(\d+)", paper_id)
        if m:
            p = int(m.group(1))
            if p < self.gi.n_papers and not self._is_arxiv(str(self.gi.paper_ids[p])):
                return p
            raise NotFound(paper_id)
        p = self._pid_of.get(paper_id)
        if p is None:
            p = self._pid_of.get(re.sub(r"v\d+$", "", paper_id))
        if p is None:
            raise NotFound(paper_id)
        return p

    def ku(self, ku_id):
        self._need_index()
        m = re.fullmatch(r"ku_(\d+)", ku_id or "")
        if not m or int(m.group(1)) >= self.gi.n_comps:
            raise NotFound(ku_id)
        c = int(m.group(1))
        return self._ku_out(c)

    def _ku_out(self, c):
        txt, cut = self.ku_text(c)
        return {"ku_id": f"ku_{c}", "type": COMP_TYPES[int(self.gi.comp_type[c])], "text": txt, "truncated": cut,
                "paper": self.paper_meta(int(self.gi.comp_paper[c]))}

    def paper(self, paper_id):
        p = self.paper_index(paper_id)
        kus = [self._ku_out(int(c)) for c in self.gi.paper_comps[p]]
        for k in kus:
            k.pop("paper")
        return {**self.paper_meta(p), "knowledge_units": kus}

    def search_ids(self, q, limit, cursor=0):
        """Papers whose arXiv ID starts with q (the index holds no titles), in ID order."""
        import bisect
        self._need_index()
        A = self._arxiv_sorted
        lo = bisect.bisect_left(A, q) + cursor
        page = [k for k in A[lo:lo + limit + 1] if k.startswith(q)]
        more = len(page) > limit
        return [self.paper_meta(self._pid_of[k]) for k in page[:limit]], (cursor + limit if more else None)

    def paper_record(self, p):
        """A corpus paper as a summary record (section KUs only), for /v1/ask/paper."""
        rec = {"field_subfield": self.field_names[int(self.gi.paper_field[p])]}
        for c in self.gi.paper_comps[p]:
            t = COMP_TYPES[int(self.gi.comp_type[c])]
            if t in SECTION_KUS and SECTION_KUS[t] not in rec:
                rec[SECTION_KUS[t]] = self.ku_text(c)[0]
        return rec

    def parts_of_record(self, rec):
        back = {v: k for k, v in PART_COLUMN.items()}
        parts = {back[k]: v for k, v in rec.items() if k in back}
        if rec.get("research_question_hypothesis"):
            parts["problem"] = (parts.get("problem", "") + " " + rec["research_question_hypothesis"]).strip()
        return parts

    # ================================================================ search
    def field_mask(self, fields):
        """-> (bool tensor over papers or None, warnings)."""
        if not fields:
            return None, []
        if self.domain_names:                            # the spec's 10 domains = the index's canonical domains
            ids = [self.domain_names.index(f) for f in fields if f in self.domain_names]
            warn = [{"code": "field_not_in_corpus", "part": None, "message": f"No corpus papers in field {f!r}."}
                    for f in fields if f not in self.domain_names]
            return (torch.isin(self._domain, torch.tensor(ids)) if ids else None), warn
        names = [n.casefold() for n in self.field_names]
        ids, warn = set(), []
        for f in fields:
            hit = [i for i, n in enumerate(names) if n == f.casefold()]
            if not hit:                                  # e.g. "EESS & Engineering" ~ "Engineering"
                fw = {w for w in re.findall(r"[a-z]{4,}", f.casefold())}
                hit = [i for i, n in enumerate(names) if fw & set(re.findall(r"[a-z]{4,}", n))]
            if not hit:
                warn.append({"code": "field_not_in_corpus", "part": None, "message": f"No corpus papers in field {f!r}."})
            ids.update(hit)
        if not ids:
            return None, warn
        pf = torch.from_numpy(np.asarray(self.gi.paper_field, dtype=np.int64))
        return torch.isin(pf, torch.tensor(sorted(ids))), warn

    def shortlist(self, texts, k=None, filters=None, include=()):
        """Paper indices most similar to the texts (centred mean frozen embedding) -> (papers, query vector, warnings)."""
        self._need_index()
        k = k or self.shortlist_papers
        X = self.jepa.text_encoder.encode(list(texts)).astype(np.float32)
        q = X.mean(0) - self.mu
        q = q / (np.linalg.norm(q) + 1e-8)
        s = self.V @ torch.from_numpy(q)
        mask, warn = None, []
        f = filters or {}
        if f.get("fields"):
            mask, warn = self.field_mask(f["fields"])
        if f.get("year_from") is not None or f.get("year_to") is not None:
            y = self._year
            ym = (y >=(f.get("year_from") or -10 ** 9)) & (y <= (f.get("year_to") or 10 ** 9))
            mask = ym if mask is None else mask & ym
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        top = torch.topk(s, min(k, len(s)))
        papers = [int(p) for p, v in zip(top.indices.tolist(), top.values.tolist()) if v > float("-inf")]
        for p in include:
            if p not in papers:
                papers.append(int(p))
        return papers, q, warn

    @torch.no_grad()
    def target_keys(self, papers):
        from sci_jepa.evaluation.diagnostics import encode_papers
        ids, H = encode_papers(self.jepa.model, self.gi, np.asarray(papers, dtype=np.int64), self.cfg, self.dev, "target")
        return {int(c): k for k, c in enumerate(ids.tolist())}, H

    # ================================================================ the tasks
    def record(self, parts, field=None):
        rec = {"field_subfield": field or "Unknown"}
        for p, col in PART_COLUMN.items():
            if parts.get(p):
                rec[col] = parts[p]
        return rec

    @torch.no_grad()
    def predict(self, rec, hide, task):
        tpl = task if task in ("forward", "inverse", "abduction") else None
        return self.jepa.predict(rec, hide=hide, task=tpl)

    def rank(self, task, parts, top_k=10, scoring="hybrid", filters=None, rec=None, include=(), truth=None):
        """Answers of a ranking task -> (answers, warnings, extra)."""
        self._need_index()
        gi = self.gi
        rec = rec or self.record(parts, (filters or {}).get("fields", [None])[0] if (filters or {}).get("fields") else None)
        targets = T.TASKS["forward" if task == "intervention" else task]["answers"]
        given = [parts[p] for p in T.TASKS["forward" if task == "intervention" else task]["requires"]
                 if isinstance(parts.get(p), str) and parts.get(p)]
        preds = {t: F.normalize(self.predict(rec, t, task), dim=-1) for t in targets}
        papers, q, warn = self.shortlist(given, filters=filters, include=include)
        tids = {COMP_TYPES.index(t) for t in targets}
        pos, H = self.target_keys(papers)
        cands = [int(c) for p in papers for c in gi.paper_comps[p] if int(gi.comp_type[c]) in tids and int(c) in pos]
        if not cands:
            return [], warn + [{"code": "no_candidates", "part": None,
                                "message": "No knowledge units of the answer type in the matching papers."}], {}
        ctype = [COMP_TYPES[int(gi.comp_type[c])] for c in cands]
        K = F.normalize(H[[pos[c] for c in cands]], dim=-1)
        P_ = torch.stack([preds[t] for t in ctype])
        s_jepa = (K * P_).sum(-1).numpy()
        Xc = np.asarray(gi.comp_x[np.asarray(cands)], dtype=np.float32) - self.mu
        s_text = Xc @ q / (np.linalg.norm(Xc, axis=1) + 1e-8)
        texts = [self.ku_text(c) for c in cands]
        s_bm = bm25(" ".join(given), [t for t, _ in texts])
        if scoring == "hybrid":
            score = HYBRID["jepa"] * _minmax(s_jepa) + HYBRID["text"] * _minmax(s_text) + HYBRID["bm25"] * _minmax(s_bm)
        else:
            score = {"jepa": s_jepa, "text": s_text, "bm25": s_bm}[scoring]
        order = sorted(range(len(cands)), key=lambda i: (-score[i], cands[i]))   # ties: stable by KU id
        out = []
        for r, i in enumerate(order[:top_k], 1):
            c = cands[i]
            out.append({"rank": r, "text": texts[i][0], "truncated": texts[i][1], "score": round(float(score[i]), 4),
                        "scores": {"jepa": round(float(s_jepa[i]), 4), "text": round(float(s_text[i]), 4),
                                   "bm25": round(float(s_bm[i]), 4)},
                        "source": {"ku_id": f"ku_{c}", "type": ctype[i], "paper": self.paper_meta(int(gi.comp_paper[c]))}})
        extra = {"candidates": len(cands), "papers_searched": len(papers)}
        if truth is not None:
            rank_of = {cands[i]: r for r, i in enumerate(order, 1)}
            extra["truth"] = [{**self._ku_out(c), "rank": rank_of.get(int(c)), "of": len(cands)} for c in truth]
        if self.comp_text is None:
            warn.append({"code": "snippets_only", "part": None,
                         "message": "Answer texts are the index's short snippets; start the server with --full-texts "
                                    "for complete texts."})
        return out, warn, extra

    @torch.no_grad()
    def consistency(self, claim, evidence, related=5, filters=None):
        r = self.jepa.consistency(claim, evidence)
        out = {"verdict": "supports" if r["supports"] >= r["contradicts"] else "challenges",
               "probabilities": {"supports": round(r["supports"], 4), "challenges": round(r["contradicts"], 4)}}
        warn = []
        if self.gi is not None and self.V is not None and related:
            papers, q, warn = self.shortlist([claim], k=32, filters=filters)
            ev = COMP_TYPES.index("evidence")
            cands = [int(c) for p in papers for c in self.gi.paper_comps[p] if int(self.gi.comp_type[c]) == ev]
            rel = {"supports": [], "challenges": []}
            if cands:
                Xc = np.asarray(self.gi.comp_x[np.asarray(cands)], dtype=np.float32) - self.mu
                sim = Xc @ q / (np.linalg.norm(Xc, axis=1) + 1e-8)
                for i in np.argsort(-sim)[:2 * related]:
                    c = cands[int(i)]
                    txt = self.ku_text(c)[0]
                    rr = self.jepa.consistency(claim, txt)
                    side = "supports" if rr["supports"] >= rr["contradicts"] else "challenges"
                    if len(rel[side]) < related:
                        rel[side].append({"text": txt, "probability": round(max(rr["supports"], rr["contradicts"]), 4),
                                          "source": {"ku_id": f"ku_{c}", "type": "evidence",
                                                     "paper": self.paper_meta(int(self.gi.comp_paper[c]))}})
            out["related_evidence"] = rel
        return out, warn

    @torch.no_grad()
    def _keys_alone(self, texts, ku_type):
        from sci_jepa.data.graph_index import HUB
        from sci_jepa.data.instances import collate_views
        from sci_jepa.data.ku_schema import REL_MEMBER, REL_NONE
        from sci_jepa.models.model import to_device
        X = self.jepa.text_encoder.encode(list(texts))
        views = [{"x": np.stack([np.zeros_like(x), x]).astype(np.float32),
                  "type": np.asarray([HUB, COMP_TYPES.index(ku_type)]), "comp": np.asarray([-1, 0]),
                  "slot": np.zeros(2, np.int64), "deg": np.ones(2, np.int64),
                  "rel": np.asarray([[REL_NONE, REL_MEMBER], [REL_MEMBER, REL_NONE]])} for x in X]
        return self.jepa.model.target(to_device(collate_views(views), self.dev))[:, 1].float().cpu()

    @torch.no_grad()
    def edit_check(self, original, edited, context=None):
        kind = T.edit_kind(original, edited)
        if context:                                      # the evaluation's rule: does the prediction prefer the original?
            rec = self.record(context)
            sc = dict((t, s) for s, t in self.jepa.rank(rec, "result", [original, edited], task="forward"))
            margin, mode = sc[original] - sc[edited], "prediction_prefers_original"
            sep = margin > 0
        else:                                            # how far apart the model puts the two findings
            k = F.normalize(self._keys_alone([original, edited], "result"), dim=-1)
            margin, mode = 1.0 - float(k[0] @ k[1]), "representation_distance"
            sep = margin >= self.edit_threshold
        notes = ["Trained edit kinds: direction, negation, number, substitution. Other kinds are out of scope."]
        if mode == "representation_distance":
            notes.append(f"Without context parts, `margin` is 1 - cosine between the two findings' Sci-JEPA "
                         f"representations; separated means margin >= {self.edit_threshold} (a starting value). "
                         "Send `context` (problem, method, conditions) for the evaluation's rule.")
        return {"edit_kind": kind, "separated": bool(sep), "margin": round(float(margin), 4), "mode": mode,
                "notes": notes}

    @torch.no_grad()
    def _reference_predictions(self, n=16):
        """Predicted results of unrelated corpus papers: the scale of 'switching to an unrelated paper'."""
        if self._ref is None and self.gi is not None:
            rng = np.random.default_rng(0)
            pool = np.asarray(self.gi.papers("val")) if len(self.gi.papers("val")) >= n else np.arange(self.gi.n_papers)
            preds = []
            for p in rng.choice(pool, min(n, len(pool)), replace=False):
                rec = self.paper_record(int(p))
                if any(rec.get(c) for c in ("methodological_details", "procedures_architectures")):
                    preds.append(F.normalize(self.predict(rec, "result", "forward"), dim=-1))
            self._ref = torch.stack(preds) if preds else None
        return self._ref

    @torch.no_grad()
    def intervention(self, parts, change, top_k=10, scoring="hybrid", filters=None):
        after_parts = {**parts, change["part"]: change["text"]}
        pb = F.normalize(self.predict(self.record(parts), "result", "forward"), dim=-1)
        pa = F.normalize(self.predict(self.record(after_parts), "result", "forward"), dim=-1)
        raw = 1.0 - float(pb @ pa)
        ref = self._reference_predictions()
        out = {"shift_raw": round(raw, 4)}
        if ref is not None:
            scale = max(1e-6, 1.0 - float((ref @ pb).mean()))
            out["shift"] = round(min(1.0, max(0.0, raw / scale)), 4)
        else:
            out["shift"] = None
        out["shift_meaning"] = ("0 = the change makes no difference; 1 = as large as switching to an unrelated paper "
                                "(the mean distance to the predicted results of 16 corpus papers)")
        warn = []
        if self.gi is not None and self.V is not None:
            out["before"], w1, _ = self.rank("forward", parts, top_k, scoring, filters)
            out["after"], w2, _ = self.rank("forward", after_parts, top_k, scoring, filters)
            warn = w1 + [w for w in w2 if w not in w1]
        else:
            out["before"] = out["after"] = []
        return out, warn
