"""
Comparison systems for the benchmark (experiment BENCH), evaluated exactly like the trained
models: same instances (same leak-safe and task-aware masking), same splits, same topic-matched
100-candidate pools, same counterfactual variants, same chance levels and the same five
reasoning tasks, so every row sits in one table.

  text        the frozen text encoder of the index (gte-modernbert): query = mean embedding of
              the hidden KU's VISIBLE paper-mates, candidates ranked by cosine (centred: the
              encoder is anisotropic). A trained model is only worth publishing if it beats it.
  bm25        lexical: query = the visible paper-mates' texts, candidates ranked by Okapi BM25
              (k1 1.2, b 0.75; statistics from the candidate bank). Consistency and intervention
              use TF-IDF + SVD (LSA, 256-d) vectors.
  scincl      a scientific-document encoder (malteos/scincl, [CLS], 512 tokens), same protocol as text
  bge         a general-purpose encoder (BAAI/bge-base-en-v1.5, [CLS], 512 tokens), same protocol
  supervised  same-data control: an MLP trained on the TRAIN split's instances (same masking) to map
              the visible paper-mates' mean embedding (+ the hidden KU's type) to the hidden KU's
              embedding (InfoNCE, in-batch negatives; residual, so it starts from the text row).
              No graph, no JEPA, no EMA target: does the JEPA world model add anything over a
              supervised mapping trained on the same data?

Consistency (support vs contradict) is a supervised linear probe on each system's frozen vectors
(logistic regression on [c, e, c*e, |c-e|], train split), the fair counterpart of the models'
trained relation decoder. Text features for bm25 / scincl / bge come from the index's text file.
"""
import math
import os
import time

import numpy as np

from sci_jepa.evaluation.diagnostics import _chance, hard_pool
from sci_jepa.data.graph_index import COMP_TYPES, HUB
from sci_jepa.data.instances import Sampler
from sci_jepa.data.ku_schema import REL_ID, TEMPLATES

NAN = float("nan")
# encoder baselines: name -> (model, pooling, max tokens, prefix)
ENCODERS = {"scincl": ("malteos/scincl", "cls", 512, ""),
            "bge-m3": ("BAAI/bge-m3", "cls", 8192, ""),          # long-context (8,192 tokens)
            "bge": ("BAAI/bge-base-en-v1.5", "cls", 512, ""),
            "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", 256, "")}
LABELS = {"llm": "LLM-Reasoning (open LLM, calibrated log-likelihood)",
          "text": "frozen text encoder (index features, cosine)", "bm25": "BM25 (lexical)",
          "scincl": "SciNCL (scientific encoder, long texts chunked)", "bge-m3": "BGE-M3 (long-context encoder, 8,192 tokens)",
          "bge": "bge-base-en-v1.5 (general encoder, long texts chunked)",
          "minilm": "MiniLM-L6 (small general encoder)", "supervised": "supervised regressor (same data)"}


def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def _split_ids(gi, split):
    return np.concatenate([gi.paper_comps[int(p)] for p in gi.papers(split)]).astype(np.int64)


# Claim-evidence pairs the consistency probe trains on. The full arXiv train split has ~10M; a probe of a few linear
# weights does not need them, and every one costs an encoding in the encoder baselines (21M texts instead of
# ~2.5M) and a row in a ~140 GB feature matrix. A fixed sample (seed 0), the same for every system and seed.
PROBE_TRAIN_PAIRS = 100_000
SUPERVISED_PASS = 2_000_000         # training instances per pass of the supervised regressor (~12 GB of X/Y)
LLM_CONSISTENCY_PAIRS = 3_000       # test pairs the LLM judges (one prompt each); fixed sample, every seed the same


def _sample_pairs(pairs, cap, balance=True):
    """`pairs` if at most `cap`, else a fixed sample (seed 0) in the original order; balance: the rarer label is kept
    whole up to half the cap, so a rare 'contradicts' class is not thinned further."""
    if not cap or len(pairs) <= cap:
        return pairs
    rng = np.random.default_rng(0)
    if not balance:
        return [pairs[i] for i in sorted(rng.choice(len(pairs), cap, replace=False))]
    lab = np.asarray([l for *_, l in pairs])
    rare = int((lab == 1).sum() < (lab == 0).sum())
    i_rare, i_common = np.flatnonzero(lab == rare), np.flatnonzero(lab != rare)
    n_rare = min(len(i_rare), cap // 2)
    pick = np.concatenate([rng.choice(i_rare, n_rare, replace=False), rng.choice(i_common, cap - n_rare, replace=False)])
    return [pairs[i] for i in sorted(pick)]


def _train_pairs(gi):
    return _sample_pairs(_pairs(gi, gi.papers("train")), PROBE_TRAIN_PAIRS)


def _pairs(gi, papers):
    allowed = set(int(p) for p in papers)
    s_rel, c_rel = REL_ID["supported_by"], REL_ID["challenged_by"]
    return [(a, b, int(r == c_rel)) for a, b, r in zip(gi.role_src.tolist(), gi.role_dst.tolist(), gi.role_rel.tolist())
            if r in (s_rel, c_rel) and int(gi.comp_paper[a]) in allowed]


# ---------------------------------------------------------------- systems
class DenseSystem:
    """Cosine retrieval in a vector space. feat(ids) -> (n, d) float32 vectors of KUs, mp_feat(rows)
    -> vectors of minimal-pair variants. Queries are the centred mean of the visible KUs' vectors."""

    def __init__(self, gi, split, feat, mp_feat, probe_mu=None, name="text"):
        self.gi, self.feat, self.mp_feat, self.name = gi, feat, mp_feat, name
        self.ids = _split_ids(gi, split)
        X = feat(self.ids)
        self.mu = X.mean(0, keepdims=True)
        self.Xc = _unit(X - self.mu)
        self.pos_of = {int(c): k for k, c in enumerate(self.ids)}
        self.probe_mu = probe_mu                         # centring of the consistency probe's features

    def dense(self, comp_ids, t):
        """Query vector in the candidate space from the visible KUs (t: hidden KU's type id)."""
        q = self.feat(np.asarray(comp_ids)).mean(0, keepdims=True) - self.mu
        return _unit(q)[0]

    query = dense

    def score(self, q, idx):
        return self.Xc[idx] @ q

    def score_mp(self, q, rows):
        return _unit(self.mp_feat(rows) - self.mu) @ q if len(rows) else np.zeros(0)

    def cand(self, idx):
        return self.Xc[idx]

    def probe_feats(self, ids):
        X = self.feat(np.asarray(ids))
        return X - (self.probe_mu if self.probe_mu is not None else self.mu)


def text_system(gi, split):
    return DenseSystem(gi, split, lambda ids: gi.comp_x[ids].astype(np.float32),
                       lambda rows: gi.mp_x[rows].astype(np.float32), probe_mu=gi.feat_mean(), name="text")


def _needed_ids(gi, split):
    tr = _train_pairs(gi)
    return np.unique(np.concatenate([_split_ids(gi, split), np.asarray([a for a, _, _ in tr] + [b for _, b, _ in tr],
                                                                        dtype=np.int64)]))


def _encode_chunked(emb, texts, max_tok, overlap=32):
    """No truncation: a text longer than the encoder's context is split into overlapping token windows, every
    window is encoded, and the window vectors are averaged (weighted by window length) and re-normalised."""
    tok = emb.m.tokenizer
    win = max(16, max_tok - 4)                                  # room for special tokens
    ids = tok(list(texts), add_special_tokens=False)["input_ids"]
    pieces, owner, weight = [], [], []
    for i, x in enumerate(ids):
        if len(x) <= win:
            pieces.append(texts[i]); owner.append(i); weight.append(max(len(x), 1))
            continue
        for s in range(0, len(x), win - overlap):
            w = x[s:s + win]
            pieces.append(tok.decode(w)); owner.append(i); weight.append(len(w))
            if s + win >= len(x):
                break
    V = np.asarray(emb.encode(pieces), np.float32)
    out = np.zeros((len(texts), V.shape[1]), np.float32); tot = np.zeros(len(texts), np.float32)
    np.add.at(out, owner, V * np.asarray(weight, np.float32)[:, None]); np.add.at(tot, owner, weight)
    out /= np.maximum(tot, 1)[:, None]
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    emb.n_chunked = int(sum(1 for x in ids if len(x) > win))
    return out.astype(np.float16)


def encoder_system(gi, cfg, split, name):
    """Another text encoder over the same KU texts (index text file); vectors cached on disk."""
    model, pooling, max_tok, prefix = ENCODERS[name]
    texts, mp_texts = gi.texts()
    ids = _needed_ids(gi, split)
    cache = os.path.join(cfg.index_dir, "baseline_features", f"{name}_chunked_{len(texts)}_{len(mp_texts)}_{split}_p{PROBE_TRAIN_PAIRS}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        ok = np.array_equal(z["ids"], ids)
        F, M = (z["X"], z["M"]) if ok else (None, None)
    else:
        F = M = None
    if F is None:
        from sci_jepa.config import resolve_device
        from sci_jepa.data.ku_graph import STEmbedder
        emb = STEmbedder(model, resolve_device(cfg.device), max_tok, 64, prefix, bucket=64,
                         dtype="float16" if resolve_device(cfg.device) != "cpu" else "float32")
        if pooling == "cls":                              # SciNCL / bge are [CLS]-pooled
            for mod in emb.m:
                if hasattr(mod, "pooling_mode_cls_token"):
                    mod.pooling_mode_cls_token, mod.pooling_mode_mean_tokens = True, False
        F = _encode_chunked(emb, [texts[int(c)] for c in ids], max_tok)
        M = _encode_chunked(emb, list(mp_texts), max_tok) if len(mp_texts) else np.zeros((0, F.shape[1]), np.float16)
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, ids=ids, X=F, M=M)
        except OSError:                                   # index folder not writable: no cache, still evaluated
            pass
    row = {int(c): k for k, c in enumerate(ids)}
    feat = lambda q: F[[row[int(c)] for c in q]].astype(np.float32)
    return DenseSystem(gi, split, feat, lambda r: M[np.asarray(r, dtype=np.int64)].astype(np.float32), name=name)


class BM25System(DenseSystem):
    """Okapi BM25 for retrieval; LSA (TF-IDF + SVD) vectors for consistency and intervention."""

    def __init__(self, gi, split, k1=1.2, b=0.75):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        texts, mp_texts = gi.texts()
        self.texts, self.mp_texts, self.k1, self.b = texts, mp_texts, k1, b
        ids = _needed_ids(gi, split)
        bank = _split_ids(gi, split)
        self.cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w[\w\-]*\w\b|\b\w\b")
        C = self.cv.fit_transform([texts[int(c)] for c in bank]).tocsr().astype(np.float64)
        n = C.shape[0]
        df = np.bincount(C.indices, minlength=C.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.avgdl = float(C.sum(1).mean())
        self.W = self._weight(C)
        self.W_mp = self._weight(self.cv.transform(list(mp_texts)).tocsr().astype(np.float64)) if len(mp_texts) else None
        tf = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, min_df=1)
        T = tf.fit_transform([texts[int(c)] for c in ids])
        svd = TruncatedSVD(n_components=min(256, max(T.shape[1] - 1, 2), max(T.shape[0] - 1, 2)), random_state=0)
        L = svd.fit_transform(T).astype(np.float32)
        Lmp = svd.transform(tf.transform(list(mp_texts))).astype(np.float32) if len(mp_texts) else np.zeros((0, L.shape[1]), np.float32)
        row = {int(c): k for k, c in enumerate(ids)}
        super().__init__(gi, split, lambda q: L[[row[int(c)] for c in q]], lambda r: Lmp[np.asarray(r, dtype=np.int64)],
                         name="bm25")
        assert np.array_equal(self.ids, bank)

    def _weight(self, C):
        C = C.copy()
        dl = np.asarray(C.sum(1)).ravel()
        rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        denom = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
        C.data = self.idf[C.indices] * C.data * (self.k1 + 1) / (C.data + denom[rows])
        return C

    def query(self, comp_ids, t):
        q = self.cv.transform([" ".join(self.texts[int(c)] for c in comp_ids)]).tocsr()
        q.data[:] = 1.0                                   # each query term once
        return q

    def score(self, q, idx):
        return np.asarray((self.W[idx] @ q.T).todense()).ravel()

    def score_mp(self, q, rows):
        if not len(rows) or self.W_mp is None:
            return np.zeros(0)
        return np.asarray((self.W_mp[np.asarray(rows, dtype=np.int64)] @ q.T).todense()).ravel()


class SupervisedSystem(DenseSystem):
    """Same-data control: MLP(visible mean embedding, hidden type) -> hidden KU embedding."""

    def __init__(self, gi, cfg, split, seed):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sci_jepa.config import resolve_device
        mu = gi.feat_mean()[None]
        super().__init__(gi, split, lambda ids: gi.comp_x[ids].astype(np.float32),
                         lambda rows: gi.mp_x[rows].astype(np.float32), probe_mu=mu[0], name="supervised")
        self.mu = mu                                      # one centring for training and test
        self.Xc = _unit(gi.comp_x[self.ids].astype(np.float32) - mu)
        torch.manual_seed(seed)
        dev = torch.device(resolve_device(cfg.device))
        d, nt = gi.comp_x.shape[1], len(COMP_TYPES)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Sequential(nn.Linear(d + nt, 1024), nn.GELU(), nn.Linear(1024, d))
                nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)   # starts as the text row

            def forward(self, x, t):
                return F.normalize(x + self.f(torch.cat([x, F.one_hot(t, nt).float()], -1)), dim=-1)
        self.net, self.dev, self.nt = Net().to(dev), dev, nt

        def data(split_, n, s):
            smp = Sampler(gi, cfg.override(n_subgraphs=1), split_, s)   # own paper only: the query's input
            smp.train = split_ == "train"
            X, Y, T, left = [], [], [], n
            while left > 0:
                for inst in smp.sample(min(512, left)):
                    c = inst["ctx"]
                    vis = (c["slot"] == 0) & (c["type"] != HUB) & (~inst["is_mask"]) & (c["comp"] >= 0)
                    if vis.any():
                        X.append(gi.comp_x[c["comp"][vis]].astype(np.float32).mean(0))
                        Y.append(gi.comp_x[inst["target_comp"]].astype(np.float32)); T.append(inst["target_type"])
                left -= 512
            to = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
            return to(_unit(np.asarray(X) - mu)), to(_unit(np.asarray(Y) - mu)), to(T, torch.long)
        n_anchor = Sampler(gi, cfg, "train", seed).n_anchors()
        # one pass = every hidden KU (as the JEPA models), but at most SUPERVISED_PASS instances: the X/Y of a pass
        # are built in host RAM (42.5M anchors at 2M papers = ~260 GB, which OOM-killed jobs 38502 / 38539)
        per_pass = min(max(n_anchor, 512), SUPERVISED_PASS)
        Xv, Yv, Tv = data("val", min(max(len(gi.papers("val")) * 22, 512), 20000), seed + 7)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=0.01)
        best, best_state, bad, self.passes = -1.0, None, 0, 0
        for ep in range(int(max(cfg.epochs, 1))):          # same budget as the trained models; early stop
            Xt, Yt, Tt = data("train", per_pass, seed + 100 + ep)
            self.net.train()
            for i in range(0, len(Xt), 256):
                p = self.net(Xt[i:i + 256], Tt[i:i + 256])
                logits = p @ Yt[i:i + 256].T / 0.05
                loss = F.cross_entropy(logits, torch.arange(len(p), device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():                         # validation: in-batch retrieval accuracy
                p = self.net(Xv, Tv)
                acc = float((torch.argmax(p @ Yv.T, 1) == torch.arange(len(p), device=dev)).float().mean())
            self.passes = ep + 1
            if acc > best + 1e-4:
                best, bad = acc, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= max(cfg.patience, 3):
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.val_acc = best

    def dense(self, comp_ids, t):
        import torch
        x = _unit(gi_mean(self, comp_ids) - self.mu)
        with torch.no_grad():
            q = self.net(torch.as_tensor(x, device=self.dev), torch.as_tensor([int(t)], device=self.dev))
        return q[0].float().cpu().numpy()

    query = dense


_LABEL = {"context": "Research context", "question": "Research question / hypothesis", "method": "Method",
          "procedure": "Procedure / setup", "result": "Key result", "interpretation": "Interpretation",
          "limitation": "Limitation", "claim": "Claim", "evidence": "Evidence", "implication": "Implication"}


class LLMSystem:
    """LLM-Reasoning: an open causal language model (cfg.llm_model, Hugging Face) ranks the same candidates
    as every other system. Score of a candidate = mean token log-probability of its text after a prompt
    with the visible parts of the paper, minus the same without the paper (cfg.llm_score="pmi", calibrated:
    removes the model's preference for fluent or common texts) or without that correction ("loglik").
    Consistency: zero-shot, P(" supports") vs P(" contradicts") after claim + evidence, calibrated with a
    content-free prompt. Intervention: does swapping the procedure / method raise the donor result's score
    relative to the paper's own result? Needs a CUDA GPU (skipped elsewhere unless llm_allow_cpu)."""
    rank_whole_bank = False                              # no standard rank over the whole bank (too costly)

    def __init__(self, gi, cfg, split):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sci_jepa.config import resolve_device
        self.gi, self.cfg, self.name = gi, cfg, "llm"
        self.texts, self.mp_texts = gi.texts()
        self.ids = _split_ids(gi, split)
        self.pos_of = {int(c): k for k, c in enumerate(self.ids)}
        dev = resolve_device(cfg.device)
        self.dev = torch.device(dev)
        self.tok = AutoTokenizer.from_pretrained(cfg.llm_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if dev == "cpu" else getattr(torch, cfg.llm_mps_dtype if dev == "mps" else cfg.llm_dtype)
        if dev == "mps":                                  # unified memory: keep candidate batches small
            self.cfg = cfg = cfg.override(llm_batch=min(cfg.llm_batch, 4))
        self.lm = AutoModelForCausalLM.from_pretrained(cfg.llm_model, torch_dtype=dtype).to(self.dev).eval()
        mc = self.lm.config                               # prompt + candidate must fit the model's context
        max_pos = getattr(mc, "max_position_embeddings", None) or getattr(mc, "n_positions", None) or 4096
        self.ctx_tokens = max(64, min(cfg.llm_ctx_tokens, max_pos - cfg.llm_cand_tokens - 64))
        self._null = {}
        print(f"[llm] {cfg.llm_model} on {dev} ({cfg.llm_score} scoring)", flush=True)

    # ---- progress
    def progress(self, label, done, total, t0, every=0.1):
        """Print '[llm] <label>: done/total, elapsed, ETA' about every 10% of a task (and at its end)."""
        import time
        if total <= 0:
            return
        now = time.time()                                 # infrastructure monitor: about every 5 s, never fatal
        if now - getattr(self, "_prog_t", 0) >= 5 or done == total:
            self._prog_t = now
            try:
                import torch
                from sci_jepa.infra.monitor import write_progress
                el = now - t0
                mem = ({"device": torch.cuda.get_device_name(self.dev),
                        "gpu_mem_alloc_gb": round(torch.cuda.memory_allocated(self.dev) / 1e9, 2),
                        "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated(self.dev) / 1e9, 2)}
                       if self.dev.type == "cuda" else {})
                write_progress(self.cfg.out_dir, f"LLM@{self.cfg.llm_model.split('/')[-1]}", status="llm",
                               task=label, done=int(done), total=int(total), elapsed_s=round(el),
                               queries_per_s=round(done / max(el, 1e-9), 3),
                               eta_max_s=round(el / max(done, 1) * (total - done)), **mem)
            except Exception:                             # noqa: BLE001
                pass
        step = max(1, int(total * every))
        if done == total or done % step == 0:
            el = time.time() - t0
            eta = el / max(done, 1) * (total - done)
            print(f"[llm] {label}: {done}/{total} ({done / total:.0%}), {el / 60:.1f} min"
                  + (f", ~{eta / 60:.1f} min left" if done < total else " - done"), flush=True)

    # ---- scoring
    def _free(self):
        import torch
        if self.dev.type == "mps":
            torch.mps.empty_cache()
        elif self.dev.type == "cuda":
            torch.cuda.empty_cache()

    def close(self):
        """Release the language model's memory (called when the arm is finished)."""
        import gc
        self.lm = None
        gc.collect()
        self._free()

    def _cut(self, text, n):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return self.tok.decode(ids[-n:]) if len(ids) > n else text

    def _lp(self, prompts, conts):
        """mean log-probability of each continuation after its prompt (batched)."""
        import torch
        out = []
        B = max(1, self.cfg.llm_batch)
        for s in range(0, len(prompts), B):
            P = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts[s:s + B]]
            C = [self.tok(" " + c.strip(), add_special_tokens=False)["input_ids"][:self.cfg.llm_cand_tokens]
                 for c in conts[s:s + B]]
            seqs = [p + c for p, c in zip(P, C)]
            L = max(len(x) for x in seqs)
            pad = self.tok.pad_token_id
            inp = torch.full((len(seqs), L), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, x in enumerate(seqs):
                inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            with torch.no_grad():
                logits = self.lm(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev)).logits.float()
            lps = torch.log_softmax(logits[:, :-1], -1)
            for i, (p, c) in enumerate(zip(P, C)):
                if not c:
                    out.append(-1e9); continue
                pos = torch.arange(len(p) - 1, len(p) - 1 + len(c), device=self.dev)
                tgt = torch.tensor(c, device=self.dev)
                out.append(float(lps[i, pos, tgt].mean()))
        return np.asarray(out)

    def _lp_shared(self, prompt, conts):
        """mean log-probability of each continuation after ONE shared prompt: the prompt is encoded once and its
        key / value cache reused for every batch of continuations (~20x fewer tokens than re-reading it)."""
        import copy
        import torch
        P = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(P) < 2:
            return self._lp([prompt] * len(conts), conts)
        bk = self.cfg.llm_shape_bucket
        if bk and len(P) > bk:                             # few distinct shapes: Metal caches buffers per shape
            P = P[len(P) % bk:]                            # (drops < bk of the OLDEST context tokens)
        with torch.no_grad():
            po = self.lm(input_ids=torch.tensor([P], device=self.dev), use_cache=True, logits_to_keep=1)
        cache, last = po.past_key_values, torch.log_softmax(po.logits[0, -1].float(), -1)
        del po
        out, B = [], max(1, self.cfg.llm_batch)
        for s in range(0, len(conts), B):
            C = [self.tok(" " + c.strip(), add_special_tokens=False)["input_ids"][:self.cfg.llm_cand_tokens] or [self.tok.eos_token_id]
                 for c in conts[s:s + B]]
            L = max(len(c) for c in C)
            if bk:                                         # candidate length and batch size padded to fixed sizes
                L = min(-(-L // 32) * 32, max(32, -(-self.cfg.llm_cand_tokens // 32) * 32))
                C_real = len(C)
                C = C + [C[0]] * (B - len(C))
            else:
                C_real = len(C)
            n = len(C)
            inp = torch.full((n, L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((n, len(P) + L), dtype=torch.long); att[:, :len(P)] = 1
            for i, c in enumerate(C):
                inp[i, :len(c)] = torch.tensor(c); att[i, len(P):len(P) + len(c)] = 1
            kv = copy.deepcopy(cache)
            kv.batch_repeat_interleave(n)
            with torch.no_grad():
                lg = self.lm(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), past_key_values=kv,
                             use_cache=False).logits
                for i, c in enumerate(C[:C_real]):         # one row at a time: no (batch x length x vocab) copies
                    tgt = torch.tensor(c, device=self.dev)
                    first = last[tgt[0]]                   # first token is predicted by the prompt's last position
                    if len(c) > 1:
                        row = lg[i, :len(c) - 1].float()
                        rest = row.gather(-1, tgt[1:, None])[:, 0] - torch.logsumexp(row, -1)
                    else:
                        rest = first.new_zeros(0)
                    out.append(float(torch.cat([first.view(1), rest]).mean()))
            del lg, kv
            self._free()
        return np.asarray(out)

    def query(self, comp_ids, t):
        gi = self.gi
        by = {}
        for c in comp_ids:
            by.setdefault(COMP_TYPES[gi.comp_type[c]], []).append(self.texts[int(c)])
        parts = [f"{_LABEL.get(k, k)}: {' '.join(v)}" for k, v in by.items()]
        head = self._cut("Parts of a scientific paper.\n\n" + "\n".join(parts), self.ctx_tokens)
        target = _LABEL.get(COMP_TYPES[int(t)], COMP_TYPES[int(t)])
        return (head + f"\n\nWhat is the paper's {target.lower()}?\n{target}:", f"{target}:", int(t))

    dense = query

    def _score_texts(self, q, keys, texts):
        prompt, null, t = q
        s = self._lp_shared(prompt, texts)
        if self.cfg.llm_score == "pmi":
            miss = [i for i, k in enumerate(keys) if (k, t) not in self._null]
            if miss:
                v = self._lp_shared(null, [texts[i] for i in miss])
                for i, x in zip(miss, v):
                    self._null[(keys[i], t)] = x
            s = s - np.asarray([self._null[(k, t)] for k in keys])
        return s

    def score(self, q, idx):
        keys = [("c", int(self.ids[i])) for i in idx]
        return self._score_texts(q, keys, [self.texts[int(self.ids[i])] for i in idx])

    def score_mp(self, q, rows):
        if not len(rows):
            return np.zeros(0)
        return self._score_texts(q, [("m", int(r)) for r in rows], [self.mp_texts[int(r)] for r in rows])

    # ---- consistency (zero-shot) and intervention
    def consistency(self, split):
        pairs = _sample_pairs(_pairs(self.gi, self.gi.papers(split)), LLM_CONSISTENCY_PAIRS, balance=False)
        if not pairs:
            return {"bal_acc": None, "n": 0}
        n = max(32, min(300, self.ctx_tokens // 3))
        fmt = lambda c, e: (f"Claim: {self._cut(c, n)}\nEvidence: {self._cut(e, n)}\n"
                            f"Does the evidence support or contradict the claim? Answer:")
        null = fmt("N/A", "N/A")
        base = self._lp([null, null], ["supports", "contradicts"])
        import time
        t0, sc = time.time(), []
        for k, (a, b, _) in enumerate(pairs):              # prompt encoded once per pair
            sc.append(self._lp_shared(fmt(self.texts[a], self.texts[b]), ["supports", "contradicts"]))
            self.progress("consistency (claim-evidence pairs)", k + 1, len(pairs), t0)
        sc = np.asarray(sc)
        s, c = sc[:, 0] - base[0], sc[:, 1] - base[1]
        y = np.asarray([l for *_, l in pairs]); yhat = (c > s).astype(int)
        rec = [float((yhat[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
        return {"bal_acc": float(np.mean(rec)), "acc": float((yhat == y).mean()), "chance": 0.5, "n": int(len(y)),
                "n_supports": int((y == 0).sum()), "n_contradicts": int((y == 1).sum()),
                "recall_supports": rec[0] if (y == 0).any() else None,
                "recall_contradicts": rec[-1] if (y == 1).any() else None,
                "q": [[int(a), int(b), int(self.gi.paper_year[int(self.gi.comp_paper[p])]),
                       _pdom(self.gi, int(self.gi.comp_paper[p]))] for a, b, (p, _, _) in zip(y, yhat, pairs)]}

    def intervention(self, items):
        """items: [(visible ids, visible ids with the swapped section, own result comp, donor result comp,
        year)] -> metrics as in the models' intervention test, on the LLM's calibrated scores."""
        import time
        t_res = COMP_TYPES.index("result")
        toward, shifts, own1, yrs, t0 = [], [], [], [], time.time()
        for k_, (vis, vis2, own, donor, yr, dm) in enumerate(items):
            self.progress("intervention (swapped condition / method)", k_ + 1, len(items), t0)
            q1, q2 = self.query(vis, t_res), self.query(vis2, t_res)
            keys = [("c", int(own)), ("c", int(donor))]
            tx = [self.texts[int(own)], self.texts[int(donor)]]
            s1, s2 = self._score_texts(q1, keys, tx), self._score_texts(q2, keys, tx)
            toward.append(float((s2[1] - s1[1]) - (s2[0] - s1[0]) > 0))
            shifts.append(float(np.abs(s2 - s1).mean())); own1.append(float(s1[0])); yrs.append((yr, dm))
        if len(toward) < 2:
            return {"rel_shift": None, "toward_donor": None, "n": len(toward)}
        base = float(np.mean(np.abs(np.subtract.outer(own1, own1))[~np.eye(len(own1), dtype=bool)]))
        return {"shift": float(np.mean(shifts)), "rel_shift": float(np.mean(shifts) / max(base, 1e-9)),
                "toward_donor": float(np.mean(toward)), "away_from_own": None, "chance": 0.5, "n": len(toward),
                "q": [[t, sh, y, d] for t, sh, (y, d) in zip(toward, shifts, yrs)], "base_gap": base}


def _pdom(gi, p):
    return int(gi.paper_domain[p]) if hasattr(gi, "paper_domain") else -1


def gi_mean(sys, comp_ids):
    return sys.gi.comp_x[np.asarray(comp_ids)].astype(np.float32).mean(0, keepdims=True)


def _isolated_child(cfg_dict, split, seed, out_path):
    """Child process of run_isolated: rebuild the config, load the cached index, evaluate, write JSON."""
    import json
    from sci_jepa.config import PaperAConfig
    from sci_jepa.data.graph_index import get_index
    base = PaperAConfig()
    cfg = base.override(**{k: v for k, v in cfg_dict.items() if hasattr(base, k) and k != "embedder"})
    gi = get_index(cfg)
    m = text_baseline(cfg, gi, split, seed)
    with open(out_path, "w") as f:
        json.dump(m, f, default=str)


def run_isolated(cfg, split="test", seed=1234):
    """Evaluate a comparison system in a fresh Python process (used for LLM-Reasoning): its GPU memory and the
    Metal / CUDA caches of the parent never mix, and everything is released when the child exits. A plain
    subprocess (not multiprocessing), so it works the same from scripts, `python -m` and notebooks."""
    import json
    import subprocess
    import sys
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../src (sci_jepa importable)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg.as_dict(), f, default=str); cfg_path = f.name
    out = cfg_path[:-5] + "_out.json"
    env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import json, sys; from sci_jepa.evaluation.baselines import _isolated_child; "
            "_isolated_child(json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), sys.argv[4])")
    r = subprocess.run([sys.executable, "-c", code, cfg_path, split, str(seed), out], env=env, cwd=os.getcwd())   # same cwd as the parent: relative data paths resolve the same way
    os.unlink(cfg_path)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"isolated {cfg.baseline} evaluation failed (exit code {r.returncode})")
    with open(out) as f:
        m = json.load(f)
    os.unlink(out)
    return m


def make_system(cfg, gi, split, seed):
    name = cfg.baseline
    if name == "text":
        return text_system(gi, split)
    if name == "bm25":
        return BM25System(gi, split)
    if name == "supervised":
        return SupervisedSystem(gi, cfg, split, seed)
    if name in ENCODERS:
        return encoder_system(gi, cfg, split, name)
    if name == "llm":
        return LLMSystem(gi, cfg, split)
    raise KeyError(f"unknown baseline {name!r}; choose from {sorted(LABELS)}")


# ---------------------------------------------------------------- evaluation (shared by every system)
def _visible(inst):
    c = inst["ctx"]
    vis = (c["slot"] == 0) & (c["type"] != HUB) & (~inst["is_mask"]) & (c["comp"] >= 0)
    return c["comp"][vis]


def _rank_instances(gi, cfg, sampler, sys, n, rng, variants=False, label="retrieval"):
    import time
    ctype, cpaper = gi.comp_type[sys.ids], gi.comp_paper[sys.ids]
    rec, left, t0, done = [], n, time.time(), 0
    show = getattr(sys, "progress", None)
    while left > 0:
        insts = sampler.sample(min(cfg.batch_size, left))
        left -= cfg.batch_size
        for inst in insts:
            g = sys.pos_of.get(int(inst["target_comp"]))
            vis = _visible(inst)
            if g is None or not len(vis):
                continue
            t = int(inst["target_type"]); same = np.nonzero(ctype == t)[0]
            q = sys.query(vis, t)
            if getattr(sys, "rank_whole_bank", True):
                s_all = sys.score(q, same)
                std = 1 + int((s_all > s_all[np.nonzero(same == g)[0][0]]).sum())
            else:                                        # LLM: candidate pools only
                std = None
            pool = [g] + hard_pool(gi, int(inst["papers"][0]), same, cpaper, cfg.balanced_k, rng, mode=cfg.pool_mode)
            s = sys.score(q, pool)
            var = gi.mp_of.get(int(inst["target_comp"]), []) if (variants and cfg.minimal_pairs) else []
            sv = sys.score_mp(q, var) if var else np.zeros(0)
            done += 1
            if show:
                show(label, min(done, n), n, t0)
            rec.append({"type": COMP_TYPES[t], "rank_bal": 1 + int((s[1:] > s[0]).sum()) + int((sv > s[0]).sum()),
                        "pool": len(pool) + len(var), "rank_std": std, "bank": len(same),
                        "mp_pair": list((s[0] > sv).astype(float)) if var else None,
                        "mp_all": float((s[0] > sv).all()) if var else None,
                        "mp_kind": [int(gi.mp_kind[r]) for r in var] if var else None,
                        "year": int(gi.paper_year[int(inst["papers"][0])]),
                        "dom": int(gi.paper_domain[int(inst["papers"][0])]) if hasattr(gi, "paper_domain") else -1})
    return rec


def _agg(rec):
    if not rec:
        return {"MRR": NAN, "Hits@1": NAN, "Hits@10": NAN, "chance_MRR": NAN, "n": 0}
    r = np.asarray([x["rank_bal"] for x in rec], float)
    return {"MRR": float((1 / r).mean()), "Hits@1": float((r <= 1).mean()), "Hits@10": float((r <= 10).mean()),
            "chance_MRR": float(np.mean([_chance(x["pool"]) for x in rec])), "mean_pool": float(np.mean([x["pool"] for x in rec])),
            "n": len(rec)}


def _consistency_unsup(gi, split):
    """Label-free rule on frozen text: cosine(claim, evidence) below the median -> 'contradicts'."""
    pairs = _pairs(gi, gi.papers(split))
    if not pairs:
        return {"bal_acc": None, "n": 0}
    X = gi.comp_x                                       # fp16; rows are cast when used (no full copy)
    mu = X[[a for a, _, _ in pairs] + [b for _, b, _ in pairs]].astype(np.float32).mean(0)
    def cos(a, b):
        u, v = X[a].astype(np.float32) - mu, X[b].astype(np.float32) - mu
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))
    c = np.asarray([cos(a, b) for a, b, _ in pairs]); y = np.asarray([l for _, _, l in pairs])
    yhat = (c < np.median(c)).astype(int)
    rec = [float((yhat[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
    return {"bal_acc": float(np.mean(rec)), "acc": float((yhat == y).mean()), "chance": 0.5, "n": int(len(y)),
            "n_supports": int((y == 0).sum()), "n_contradicts": int((y == 1).sum()),
            "recall_supports": rec[0] if (y == 0).any() else None,
            "recall_contradicts": rec[-1] if (y == 1).any() else None}


def _consistency_probe(gi, split, sys):
    """Support vs contradict from the system's frozen vectors: a supervised linear probe trained on
    the TRAIN split's claim-evidence pairs (class-balanced), scored on `split`."""
    from sklearn.linear_model import LogisticRegression
    tr, te = _train_pairs(gi), _pairs(gi, gi.papers(split))
    if not tr or not te or len({l for *_, l in tr}) < 2:
        return {"bal_acc": None, "n": 0}
    def feats(P):
        c = _unit(sys.probe_feats([a for a, _, _ in P])); e = _unit(sys.probe_feats([b for _, b, _ in P]))
        return np.concatenate([c, e, c * e, np.abs(c - e)], 1)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(feats(tr), np.asarray([l for *_, l in tr]))
    y = np.asarray([l for *_, l in te]); yhat = clf.predict(feats(te))
    rec = [float((yhat[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
    return {"bal_acc": float(np.mean(rec)), "acc": float((yhat == y).mean()), "chance": 0.5, "n": int(len(y)),
            "n_supports": int((y == 0).sum()), "n_contradicts": int((y == 1).sum()),
            "recall_supports": rec[0] if (y == 0).any() else None,
            "recall_contradicts": rec[-1] if (y == 1).any() else None, "n_train": len(tr),
            "q": [[int(a), int(b), int(gi.paper_year[int(gi.comp_paper[c])]), _pdom(gi, int(gi.comp_paper[c]))]
                  for a, b, (c, _, _) in zip(y, yhat, te)]}


def _intervention(gi, cfg, split, sys, rng, seed):
    """The models' intervention test: forward query (problem, method, procedure visible), swap the
    procedure (condition) or method for a donor paper's, and check whether the query moves toward the
    donor's RESULT in the system's vector space."""
    t_res = COMP_TYPES.index("result")
    res_of = {int(gi.comp_paper[c]): k for k, c in enumerate(sys.ids) if gi.comp_type[c] == t_res}
    allowed = set(int(p) for p in gi.papers(split))
    llm = hasattr(sys, "intervention")
    centre = sys.cand(list(res_of.values())).mean(0) if res_of and not llm else 0
    out = {}
    for name, swap in (("condition", "procedure"), ("method", "method")):
        t_sw = COMP_TYPES.index(swap)
        s = Sampler(gi, cfg.override(target_types=("result",)), split, seed + 1)
        s.given = set(TEMPLATES["forward"][0])
        q1s, q2s, toward, iyr, idm, items = [], [], [], [], [], []
        for inst in s.sample(max(cfg.task_eval_instances // 2, 16)):
            p0 = int(inst["papers"][0])
            vis = list(_visible(inst))
            k = [i for i, c in enumerate(vis) if gi.comp_type[c] == t_sw]
            pool = [q for q in gi.subfield_papers[int(gi.paper_subfield[p0])] if q != p0 and q in allowed and q in res_of
                    and q not in inst["papers"]] or [q for q in res_of if q != p0 and q in allowed and q not in inst["papers"]]
            if not k or not pool:
                continue
            d = int(rng.choice(pool))
            dc = [int(x) for x in gi.paper_comps[d] if gi.comp_type[x] == t_sw]
            if not dc:
                continue
            vis2 = list(vis); vis2[k[0]] = dc[0]
            if llm:
                if p0 in res_of:
                    items.append((vis, vis2, int(sys.ids[res_of[p0]]), int(sys.ids[res_of[d]]), int(gi.paper_year[p0]),
                                  _pdom(gi, p0)))
                continue
            q1, q2 = sys.dense(vis, t_res), sys.dense(vis2, t_res)
            q1s.append(q1); q2s.append(q2)
            toward.append(float((q2 - q1) @ (sys.cand([res_of[d]])[0] - centre) > 0))
            iyr.append(int(gi.paper_year[p0])); idm.append(_pdom(gi, p0))
        if llm:
            out[name] = {"swap": swap, **sys.intervention(items)}
            continue
        if len(q1s) > 1:
            Q1, Q2 = np.stack(q1s), np.stack(q2s)
            shift = 1 - (Q1 * Q2).sum(1)
            base = 1 - (Q1 @ Q1.T)[~np.eye(len(Q1), dtype=bool)].mean()
            out[name] = {"swap": swap, "shift": float(shift.mean()), "rel_shift": float(shift.mean() / max(base, 1e-12)),
                         "toward_donor": float(np.mean(toward)), "away_from_own": None, "chance": 0.5, "n": len(q1s),
                         "q": [[float(t), float(sh), int(y), int(d)] for t, sh, y, d in zip(toward, shift, iyr, idm)],
                         "base_gap": float(base)}
        else:
            out[name] = {"swap": swap, "rel_shift": None, "toward_donor": None, "n": len(q1s)}
    out["note"] = f"{sys.name}: the swapped section changes the query directly"
    return out


def _by_kind(rr):
    """Share of counterfactual variants ranked below the true KU, per kind of edit."""
    from sci_jepa.data.ku_schema import MP_KINDS
    out = {}
    for x in rr:
        for k, v in zip(x.get("mp_kind") or [], x.get("mp_pair") or []):
            out.setdefault(MP_KINDS[k], []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def text_baseline(cfg, gi, split="test", seed=1234):
    """Evaluate the comparison system cfg.baseline ("text", "bm25", "scincl", "bge", "minilm",
    "supervised"). The name is kept for backward compatibility."""
    t0 = time.time()
    if cfg.minimal_pairs and cfg.mp_filter:            # the same filtered counterfactuals as the models
        from sci_jepa.evaluation.validity import apply_filter
        apply_filter(gi, cfg)
    sys = make_system(cfg, gi, split, seed)
    rng = np.random.default_rng(seed)
    if isinstance(sys, LLMSystem):                    # LLM scoring is costly: smaller query budgets
        cfg = cfg.override(eval_instances=min(cfg.eval_instances, cfg.llm_eval_instances),
                           task_eval_instances=min(cfg.task_eval_instances, cfg.llm_eval_instances))
    rec = _rank_instances(gi, cfg, Sampler(gi, cfg, split, seed), sys, cfg.eval_instances, rng, label="masked-KU retrieval")
    rb = _agg(rec)
    rs = np.asarray([x["rank_std"] for x in rec if x["rank_std"] is not None], float)
    rs = rs if len(rs) else np.asarray([NAN])
    per_type = {}
    for t in sorted({x["type"] for x in rec}):
        g = [x for x in rec if x["type"] == t]
        a = _agg(g)
        per_type[t] = {"n": a["n"], "MRR_bal": a["MRR"], "chance_bal": a["chance_MRR"], "Hits1_bal": a["Hits@1"],
                       "MRR_std": float(np.mean([1 / x["rank_std"] for x in g if x["rank_std"]] or [NAN])),
                       "top1_same_paper": NAN}
    tasks = {}
    for key, label, query in (("forward", "Forward prediction", "mask Result | given Problem, Method, Material/Condition"),
                              ("inverse", "Experiment design (inverse)", "mask Method / Material | given Problem, Result"),
                              ("abduction", "Abduction", "mask Claim / Hypothesis | given Results, Conditions")):
        given, targets = TEMPLATES[key]
        by, allr = {}, []
        for tg in targets:
            s = Sampler(gi, cfg.override(target_types=(tg,)), split, seed)
            s.given = set(given)
            rr = _rank_instances(gi, cfg, s, sys, max(cfg.task_eval_instances // len(targets), 16), rng, variants=True,
                                 label=f"{label.lower()} ({tg})")
            allr += rr
            a = _agg(rr)
            mpp = [v for x in rr if x["mp_pair"] for v in x["mp_pair"]]; mpa = [x["mp_all"] for x in rr if x["mp_all"] is not None]
            if a["n"]:
                by[tg] = {**a, "chance": a["chance_MRR"], "ratio": a["MRR"] / max(a["chance_MRR"], 1e-9),
                          "mp_pair": float(np.mean(mpp)) if mpp else None, "mp_all": float(np.mean(mpa)) if mpa else None,
                          "mp_n": len(mpa)}
        n = sum(v["n"] for v in by.values())
        avg = (lambda k: float(sum(v[k] * v["n"] for v in by.values()) / n)) if n else (lambda k: NAN)
        mrr, ch = avg("MRR"), avg("chance")
        mp = [(v["mp_pair"], v["mp_all"], v["mp_n"]) for v in by.values() if v.get("mp_pair") is not None]
        mpn = sum(x[2] for x in mp)
        tasks[key] = {"MRR": mrr, "chance": ch, "ratio": mrr / ch if n and ch else NAN, "Hits@1": avg("Hits@1"),
                      "Hits@10": avg("Hits@10"), "n": n, "by_target": by, "label": label, "query": query,
                      "mp_pair": float(sum(a * w for a, _, w in mp) / mpn) if mpn else None,
                      "mp_all": float(sum(b * w for _, b, w in mp) / mpn) if mpn else None, "mp_n": mpn,
                      "mp_by_kind": _by_kind(allr),
                      "q": [[1.0 / x["rank_bal"], float(_chance(x["pool"])), x["year"], x["dom"]] for x in allr]}
    cons = sys.consistency(split) if hasattr(sys, "consistency") else _consistency_probe(gi, split, sys)
    tasks["consistency"] = {**cons, "label": "Consistency checking",
                            "query": ("zero-shot: P(supports) vs P(contradicts) after claim + evidence, calibrated"
                                      if hasattr(sys, "consistency") else
                                      "supervised probe on the system's frozen vectors (logistic regression on "
                                      "[c, e, c*e, |c-e|], trained on the train split): the fair counterpart of the "
                                      "models' trained relation decoder"),
                            "unsupervised": _consistency_unsup(gi, split)}
    tasks["intervention"] = {**_intervention(gi, cfg, split, sys, rng, seed), "label": "Intervention conditioning",
                             "query": "same procedure / method swap as for the models, in the system's vector space"}
    from sci_jepa.evaluation.tasks import add_domains, add_periods
    add_periods(tasks, cfg.encoder_cutoff_year)
    add_domains(tasks, gi, cfg.min_domain_queries, cfg.min_domains)
    if hasattr(sys, "close"):
        sys.close()
    out = {"baseline": LABELS.get(cfg.baseline, cfg.baseline) + (f" ({cfg.embed_model})" if cfg.baseline == "text" else ""),
           "retrieval_balanced": rb,
           "retrieval_standard": {"MRR": float((1 / rs).mean()), "Hits@1": float((rs <= 1).mean()),
                                  "Hits@10": float((rs <= 10).mean())},
           "chance_MRR_standard": float(np.mean([(math.log(x["bank"]) + 0.5772) / x["bank"] for x in rec])) if rec else NAN,
           "per_type": per_type, "tasks": tasks,
           "target_space": {"chi": NAN, "chi_raw": NAN, "eff_rank": NAN}, "context_space": {"chi": NAN, "eff_rank": NAN},
           "partition_probe": {}, "relation_sensitivity": NAN, "inverse_acc": NAN,
           "audit": {"flagged_frac": NAN}, "shortcut_margin_MRR": {"mean": NAN},
           "history": [], "epoch_results": [], "n_eval": len(rec),
           "train_sec": round(time.time() - t0, 1), "params_trainable": 0, "samples_seen": 0}
    if isinstance(sys, SupervisedSystem):
        out["params_trainable"] = int(sum(p.numel() for p in sys.net.parameters()))
        out["supervised"] = {"passes": sys.passes, "val_in_batch_acc": sys.val_acc}
    return out
