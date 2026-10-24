"""
Build the Paper A component graph straight from the normalized parquet shards
(Scientific-Summaries-Data/normalized/<subset>/*.parquet). Replaces the old
raw-JSON builder (core/data_utils/paper_graph.py), which kept only five KU
types and dropped procedures whenever methodological details existed.

Streaming: shards are read a few columns at a time (never text_raw /
text_sanitized), KUs are embedded in chunks, and only float16 embeddings are
kept. Output is the blob GraphIndex consumes.

    python -m sci_jepa.data.ku_graph --set max_papers=60000     # build + cache D0
"""
import glob
import hashlib
import os
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

from sci_jepa.data import ku_schema as S

INDEX_VERSION = 6          # bump when the blob format changes (2: KU text snippets; 6: canonical domains +
                           # domain-balanced corpus; 5: experiment-design
                           # counterfactuals + full KU texts in a side file for the text baselines)

READ_COLS = (list(S.SECTION_KUS.values()) + list(S.OPTIONAL_KUS.values())
             + ["claims", "paper_id", "subset", "source_year", "oa_year", "field_subfield",
                "type_of_paper", "openalex_id", "oa_referenced_works"])


# ---------------------------------------------------------------- embedders
class HashEmbedder:
    """Deterministic bag-of-words random projection. FOR SMOKE TESTS ONLY: it
    has no semantics beyond shared words."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = zlib.crc32(w.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-8)


class STEmbedder:
    def __init__(self, name, device, max_tokens, batch=32, prefix="", bucket=512, dtype="float32"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=device)
        if max_tokens:
            self.m.max_seq_length = min(max_tokens, self.m.max_seq_length or max_tokens)
        if dtype != "float32" and str(device) != "cpu":   # fp16 on GPU: 1.9x faster, cosine to fp32 >= 0.99999
            self.m.to(getattr(torch, dtype))
        self.dim = self.m.get_sentence_embedding_dimension()
        self.batch, self.prefix, self.bucket = batch, prefix, bucket
        self.n_seen = self.n_truncated = 0
        print(f"[ku_graph] text encoder {name}: {self.dim}-d, context {self.m.max_seq_length} tokens, {device}, "
              f"{dtype}, padding to multiples of {bucket}")

    def encode(self, texts, token_budget=8192, bucket=None):
        """Batches by a token budget (many short KUs, few long papers per pass) and pads
        to a multiple of `bucket`: on Apple GPUs every distinct sequence length keeps its
        own cached buffers, and thousands of lengths ran an 88 GB GPU out of memory.
        Texts are sorted by length, so 64 gives only 32 distinct lengths (at 2,048 tokens),
        identical vectors and 4.3x the speed of 512 on short KUs (median 36 tokens)."""
        import torch
        bucket = bucket or self.bucket
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        tok, L = self.m.tokenizer, self.m.max_seq_length
        n_tok = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
        self.n_seen += len(texts)
        self.n_truncated += int((n_tok > L).sum())
        order = np.argsort(-n_tok)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        dev = self.m.device
        i, t0, shown = 0, time.time(), 0
        while i < len(order):
            if len(texts) >= 500 and i >= shown:     # visible progress for long encodes
                el = time.time() - t0
                eta = f", about {el / i * (len(texts) - i) / 60:.0f} min left" if i else ""
                print(f"[ku_graph]   encoded {i} / {len(texts)} texts ({el / 60:.1f} min{eta})", flush=True)
                shown = i + max(len(texts) // 10, 1)
            padded = min(int(np.ceil(min(n_tok[order[i]], L) / bucket) * bucket), L)
            idx = order[i:i + max(1, min(self.batch, token_budget // padded))]
            f = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=L,
                    pad_to_multiple_of=bucket, return_tensors="pt")
            with torch.no_grad():
                z = self.m.forward({k: v.to(dev) for k, v in f.items()})["sentence_embedding"]
            out[idx] = z.float().cpu().numpy()
            i += len(idx)
            if dev.type == "mps":
                torch.mps.empty_cache()
        return out


class ResumableEmbedder:
    """Saves the vectors of every chunk of texts under `root`, named by a hash of the chunk's
    texts. A build that stops (sleep, crash, Ctrl-C) is rerun with the same settings and skips
    every chunk already encoded; get_index deletes the folder once the index is saved."""

    def __init__(self, emb, root, chunk=8192):
        self.emb, self.root, self.chunk, self.reused = emb, root, chunk, 0
        os.makedirs(root, exist_ok=True)

    def __getattr__(self, k):
        return getattr(self.emb, k)

    def encode(self, texts):
        out = []
        for s in range(0, len(texts), self.chunk):
            part = list(texts[s:s + self.chunk])
            f = os.path.join(self.root, hashlib.sha1("\x1e".join(part).encode()).hexdigest()[:20] + ".npy")
            if os.path.exists(f):
                out.append(np.load(f)); self.reused += len(part)
                continue
            X = np.asarray(self.emb.encode(part), dtype=np.float16)
            np.save(f[:-4] + ".tmp.npy", X)
            os.replace(f[:-4] + ".tmp.npy", f)             # complete files only
            out.append(X)
        return np.concatenate(out) if out else np.zeros((0, self.emb.dim), np.float16)


def embed_cache_dir(cfg):
    return os.path.join(cfg.index_dir, f"embed_cache_{data_fingerprint(cfg)}")


def make_embedder(cfg, dev):
    if cfg.embedder == "hash":
        return HashEmbedder(cfg.feat_dim)
    return STEmbedder(cfg.embed_model, dev, cfg.embed_max_tokens, cfg.embed_batch, cfg.embed_prefix,
                      cfg.embed_bucket, cfg.embed_dtype)


# ---------------------------------------------------------------- reading
def shard_files(cfg):
    if cfg.split_dir:                        # stored split: its own files, train / val / test in order
        from sci_jepa.data.split import split_files
        return split_files(cfg.split_dir)
    files = []
    for d in cfg.parquet_dirs:
        files += sorted(glob.glob(os.path.join(d, "*.parquet")))
    rng = np.random.default_rng(cfg.data_seed)
    rng.shuffle(files)                       # tiers are random papers, not the first shards
    return files


def iter_records(cfg):
    """Each shard is a contiguous slice of one subset and a narrow year range, so
    a tier takes `rows_per_shard` random rows from many shuffled shards instead of
    whole shards (whole shards would make the time split degenerate)."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(cfg.data_seed + 1)
    for f in shard_files(cfg):
        pf = pq.ParquetFile(f)
        cols = [c for c in READ_COLS if c in pf.schema_arrow.names]
        n = pf.metadata.num_rows
        if cfg.rows_per_shard and cfg.rows_per_shard < n:
            take = np.sort(rng.choice(n, cfg.rows_per_shard, replace=False))
            t = pf.read(columns=cols).take(take)
            yield from t.to_pylist()
            del t
        else:
            for batch in pf.iter_batches(batch_size=512, columns=cols):
                yield from batch.to_pylist()


def _temperature_quota(counts, total, alpha, cap=0):
    """Papers per domain: share proportional to count^alpha (alpha 1 = natural, 0.5 = square root, 0 = equal),
    never more than a domain has (or than cap), leftovers redistributed to the domains that still have papers."""
    avail = {d: (min(n, cap) if cap else n) for d, n in counts.items() if n > 0}
    quota, left = {d: 0 for d in avail}, int(min(total, sum(avail.values())))
    while left > 0:
        open_ = [d for d in avail if quota[d] < avail[d]]
        if not open_:
            break
        w = np.asarray([counts[d] ** alpha for d in open_], float); w /= w.sum()
        add = np.floor(w * left).astype(int)
        add[np.argsort(-w)[:left - add.sum()]] += 1             # remainder to the largest shares
        moved = 0
        for d, a in zip(open_, add):
            g = int(min(a, avail[d] - quota[d])); quota[d] += g; moved += g
        left -= moved
        if moved == 0:
            break
    return quota


def select_balanced(cfg):
    """Domain-balanced corpus (field_balance): a first pass reads only cheap metadata columns of candidate rows
    (rows_per_shard x field_pool per shard, until field_pool x max_papers candidates), maps every paper to its
    canonical domain (ku_schema.domain_of) and draws max_papers of them with temperature sampling over domains.
    -> ([(file, row)], stats). Nothing is embedded for papers that are not selected."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(cfg.data_seed + 7)
    target = cfg.field_pool * cfg.max_papers
    meta = ["paper_id", "subset", "source_year", "oa_year", "field_subfield"]
    cand = []
    for f in shard_files(cfg):
        pf = pq.ParquetFile(f)
        n = pf.metadata.num_rows
        cols = [c for c in meta if c in pf.schema_arrow.names]
        k = min(n, cfg.rows_per_shard * cfg.field_pool) if cfg.rows_per_shard else n
        take = np.sort(rng.choice(n, k, replace=False)) if k < n else np.arange(n)
        rows = pf.read(columns=cols).take(take).to_pylist()
        for i, rec in zip(take, rows):
            if keep_record(rec, cfg)[0]:
                cand.append((f, int(i), S.domain_of(rec.get("field_subfield"))))
        if len(cand) >= target:
            break
    by = defaultdict(list)
    for c in cand:
        by[c[2]].append(c)
    counts = {d: len(v) for d, v in by.items()}
    alpha = {"temperature": cfg.field_alpha, "uniform": 0.0, "cap": 1.0}.get(cfg.field_balance, 1.0)
    quota = _temperature_quota(counts, cfg.max_papers, alpha, cfg.field_cap)
    sel = []
    for d, v in by.items():
        sel += [v[i][:2] for i in rng.permutation(len(v))[:quota.get(d, 0)]]
    stats = {"mode": cfg.field_balance, "alpha": alpha, "cap": cfg.field_cap, "candidates": len(cand),
             "pool_by_domain": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
             "selected_by_domain": dict(sorted(quota.items(), key=lambda kv: -kv[1]))}
    print(f"[ku_graph] domain-balanced corpus ({cfg.field_balance}, alpha {alpha}): {len(sel)} of {len(cand)} "
          f"candidates; " + ", ".join(f"{d} {counts[d]}->{quota[d]}" for d in stats["selected_by_domain"]), flush=True)
    return sel, stats


def iter_selected(cfg, sel):
    """Records of the selected (file, row) pairs, one read per file."""
    import pyarrow.parquet as pq
    by = defaultdict(list)
    for f, r in sel:
        by[f].append(r)
    for f in shard_files(cfg):
        if f not in by:
            continue
        pf = pq.ParquetFile(f)
        cols = [c for c in READ_COLS if c in pf.schema_arrow.names]
        t = pf.read(columns=cols).take(np.asarray(sorted(by[f])))
        yield from t.to_pylist()
        del t


def keep_record(rec, cfg):
    """-> (keep, year or None)."""
    if cfg.subsets and rec.get("subset") not in cfg.subsets:
        return False, None
    y = S.paper_year(rec)
    if cfg.split_mode == "time" and y is None:
        return False, None
    if y is not None and not (cfg.year_min <= y <= cfg.year_max):
        return False, y
    return True, y


# ---------------------------------------------------------------- build
def build_blob(cfg, records=None):
    """records: optional iterable of dicts in the normalized schema (synthetic
    tests); default streams cfg.parquet_dirs."""
    from sci_jepa.config import resolve_device
    dev = resolve_device(cfg.device)
    emb = make_embedder(cfg, dev)
    if records is None and cfg.embedder != "hash":     # real data: resumable embedding
        emb = ResumableEmbedder(emb, embed_cache_dir(cfg))
    t0 = time.time()
    balance = None
    if records is None and cfg.split_dir:              # stored split (sci-jepa split): no re-sampling or re-splitting
        from sci_jepa.data.split import iter_stored, load_manifest
        man = load_manifest(cfg.split_dir)
        balance = {"stored_split": cfg.split_dir, "split_id": man["split_id"], **man["balance"],
                   "pool_by_domain": man["pool_by_domain"], "selected_by_domain": {s: v["by_domain"] for s, v in man["selected"].items()}}
        print(f"[ku_graph] stored split {cfg.split_dir} ({man['split_id']}): years {man['years']}", flush=True)
        records = iter_stored(cfg, READ_COLS)
    elif records is None and cfg.field_balance != "none" and cfg.max_papers:   # domain-balanced corpus
        sel, balance = select_balanced(cfg)
        records = iter_selected(cfg, sel)
    records = iter_records(cfg) if records is None else records

    comp_x, comp_type, comp_paper, comp_flags, comp_snip, comp_text = [], [], [], defaultdict(list), [], []
    role_src, role_dst, role_rel = [], [], []
    p_year, p_field, p_sub, p_type, p_key, p_id, p_refs, p_dom = [], [], [], [], [], [], [], []
    p_split = []                              # stored split only: each paper's train / val / test label
    stats, skipped = Counter(), Counter()
    buf_txt, n_comp = [], 0
    pm_parts = []                             # paper_masked: each paper's cleaned KUs + edges
    mp_src = []                               # (comp id, text) of results / claims: minimal-pair sources

    def flush():
        if buf_txt:
            comp_x.append(np.asarray(emb.encode(buf_txt), dtype=np.float16))
            buf_txt.clear()

    for rec in records:
        keep, y = keep_record(rec, cfg)
        if not keep:
            skipped["filtered"] += 1
            continue
        kus, edges, st = S.record_to_kus(rec, cfg.ku_min_chars, cfg.optional_kus)
        stats.update(st)
        types = [t for t, _, _ in kus]
        if len(kus) < cfg.min_kus_per_paper or not all(t in types for t in cfg.required_kus):
            skipped["too_few_kus"] += 1
            continue
        if cfg.ku_mode == "paper":           # same papers as section mode, each collapsed into one KU
            kus, edges = S.paper_ku(kus, edges)
        elif cfg.ku_mode == "paper_masked":
            pm_parts.append((kus, edges))
        p = len(p_year)
        p_year.append(-1 if y is None else y)
        p_split.append(rec.get("_split"))
        coarse, fine = S.split_field(rec.get("field_subfield"))
        p_field.append(coarse); p_sub.append(fine); p_dom.append(S.DOMAINS.index(S.domain_of(rec.get("field_subfield"))))
        p_type.append(S.paper_type(rec.get("type_of_paper")))
        p_key.append(S.openalex_key(rec.get("openalex_id")))
        p_id.append(str(rec.get("paper_id")))
        p_refs.append(S.referenced(rec))
        if cfg.ku_mode != "paper":
            mp_src += [(n_comp + k, text) for k, (t, text, _) in enumerate(kus) if t in S.MP_TYPES]
        for t, text, fl in kus:
            buf_txt.append(text)
            comp_snip.append(text[:cfg.snippet_chars]); comp_text.append(text)
            comp_type.append(S.COMP_TYPES.index(t)); comp_paper.append(p)
            for k in ("quantitative", "cites_figure", "challenged"):
                comp_flags[k].append(bool(fl.get(k, False)))
        for a, r, b in edges:
            role_src.append(n_comp + a); role_dst.append(n_comp + b); role_rel.append(S.REL_ID[r])
        n_comp += len(kus)
        if len(buf_txt) >= cfg.embed_chunk:
            flush()
        if len(p_year) % 10000 == 0:
            print(f"[ku_graph] {len(p_year)} papers, {n_comp} KUs ({time.time() - t0:.0f}s)", flush=True)
        if cfg.max_papers and not cfg.split_dir and len(p_year) >= cfg.max_papers:   # stored split: capped per split
            break
    flush()
    P = len(p_year)
    if P == 0:
        raise RuntimeError("no papers selected -- check parquet_dirs / subsets / split_mode")

    comp_type = np.asarray(comp_type, dtype=np.int64)
    comp_paper = np.asarray(comp_paper, dtype=np.int64)
    comp_x = np.concatenate(comp_x, 0)

    # ---- fields --------------------------------------------------------------
    fields = sorted(set(p_field)); subs = sorted(set(p_sub))
    fid = {f: i for i, f in enumerate(fields)}; sid = {s: i for i, s in enumerate(subs)}
    field_x = np.asarray(emb.encode(fields), dtype=np.float16)

    # ---- citations inside the selected corpus ------------------------------------
    key_to_p = {k: i for i, k in enumerate(p_key) if k}
    pp_s, pp_d, pp_w = [], [], []
    for i, refs in enumerate(p_refs):
        for k in refs:
            j = key_to_p.get(k)
            if j is not None and j != i:
                pp_s.append(i); pp_d.append(j); pp_w.append(1)
    n_cites = len(pp_s)

    # ---- proxy grounded edges -------------------------------------------------------
    from sci_jepa.data.graph_index import build_proxy_edges
    for t in cfg.proxy_types:
        ids = np.nonzero(comp_type == S.COMP_TYPES.index(t))[0]
        pr = build_proxy_edges(comp_x[ids].astype(np.float32), comp_paper[ids],
                               cfg.proxy_tau, cfg.proxy_topk, dev)
        if len(pr):
            a, b = ids[pr[:, 0]], ids[pr[:, 1]]
            role_src += a.tolist(); role_dst += b.tolist(); role_rel += [S.REL_ID["proxy"]] * len(a)
            pp_s += comp_paper[a].tolist(); pp_d += comp_paper[b].tolist(); pp_w += [2] * len(a)
        print(f"[ku_graph] proxy '{t}' edges: {len(pr)} (tau={cfg.proxy_tau})")

    # ---- membership CSR + splits --------------------------------------------------
    order = np.argsort(comp_paper, kind="stable")
    pc_ptr = np.concatenate([[0], np.cumsum(np.bincount(comp_paper, minlength=P))])
    years = np.asarray(p_year)
    train, val, test = make_splits(cfg, years, p_split)

    pm = build_context_texts(cfg, emb, pm_parts, p_field, p_sub, (train, val, test)) \
        if cfg.ku_mode == "paper_masked" else {}
    pm.update(build_minimal_pairs(emb, mp_src, comp_paper, (val, test), comp_type, comp_x, pc_ptr, order))

    ku_stats = {f"{t}:{w}": int(c) for (t, w), c in sorted(stats.items())}
    print(f"[ku_graph] {P} papers, {len(comp_type)} KUs, {n_cites} in-corpus citations, "
          f"{time.time() - t0:.0f}s; skipped {dict(skipped)}")
    if getattr(emb, "n_seen", 0):
        print(f"[ku_graph] text encoder truncated {emb.n_truncated} of {emb.n_seen} texts "
              f"(context {cfg.embed_max_tokens} tokens)")
    if getattr(emb, "reused", 0):
        print(f"[ku_graph] resumed: {emb.reused} texts loaded from the embedding cache")
    return {"comp_x": comp_x, "comp_type": comp_type, "comp_paper": comp_paper,
            "comp_quantitative": np.asarray(comp_flags["quantitative"]),
            "comp_cites_figure": np.asarray(comp_flags["cites_figure"]),
            "comp_challenged": np.asarray(comp_flags["challenged"]),
            "comp_snippet": np.asarray(comp_snip, dtype=object),
            "comp_text": comp_text,                    # full texts: saved to a side file by get_index
            "pc_ptr": pc_ptr, "pc_idx": order,
            "role_src": np.asarray(role_src, dtype=np.int64), "role_dst": np.asarray(role_dst, dtype=np.int64),
            "role_rel": np.asarray(role_rel, dtype=np.int64),
            "pp_src": np.asarray(pp_s, dtype=np.int64), "pp_dst": np.asarray(pp_d, dtype=np.int64),
            "pp_w": np.asarray(pp_w, dtype=np.int64),
            "paper_field": np.asarray([fid[f] for f in p_field]), "paper_subfield": np.asarray([sid[s] for s in p_sub]),
            "paper_year": years, "paper_type": np.asarray(p_type), "paper_ids": p_id,
            "paper_domain": np.asarray(p_dom, dtype=np.int64), "domain_names": list(S.DOMAINS),
            "field_names": fields, "field_x": field_x,
            "train_papers": train, "val_papers": val, "test_papers": test, **pm,
            "build_stats": {"ku_outcomes": ku_stats, "skipped": dict(skipped), "in_corpus_citations": n_cites,
                            "embedder": cfg.embedder if cfg.embedder == "hash" else cfg.embed_model,
                            "embed_max_tokens": cfg.embed_max_tokens, "feat_dim": int(comp_x.shape[1]),
                            "kus_truncated": int(getattr(emb, "n_truncated", 0)),
                            "kus_embedded": int(getattr(emb, "n_seen", 0)),
                            "split_mode": cfg.split_mode, "time_T": cfg.time_T, "domain_balance": balance,
                            "domains": {S.DOMAINS[k]: int(v) for k, v in zip(*np.unique(p_dom, return_counts=True))}}}


def _topic_donors(eval_splits, comp_type, comp_x, pc_ptr, order):
    """For every validation / test paper: the other paper of the same split with the most similar
    problem (centred mean embedding of its context + question KUs), as in pool_mode="topic"."""
    prob = np.isin(comp_type, [S.COMP_TYPES.index("context"), S.COMP_TYPES.index("question")])
    mu = comp_x[np.nonzero(prob)[0]].astype(np.float32).mean(0) if prob.any() else 0
    donor = {}
    for papers in eval_splits:
        papers = [int(p) for p in papers]
        if len(papers) < 2:
            continue
        V = []
        for p in papers:
            cs = order[pc_ptr[p]:pc_ptr[p + 1]]
            sel = cs[prob[cs]] if prob[cs].any() else cs
            v = comp_x[sel].astype(np.float32).mean(0) - mu
            V.append(v / (np.linalg.norm(v) + 1e-8))
        V = np.stack(V); sim = V @ V.T
        np.fill_diagonal(sim, -np.inf)
        for i, p in enumerate(papers):
            donor[p] = papers[int(np.argmax(sim[i]))]
    return donor


def build_minimal_pairs(emb, src, comp_paper, eval_splits, comp_type, comp_x, pc_ptr, order):
    """Minimal-pair counterfactuals of every validation / test result, claim, method and procedure,
    embedded with the same text encoder (mp_*): direction / negation / number edits
    (ku_schema.counterfactuals) and, for methods and procedures, a substitution from the same section
    of the topic-matched paper (ku_schema.substitution)."""
    keep = set(int(p) for s in eval_splits for p in s)
    donor_of = _topic_donors(eval_splits, comp_type, comp_x, pc_ptr, order)
    section = {}                                          # (paper, type) -> text of its first KU of that type
    for c, text in src:
        section.setdefault((int(comp_paper[c]), int(comp_type[c])), text)
    rows, texts = [], []
    for c, text in src:
        p, t = int(comp_paper[c]), int(comp_type[c])
        if p not in keep:
            continue
        for kind, edited in S.counterfactuals(text):
            rows.append((c, S.MP_KINDS.index(kind))); texts.append(edited)
        if S.COMP_TYPES[t] in S.DESIGN_TYPES and p in donor_of:
            edited = S.substitution(text, section.get((donor_of[p], t)))
            if edited:
                rows.append((c, S.MP_KINDS.index("substitution"))); texts.append(edited)
    X = np.asarray(emb.encode(texts), dtype=np.float16) if texts else np.zeros((0, 1), np.float16)
    kinds = np.asarray([k for _, k in rows], dtype=np.int64)
    by = {S.MP_KINDS[k]: int((kinds == k).sum()) for k in np.unique(kinds)} if len(kinds) else {}
    print(f"[ku_graph] minimal pairs: {len(texts)} counterfactual texts for "
          f"{len({c for c, _ in rows})} validation / test results, claims, methods and procedures {by}")
    return {"mp_comp": np.asarray([c for c, _ in rows], dtype=np.int64), "mp_kind": kinds, "mp_x": X,
            "mp_snippet": np.asarray([t[:240] for t in texts], dtype=object), "mp_text": texts}


def build_context_texts(cfg, emb, parts, p_field, p_sub, splits):
    """ku_mode="paper_masked": for every paper and template, the given KUs joined into ONE
    text and encoded together (ctx_*). For val / test papers also counterfactual forward texts
    with the procedure (condition) or the method replaced by a donor paper's, for up to 3
    donors each (cf_*)."""
    t0 = time.time()
    names = S.active_templates(cfg.pm_templates)
    texts, c_paper, c_tpl, seen = [], [], [], {}
    for p, (kus, edges) in enumerate(parts):
        for name in names:
            given, targets = S.TEMPLATES[name]
            if not any(t == k for k, _, _ in kus for t in targets):
                continue                                   # nothing of the target type to predict
            txt = S.compose_text(kus, edges, given)
            if not txt:
                continue
            key = (p, txt)
            if key not in seen:                            # identical texts share one row
                seen[key] = len(texts); texts.append(txt)
            c_paper.append(p); c_tpl.append(S.TEMPLATE_NAMES.index(name))
    rows = [seen[(p, S.compose_text(parts[p][0], parts[p][1], S.TEMPLATES[S.TEMPLATE_NAMES[t]][0]))]
            for p, t in zip(c_paper, c_tpl)]
    # counterfactual forward contexts for evaluation papers: change(Condition := c'), change(Method := m')
    rng = np.random.default_rng(cfg.split_seed + 99)
    first = lambda kus, t: next((txt for k, txt, _ in kus if k == t), None)
    cf_txt, cf_paper, cf_kind, cf_donor = [], [], [], []
    for split in splits[1:]:
        pool = [int(p) for p in split]
        for p in pool:
            for kind, t in enumerate(("procedure", "method")):
                if first(parts[p][0], t) is None:
                    continue
                # 3 donors from different pools (same subfield, same field, anywhere in the split):
                # evaluation uses the first one that is not a visible neighbour of the query
                donors = []
                for same in (lambda q: p_sub[q] == p_sub[p], lambda q: p_field[q] == p_field[p], lambda q: True):
                    cand = [q for q in pool if q != p and q not in donors and same(q) and first(parts[q][0], t)]
                    if cand:
                        donors.append(int(rng.choice(cand)))
                for d in donors:
                    cf_txt.append(S.compose_text(parts[p][0], parts[p][1], S.TEMPLATES["forward"][0],
                                                 replace={t: first(parts[d][0], t)}))
                    cf_paper.append(p); cf_kind.append(kind); cf_donor.append(int(d))
    print(f"[ku_graph] paper_masked: {len(texts)} context texts ({len(names)} templates) + "
          f"{len(cf_txt)} counterfactual texts to encode", flush=True)
    X = np.asarray(emb.encode(texts), dtype=np.float16) if texts else np.zeros((0, 1), np.float16)
    CF = np.asarray(emb.encode(cf_txt), dtype=np.float16) if cf_txt else np.zeros((0, X.shape[1]), np.float16)
    print(f"[ku_graph] paper_masked texts encoded in {time.time() - t0:.0f}s")
    return {"ctx_x": X, "ctx_row": np.asarray(rows, dtype=np.int64), "ctx_paper": np.asarray(c_paper, dtype=np.int64),
            "ctx_tpl": np.asarray(c_tpl, dtype=np.int64), "template_names": S.TEMPLATE_NAMES,
            "cf_x": CF, "cf_paper": np.asarray(cf_paper, dtype=np.int64), "cf_kind": np.asarray(cf_kind, dtype=np.int64),
            "cf_donor": np.asarray(cf_donor, dtype=np.int64)}


def make_splits(cfg, years, labels=None):
    P = len(years)
    if cfg.split_dir:                        # stored split: the labels written by sci-jepa split
        lab = np.asarray(labels, dtype=object)
        train, val, test = (np.nonzero(lab == s)[0] for s in ("train", "val", "test"))
        for s, idx in (("train", train), ("val", val), ("test", test)):
            ys = years[idx]
            print(f"[ku_graph] stored split {s}: {len(idx)} papers, years {ys.min() if len(ys) else '-'}-{ys.max() if len(ys) else '-'}")
        return train, val, test
    if cfg.split_mode == "time":
        T = cfg.time_T
        train = np.nonzero(years <= T)[0]
        val = np.nonzero(years == T + 1)[0]
        test = np.nonzero(years >= T + 2)[0] if cfg.test_open_ended else np.nonzero(years == T + 2)[0]
        print(f"[ku_graph] time split T={T}: train <= {T}: {len(train)}  val {T + 1}: {len(val)}  "
              f"test {T + 2}{'+' if cfg.test_open_ended else ''}: {len(test)}")
        if cfg.encoder_cutoff_year and T + 2 <= cfg.encoder_cutoff_year:
            late = int((years[test] > cfg.encoder_cutoff_year).sum())
            print(f"[ku_graph] note: test years from {T + 2} overlap the text encoder's pretraining (cutoff "
                  f"{cfg.encoder_cutoff_year}); {late} of {len(test)} test papers are later than the cutoff. "
                  f"Every task is also reported separately for those (by_period), for a contamination-free reading.")
        return train, val, test
    rng = np.random.default_rng(cfg.split_seed)
    perm = rng.permutation(P)
    a = int(cfg.split[0] * P); b = a + int(cfg.split[1] * P)
    return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])


def data_fingerprint(cfg):
    keys = ["parquet_dirs", "subsets", "max_papers", "data_seed", "rows_per_shard", "year_min", "year_max", "split_mode",
            "time_T", "test_open_ended", "split", "split_seed", "ku_min_chars", "optional_kus", "min_kus_per_paper",
            "required_kus", "ku_mode", "pm_templates", "field_balance", "field_alpha", "field_cap", "field_pool", "embedder", "embed_model", "embed_max_tokens", "proxy_types", "proxy_tau",
            "proxy_topk", "snippet_chars"]
    vals = [INDEX_VERSION] + [getattr(cfg, k) for k in keys]
    if cfg.embed_dtype != "float32":                  # fp16 vectors differ in the 5th decimal: own cache
        vals.append(cfg.embed_dtype)                  # (embed_bucket does not change the vectors)
    if cfg.split_dir:                                 # stored split: which split, not where it lies
        from sci_jepa.data.split import load_manifest
        vals.append(("stored_split", load_manifest(cfg.split_dir)["split_id"]))
    return hashlib.sha1(repr(vals).encode()).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    from sci_jepa.config import PaperAConfig
    from sci_jepa.training.train import parse_overrides
    ap = argparse.ArgumentParser("ku_graph")
    ap.add_argument("--set", nargs="*", default=[])
    cfg = PaperAConfig().override(**parse_overrides(ap.parse_args().set))
    from sci_jepa.data.graph_index import get_index
    gi = get_index(cfg)
    import json
    print(json.dumps(gi.summary(), indent=1, default=str))
