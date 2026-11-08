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
import functools
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.data.ku_schema import (CHILD_RELS, COMP_TYPES, N_RELS, PARAPHRASE_SIBLINGS,  # noqa: F401
                                     REL_CITES, REL_ID, REL_MEMBER, REL_NONE, REL_SAME_FIELD,
                                     REL_SAME_PAPER, REL_SAME_SUBFIELD, REL_DO, ROLE_RELS)

HUB = len(COMP_TYPES)                  # token type id of a paper hub
N_TOKEN_TYPES = HUB + 1


# ---------------------------------------------------------------------- lookups as arrays
# The per-KU and per-paper lookups were Python dicts of lists of tuples, rebuilt in every process (~0.5 GB per million
# edges: ~60 GB per process at 100M KUs, slowly copied by every forked batch worker as reference counts change). They
# are now CSR arrays (ptr: where each node's entries start; entries in the order the dicts held them), saved next to the
# index and memory-mapped, so the processes of a node share one copy. The objects below answer exactly like the dicts.
LOOKUP_KEYS = ("ce_ptr", "ce_nbr", "ce_rel",           # comp_edges: KU -> [(other KU, rel id)]
               "ch_ptr", "ch_idx", "pa_ptr", "pa_idx",  # children: claim -> [evidence / implication], parents: back
               "pn_ptr", "pn_nbr", "pn_w",              # paper_nbrs: paper -> {paper: preference}
               "ci_ptr", "ci_nbr")                      # cites_pairs: paper -> cited / citing papers (sorted)


class _Rows:
    """node -> its entries, as the defaultdict(list) it replaces: .get(k, default) gives a new list of ints (of
    (int, int) tuples with vals) in insertion order, or default when k has none; d[k] gives [] then."""

    def __init__(self, ptr, cols, vals=None):
        self.ptr, self.cols, self.vals, self.n = ptr, cols, vals, len(ptr) - 1

    def _span(self, k):
        k = int(k)
        if 0 <= k < self.n:
            a, b = int(self.ptr[k]), int(self.ptr[k + 1])
            if a < b:
                return a, b
        return None

    def _make(self, a, b):
        cols = self.cols[a:b].tolist()
        return cols if self.vals is None else list(zip(cols, self.vals[a:b].tolist()))

    def get(self, k, default=None):
        s = self._span(k)
        return default if s is None else self._make(*s)

    def __getitem__(self, k):
        return self.get(k, self._make(0, 0))

    def __contains__(self, k):
        return self._span(k) is not None

    def __len__(self):
        return int(np.count_nonzero(np.diff(self.ptr)))

    def entries(self, keys):
        """Every entry of keys (int64 array) in one pass, in the order a loop over keys and their .get lists visits
        them: -> (position in keys of each entry's node, cols, vals or None)."""
        a = self.ptr[keys]
        n = self.ptr[keys + 1] - a
        idx = np.repeat(a - (np.cumsum(n) - n), n) + np.arange(int(n.sum()))
        return np.repeat(np.arange(len(keys)), n), self.cols[idx], None if self.vals is None else self.vals[idx]


class _NbrRows(_Rows):
    """paper -> {paper: preference}, in the dict's insertion order."""

    def _make(self, a, b):
        return dict(zip(self.cols[a:b].tolist(), self.vals[a:b].tolist()))


class _PairSet:
    """(a, b) in cites_pairs: a and b are linked by a citation (either direction). of(a): the set of a's
    (cached per process for the last 65,536 papers asked: build_view asks for the same papers again and again)."""

    def __init__(self, ptr, cols):
        self.rows = _Rows(ptr, cols)
        self.of = functools.lru_cache(maxsize=1 << 16)(self._of)

    def _of(self, a):
        return frozenset(self.rows.get(a, ()))

    def __contains__(self, pair):
        s = self.rows._span(pair[0])
        if s is None:
            return False
        row = self.rows.cols[s[0]:s[1]]
        i = int(np.searchsorted(row, int(pair[1])))
        return i < len(row) and int(row[i]) == int(pair[1])

    def __len__(self):
        return len(self.rows.cols)


class _Slices:
    """paper_comps[p]: the KUs of paper p (a view of pc_idx), as the list of views it replaces."""

    def __init__(self, ptr, idx):
        self.ptr, self.idx, self.n = ptr, idx, len(ptr) - 1

    def __getitem__(self, p):
        p = int(p)
        if p < 0:
            p += self.n
        if not 0 <= p < self.n:
            raise IndexError(f"paper {p} out of range")
        return self.idx[self.ptr[p]:self.ptr[p + 1]]

    def __len__(self):
        return self.n

    def __iter__(self):
        return (self[p] for p in range(self.n))


class _Strings:
    """KU texts stored as one UTF-8 byte array + offsets (the object array of 100M Python strings it replaces was
    unpickled into every process); s[i] is the same str."""

    def __init__(self, data, off):
        self.data, self.off = data, off

    def __getitem__(self, i):
        i = int(i)
        return bytes(self.data[self.off[i]:self.off[i + 1]]).decode("utf-8", "surrogatepass")

    def __len__(self):
        return len(self.off) - 1


def _csr(key, n, *cols):
    """Entries grouped by key (0..n-1), each group in its original order: (ptr, *cols)."""
    order = np.argsort(key, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(key, minlength=n), out=ptr[1:])
    return (ptr, *(c[order] for c in cols))


def _both_ways(a, b):
    """Edge i (a, b) as two events, 2i at a (towards b) and 2i + 1 at b (towards a): the order the dicts were filled."""
    key, nbr = np.empty(2 * len(a), np.int64), np.empty(2 * len(a), np.int64)
    key[0::2], key[1::2] = a, b
    nbr[0::2], nbr[1::2] = b, a
    return key, nbr


def build_lookups(b):
    """The LOOKUP_KEYS arrays from an index's edge arrays (blob or GraphIndex.__dict__), equal entry for entry and in
    the same order to the dicts they replace."""
    n_c, P = len(b["comp_type"]), len(b["paper_field"])
    s, d, r = (np.asarray(b[k], dtype=np.int64) for k in ("role_src", "role_dst", "role_rel"))
    key, nbr = _both_ways(s, d)
    out = dict(zip(("ce_ptr", "ce_nbr", "ce_rel"), _csr(key, n_c, nbr, np.repeat(r, 2).astype(np.int8))))
    child = np.isin(r, [REL_ID[x] for x in CHILD_RELS])
    out["ch_ptr"], out["ch_idx"] = _csr(s[child], n_c, d[child])
    out["pa_ptr"], out["pa_idx"] = _csr(d[child], n_c, s[child])
    # paper_nbrs[a][b] = max(paper_nbrs[a].get(b, 0), w) for every event: per pair the maximum (and at least 0), at
    # the place of the pair's first event
    a, bb, w = (np.asarray(b[k], dtype=np.int64) for k in ("pp_src", "pp_dst", "pp_w"))
    key, nbr = _both_ways(a, bb)
    ww = np.repeat(w, 2)
    if len(key):
        pair = key * P + nbr
        order = np.argsort(pair, kind="stable")
        ps = pair[order]
        start = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
        first, wmax = order[start], np.maximum(np.maximum.reduceat(ww[order], start), 0)
        own, other = ps[start] // P, ps[start] % P
        o = np.lexsort((first, own))
        out["pn_ptr"], out["pn_nbr"], out["pn_w"] = _csr(own[o], P, other[o], wmax[o])
    else:
        out["pn_ptr"], out["pn_nbr"], out["pn_w"] = _csr(key, P, nbr, ww)
    c1 = w == 1
    key, nbr = _both_ways(a[c1], bb[c1])
    u = np.unique(key * P + nbr)                       # sorted: by paper, then by the other paper
    out["ci_ptr"], out["ci_nbr"] = _csr(u // P, P, u % P)
    return out


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
        if any(k not in self.__dict__ for k in LOOKUP_KEYS):   # an index built in memory, or saved before LOOKUP_KEYS
            for k, v in build_lookups(self.__dict__).items():
                setattr(self, k, v)
        if hasattr(self, "comp_snippet_data"):          # older indexes hold an object array of str instead
            self.comp_snippet = _Strings(self.comp_snippet_data, self.comp_snippet_off)
        self.paper_comps = _Slices(self.pc_ptr, self.pc_idx)
        self.comp_edges = _Rows(self.ce_ptr, self.ce_nbr, self.ce_rel)   # KU -> [(other KU, rel id)]
        self.children = _Rows(self.ch_ptr, self.ch_idx)                  # claim -> its evidence / implications
        self.parents = _Rows(self.pa_ptr, self.pa_idx)                   # evidence / implication -> its claim
        self.paper_nbrs = _NbrRows(self.pn_ptr, self.pn_nbr, self.pn_w)  # paper -> {paper: preference}
        self.cites_pairs = _PairSet(self.ci_ptr, self.ci_nbr)
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
def _proxy_block(n, device, block):
    """Rows per block: `block` (1024, as always) while its scores fit in free GPU memory, else halved until they do.
    A block holds ~6 bytes per score (fp32 scores, the cross-paper mask, temporaries): ~10 GB at 2M KUs, ~28 GB at
    4.5M, so only corpora that ran out of memory before get smaller blocks (and only on CUDA)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return block
    free = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
    while block > 16 and block * n * 6 > 0.9 * free:
        block //= 2
    return block


def build_proxy_edges(x, comp_paper, tau, topk, device, block=1024):
    """Cross-paper near-duplicate pairs (i < j) among KUs of one type. Rows are scored in blocks: a block
    holds block x n fp32 scores plus a bool mask, ~10 GB at 2M KUs (4096 rows needed ~49 GB with z, more
    than a 48 GB L40S). Each row's top-k does not depend on the block size; _proxy_block keeps 1024 rows
    wherever they fit."""
    n = x.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    z = F.normalize(torch.as_tensor(x, dtype=torch.float32, device=device), dim=-1)
    owner = torch.as_tensor(comp_paper, device=device)
    block = _proxy_block(n, device, block)
    k = min(topk + 1, n)
    pairs = []
    for s in range(0, n, block):
        sim = z[s:s + block] @ z.T
        rows = torch.arange(s, min(s + block, n), device=device)
        sim[owner[rows].unsqueeze(1) == owner.unsqueeze(0)] = -1.0   # cross-paper only (also kills self)
        val, idx = sim.topk(k, dim=1)
        keep = val >= tau
        r = rows.unsqueeze(1).expand_as(idx)[keep]
        c = idx[keep]
        pairs.append(torch.stack([torch.minimum(r, c), torch.maximum(r, c)], 1).cpu())
    out = torch.unique(torch.cat(pairs, 0), dim=0)
    return out.numpy().astype(np.int64)


# The KU and counterfactual vectors (44.6M x 768 fp16 = 69 GB on the full arXiv split) live next to the index as
# .npy files, opened memory-mapped and read-only: every training process on a node then shares one cached copy instead
# of loading its own (four copies OOM-killed jobs 38878 / 38883), and a process reads only the rows it touches.
MMAP_KEYS = ("comp_x", "mp_x")
# Per-KU / per-edge arrays (and the lookups built from them) live in .npy files too, opened memory-mapped and handed
# out as plain read-only arrays: ~6 GB of int64 arrays per process at 100M KUs before, now one shared copy per node.
SHARED_KEYS = ("comp_type", "comp_paper", "comp_quantitative", "comp_cites_figure", "comp_challenged", "pc_ptr", "pc_idx",
               "role_src", "role_dst", "role_rel", "comp_snippet_data", "comp_snippet_off") + LOOKUP_KEYS
SPILL_SUFFIX = ".spill.npy"            # a .npy the build streamed to disk (ku_graph): moved into place, not copied


def _sidecar(path, key):
    return f"{path[:-3]}.{key}.npy"


def _save_sidecars(blob, path, keys):
    """Write blob[k] for keys as .npy files next to the index; -> the blob without them."""
    rest = dict(blob)
    for k in keys:
        v = rest.get(k)
        if not isinstance(v, np.ndarray):
            continue
        rest.pop(k)
        if isinstance(v, np.memmap) and str(v.filename).endswith(SPILL_SUFFIX):
            os.replace(v.filename, _sidecar(path, k))  # the open map stays valid; the file just gets its final name
            continue
        tmp = _sidecar(path, k)[:-4] + f".{os.getpid()}.tmp.npy"
        np.save(tmp, v)
        os.replace(tmp, _sidecar(path, k))
    return rest


def save_index(blob, path):
    """Write the index: MMAP_KEYS and SHARED_KEYS as .npy files, the rest as the .pt (each written to a temporary name,
    then renamed: a process loading at the same time sees the old or the new index, never half of one)."""
    rest = _save_sidecars(blob, path, MMAP_KEYS + SHARED_KEYS)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(rest, tmp, pickle_protocol=4)          # torch's default protocol 2 fails on objects > 4 GiB (full arXiv)
    os.replace(tmp, path)


def _open_sidecars(blob, path, keys, as_memmap):
    """SCIJEPA_MMAP_DIR: a folder holding copies of the .npy files (e.g. on a node's RAM disk) to open instead of
    the ones next to the index, for nodes that read the shared disk slowly."""
    opened = []
    for k in keys:
        f = _sidecar(path, k)
        local = os.path.join(os.environ.get("SCIJEPA_MMAP_DIR", ""), os.path.basename(f))
        if os.environ.get("SCIJEPA_MMAP_DIR") and os.path.exists(local) and os.path.getsize(local) == os.path.getsize(f):
            f = local
        if k not in blob and os.path.exists(f):
            m = np.load(f, mmap_mode="r")
            blob[k] = m if as_memmap else np.asarray(m)  # plain array (np.memmap's indexing is slower), same pages
            opened.append(k)
    return opened


def load_index(path):
    blob = torch.load(path, weights_only=False)
    for k in _open_sidecars(blob, path, MMAP_KEYS, True):
        print(f"[paper_a.index] {k}: memory-mapped {_sidecar(path, k)}", flush=True)
    shared = _open_sidecars(blob, path, SHARED_KEYS, False)
    if shared:
        print(f"[paper_a.index] {len(shared)} KU / edge / lookup arrays memory-mapped", flush=True)
    return blob


def _add_lookups(blob, path):
    """An index saved before LOOKUP_KEYS: build them and save them next to it, so later loads (other processes too)
    map them instead of building them again. Not fatal when the folder is read-only."""
    if all(k in blob for k in LOOKUP_KEYS):
        return blob
    t0 = time.time()
    lk = build_lookups(blob)
    try:
        _save_sidecars(lk, path, LOOKUP_KEYS)
        _open_sidecars(blob, path, LOOKUP_KEYS, False)
        print(f"[paper_a.index] lookups built and saved next to the index ({time.time() - t0:.0f}s)", flush=True)
    except OSError as e:
        blob.update(lk)
        print(f"[paper_a.index] lookups built in memory ({e})", flush=True)
    return blob


class _TextStream:
    """Pickles as a plain list of str, read one at a time from the build's text spill (ku_graph): the side file
    loads exactly as before, without all texts in memory while it is written."""

    def __init__(self, texts):
        self.texts = texts

    def __reduce_ex__(self, protocol):
        return list, (), None, iter(self.texts)


def get_index(cfg):
    from sci_jepa.data.ku_graph import build_blob, data_fingerprint
    if cfg.synthetic:
        from sci_jepa.data.synthetic import synthetic_records
        return GraphIndex(build_blob(cfg.override(embedder="hash"), records=synthetic_records(seed=cfg.split_seed)))
    path = os.path.join(cfg.index_dir, f"paper_a_ku_index_{data_fingerprint(cfg)}.pt")
    text_path = path[:-3] + ".texts.pkl"
    if os.path.exists(path):
        print(f"[paper_a.index] loaded {path}")
        gi = GraphIndex(_add_lookups(load_index(path), path))
        gi._text_path = text_path
        return gi
    os.makedirs(cfg.index_dir, exist_ok=True)
    spill = f"{path[:-3]}.{os.getpid()}"               # this process's spill files (another may build at once)
    try:
        blob = build_blob(cfg, spill=spill)
    except BaseException:                              # a failed build leaves no spill files (~150 GB at 100M KUs)
        import glob
        for f in glob.glob(glob.escape(spill) + ".*" + SPILL_SUFFIX):
            os.remove(f)
        raise
    blob.update(build_lookups(blob))
    import pickle
    texts = blob.pop("comp_text")
    with open(text_path + ".tmp", "wb") as f:         # full KU texts: only the text baselines read them
        p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
        p.fast = True                                  # no memo: it would hold every text in memory until the end
        p.dump({"comp_text": _TextStream(texts) if not isinstance(texts, list) else texts,
                "mp_text": blob.pop("mp_text", [])})
    os.replace(text_path + ".tmp", text_path)
    if hasattr(texts, "remove"):
        texts.remove()
    save_index(blob, path)
    print(f"[paper_a.index] saved {path}")
    from sci_jepa.data.ku_graph import embed_cache_dir
    import shutil
    shutil.rmtree(embed_cache_dir(cfg), ignore_errors=True)   # resumable chunks are no longer needed
    gi = GraphIndex(blob)
    gi._text_path = text_path
    return gi


def convert(path):
    """Bring an index saved by an earlier version to the current layout (loading works either way): vectors, KU / edge
    arrays and lookups as memory-mapped .npy files, snippets as UTF-8 bytes + offsets. The values do not change."""
    b = torch.load(path, weights_only=False)
    if isinstance(b.get("comp_snippet"), np.ndarray):
        from sci_jepa.data.ku_graph import strings_to_arrays
        b["comp_snippet_data"], b["comp_snippet_off"] = strings_to_arrays(b.pop("comp_snippet"))
    have = set(b) | {k for k in MMAP_KEYS + SHARED_KEYS if os.path.exists(_sidecar(path, k))}
    if not all(k in have for k in LOOKUP_KEYS):
        full = dict(b)
        _open_sidecars(full, path, SHARED_KEYS, False)
        b.update({k: v for k, v in build_lookups(full).items() if k not in have})
    todo = [k for k in MMAP_KEYS + SHARED_KEYS if isinstance(b.get(k), np.ndarray)]
    print(f"{path}: moving {todo or 'nothing (already converted)'} to .npy files", flush=True)
    if todo:
        save_index(b, path)
        print("done:", {k: f"{os.path.getsize(_sidecar(path, k)) / 2**30:.2f} GiB" for k in todo},
              f"pt {os.path.getsize(path) / 2**30:.2f} GiB")


if __name__ == "__main__":
    # convert an index saved by an earlier version (once; loading works either way, and an older index gets its lookups
    # on first load):   python -m sci_jepa.data.graph_index --mmap <runs/.../paper_a_ku_index_<fingerprint>.pt>
    import argparse
    ap = argparse.ArgumentParser("graph_index")
    ap.add_argument("--mmap", required=True)
    convert(ap.parse_args().mmap)
