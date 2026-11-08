"""The index layout for 100M+ KUs (CSR lookups, memory-mapped arrays, streamed build, blocked similarity) gives
exactly what the in-memory dicts and arrays gave before (data/graph_index.py, data/ku_graph.py, data/instances.py)."""
import os
import pickle
from collections import defaultdict

import numpy as np
import pytest
import torch

from sci_jepa.config import SIZES, PaperAConfig
from sci_jepa.data import graph_index as G
from sci_jepa.data import ku_graph as K
from sci_jepa.data.instances import _role_rel
from sci_jepa.data.ku_schema import CHILD_RELS, REL_ID
from sci_jepa.data.synthetic import synthetic_records

CFG = dict(**SIZES["tiny"], embedder="hash", device="cpu", time_T=2016, test_open_ended=False, proxy_tau=0.6)


def _old_lookups(b):
    """The dicts exactly as GraphIndex built them before LOOKUP_KEYS."""
    comp_edges, children, parents = defaultdict(list), defaultdict(list), defaultdict(list)
    child_ids = {REL_ID[r] for r in CHILD_RELS}
    for s, d, r in zip(b["role_src"].tolist(), b["role_dst"].tolist(), b["role_rel"].tolist()):
        comp_edges[s].append((d, r)); comp_edges[d].append((s, r))
        if r in child_ids:
            children[s].append(d); parents[d].append(s)
    paper_nbrs, cites = defaultdict(dict), set()
    for a, c, w in zip(b["pp_src"].tolist(), b["pp_dst"].tolist(), b["pp_w"].tolist()):
        paper_nbrs[a][c] = max(paper_nbrs[a].get(c, 0), w)
        paper_nbrs[c][a] = max(paper_nbrs[c].get(a, 0), w)
        if w == 1:
            cites.add((a, c)); cites.add((c, a))
    return comp_edges, children, parents, paper_nbrs, cites


def _random_graph(seed, n_c=300, n_p=40, n_e=900, n_pp=400):
    rng = np.random.default_rng(seed)
    child = REL_ID[CHILD_RELS[0]]
    s, d = rng.integers(0, n_c, n_e), rng.integers(0, n_c, n_e)
    d[:20] = s[:20]                                       # self loops
    s[20:40], d[20:40] = d[40:60], s[40:60]               # the same pair again, reversed
    r = rng.choice([child, 3, 5, 9], n_e)
    a, c = rng.integers(0, n_p, n_pp), rng.integers(0, n_p, n_pp)
    a[:30], c[:30] = c[30:60], a[30:60]                   # repeated pairs with other weights
    return {"comp_type": np.zeros(n_c, np.int64), "paper_field": np.zeros(n_p, np.int64),
            "role_src": s, "role_dst": d, "role_rel": r,
            "pp_src": a, "pp_dst": c, "pp_w": rng.choice([1, 2], n_pp)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lookups_answer_exactly_like_the_dicts(seed):
    b = _random_graph(seed)
    lk = G.build_lookups(b)
    ce, ch, pa, pn, ci = _old_lookups(b)
    edges, kids = G._Rows(lk["ce_ptr"], lk["ce_nbr"], lk["ce_rel"]), G._Rows(lk["ch_ptr"], lk["ch_idx"])
    par, nbrs = G._Rows(lk["pa_ptr"], lk["pa_idx"]), G._NbrRows(lk["pn_ptr"], lk["pn_nbr"], lk["pn_w"])
    cites = G._PairSet(lk["ci_ptr"], lk["ci_nbr"])
    for k in range(-2, len(b["comp_type"]) + 2):
        assert edges.get(k) == ce.get(k) and edges.get(k, ()) == ce.get(k, ())      # same lists, same order
        assert kids.get(k, []) == ch.get(k, []) and par.get(k) == pa.get(k)
        assert (k in edges) == (k in ce)
    assert len(edges) == len(ce) and len(kids) == len(ch)
    for p in range(-1, len(b["paper_field"]) + 1):
        assert list(nbrs.get(p, {}).items()) == list(pn.get(p, {}).items())         # insertion order too
        for q in range(len(b["paper_field"])):
            assert ((p, q) in cites) == ((p, q) in ci)
        assert cites.of(p) == {q for (x, q) in ci if x == p}
    assert edges[10**6] == [] and nbrs[10**6] == {}                               # d[k] of a defaultdict


@pytest.mark.parametrize("seed", [0, 1])
def test_view_relations_match_the_loop(seed):
    """_role_rel writes what the per-KU loop wrote, also with KUs linked twice by different relations."""
    b = _random_graph(seed)
    lk = G.build_lookups(b)

    class GI:
        comp_edges = G._Rows(lk["ce_ptr"], lk["ce_nbr"], lk["ce_rel"])
    rng = np.random.default_rng(seed)
    for _ in range(200):
        cid = rng.choice(np.r_[-1, np.arange(300)], size=60, replace=bool(rng.integers(2))).astype(np.int64)
        old, new = np.zeros((60, 60), np.int64), np.zeros((60, 60), np.int64)
        pos = {int(c): k for k, c in enumerate(cid) if c >= 0}
        for c, k in pos.items():
            for d, r in _old_lookups(b)[0].get(c, ()):
                m = pos.get(d)
                if m is not None:
                    old[k, m] = old[m, k] = r
        _role_rel(GI, cid, new)
        np.testing.assert_array_equal(new, old)


def _build_through_get_index(tmp_path, monkeypatch):
    """get_index's real-data path (build, spill, save, load) on the synthetic records."""
    cfg = PaperAConfig().override(**CFG, synthetic=False, index_dir=str(tmp_path), out_dir=str(tmp_path))
    monkeypatch.setattr(K, "_corpus", lambda c: (iter(synthetic_records(seed=c.split_seed)), None))
    return cfg, G.get_index(cfg)


def _assert_same_index(a, b):
    for k in G.MMAP_KEYS + G.SHARED_KEYS + ("pp_src", "pp_dst", "pp_w", "paper_year", "paper_field", "train_papers",
                                            "val_papers", "test_papers", "mp_comp", "mp_kind", "field_x"):
        if k in ("comp_snippet_data", "comp_snippet_off") and not (hasattr(a, k) and hasattr(b, k)):
            continue                                   # an older index: snippets as an object array (compared below)
        np.testing.assert_array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k)), err_msg=k)
    assert [a.comp_snippet[i] for i in range(a.n_comps)] == [b.comp_snippet[i] for i in range(b.n_comps)]
    for c in range(a.n_comps):
        assert a.comp_edges.get(c) == b.comp_edges.get(c) and a.parents.get(c) == b.parents.get(c)
    for p in range(a.n_papers):
        assert list(a.paper_nbrs.get(p, {}).items()) == list(b.paper_nbrs.get(p, {}).items())
        assert list(a.paper_comps[p]) == list(b.paper_comps[p])


def test_streamed_and_saved_index_equals_the_in_memory_build(tmp_path, monkeypatch):
    cfg, built = _build_through_get_index(tmp_path, monkeypatch)
    loaded = G.get_index(cfg)                                                     # the saved index, memory-mapped
    mem = G.GraphIndex(K.build_blob(cfg, records=synthetic_records(seed=cfg.split_seed)))
    _assert_same_index(built, mem)
    _assert_same_index(loaded, mem)
    assert isinstance(loaded.comp_x, np.memmap) and not isinstance(loaded.role_src, np.memmap)
    assert loaded.texts()[0] == mem.comp_text                                    # the side file, as a plain list
    left = [f for f in os.listdir(tmp_path) if ".spill" in f or ".tmp" in f]
    assert not left, left                                                        # spill files moved or removed


def test_index_saved_before_the_lookups_still_loads_and_converts(tmp_path, monkeypatch):
    cfg, built = _build_through_get_index(tmp_path, monkeypatch)
    path = os.path.join(cfg.index_dir, f"paper_a_ku_index_{K.data_fingerprint(cfg)}.pt")
    old = {k: v for k, v in G.load_index(path).items() if k not in G.LOOKUP_KEYS + ("comp_snippet_data", "comp_snippet_off")}
    old = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in old.items()}
    old["comp_snippet"] = np.asarray([built.comp_snippet[i] for i in range(built.n_comps)], dtype=object)
    for f in os.listdir(tmp_path):
        if f.endswith(".npy"):
            os.remove(tmp_path / f)
    torch.save(old, path, pickle_protocol=4)                                      # one .pt, as the first versions
    first = G.get_index(cfg)                                                      # builds + saves the lookups
    assert all(os.path.exists(G._sidecar(path, k)) for k in G.LOOKUP_KEYS)
    _assert_same_index(first, built)
    G.convert(path)                                                               # everything to .npy files
    assert "comp_snippet" not in torch.load(path, weights_only=False)
    _assert_same_index(G.get_index(cfg), built)


def test_minimal_pair_filter_changes_nothing(monkeypatch):
    """Only validation / test papers' texts are kept as minimal-pair sources while building: same pairs as all."""
    cfg = PaperAConfig().override(**CFG)
    a = K.build_blob(cfg, records=synthetic_records(seed=cfg.split_seed))
    monkeypatch.setattr(K, "_maybe_eval", lambda *_: True)
    b = K.build_blob(cfg, records=synthetic_records(seed=cfg.split_seed))
    for k in ("mp_comp", "mp_kind", "mp_x", "mp_snippet"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert a["mp_text"] == b["mp_text"] and len(a["mp_text"]) > 0


def test_topic_donors_in_blocks_equal_the_full_matrix(monkeypatch):
    rng = np.random.default_rng(0)
    n_p, per = 400, 5
    comp_paper = np.repeat(np.arange(n_p), per)
    comp_type = rng.integers(0, 4, n_p * per)
    comp_x = rng.standard_normal((n_p * per, 32)).astype(np.float16)
    order = np.argsort(comp_paper, kind="stable")
    pc_ptr = np.r_[0, np.cumsum(np.bincount(comp_paper))]
    splits = (np.arange(0, 200), np.arange(200, 400))
    dense = K._topic_donors(splits, comp_type, comp_x, pc_ptr, order)
    monkeypatch.setattr(K, "DONOR_DENSE_MAX", 0)
    monkeypatch.setattr(K, "DONOR_BLOCK_BYTES", 4 * 200 * 7)                     # 7 rows per block
    assert K._topic_donors(splits, comp_type, comp_x, pc_ptr, order) == dense


def test_proxy_block_keeps_1024_rows_unless_they_do_not_fit(monkeypatch):
    assert G._proxy_block(10**7, "cpu", 1024) == 1024
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (80 * 2**30, 80 * 2**30))
    assert G._proxy_block(2_000_000, "cuda", 1024) == 1024                       # ~12 GB: fits as before
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (20 * 2**30, 80 * 2**30))
    b = G._proxy_block(4_500_000, "cuda", 1024)
    assert b < 1024 and b * 4_500_000 * 6 <= 0.9 * 20 * 2**30
    x = np.random.default_rng(0).standard_normal((500, 16)).astype(np.float32)
    owner = np.arange(500) // 3
    np.testing.assert_array_equal(G.build_proxy_edges(x, owner, 0.5, 5, "cpu", block=1024),
                                  G.build_proxy_edges(x, owner, 0.5, 5, "cpu", block=16))


def test_text_side_file_streams_as_a_list(tmp_path):
    texts = ["plain", "ünïcödé ✓", "", "lone surrogate \udc80 kept", "x" * 10000]
    sp = K._StrSpill(str(tmp_path / f"t{K.SPILL_SUFFIX}"))
    for t in texts:
        sp.append(t)
    st = K._SpilledTexts(*sp.finish())
    with open(tmp_path / "t.pkl", "wb") as f:
        p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL); p.fast = True
        p.dump({"comp_text": G._TextStream(st)})
    with open(tmp_path / "t.pkl", "rb") as f:
        assert pickle.load(f)["comp_text"] == texts
    data, off = K.strings_to_arrays(texts)
    assert [G._Strings(data, off)[i] for i in range(len(texts))] == texts
