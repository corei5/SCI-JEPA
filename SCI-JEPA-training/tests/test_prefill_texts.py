"""The paper_masked text prefill (ku_graph.prefill_context_texts) writes the build's chunks: same texts, same hashes.

A build with ku_mode="paper_masked" on synthetic records, with the hash embedder behind the resumable cache and
small chunks; then the prefill over the same records must find every one of the build's chunks in the cache, and a
prefill into an empty cache must write exactly the files the build then loads. A few seconds on CPU:
    python -m pytest -q tests/test_prefill_texts.py
"""
import os

import numpy as np

from sci_jepa.config import PaperAConfig
from sci_jepa.data import ku_graph as K
from sci_jepa.data.synthetic import synthetic_records

CHUNK = 64


def _setup(tmp_path, monkeypatch, cache):
    cfg = PaperAConfig().override(embedder="hash", device="cpu", ku_mode="paper_masked", time_T=2016,
                                  test_open_ended=False, out_dir=str(tmp_path))
    recs = list(synthetic_records(n_papers=120, seed=0))
    real = K.ResumableEmbedder.__init__

    def small_chunks(self, emb, root, chunk=8192):
        real(self, emb, root, CHUNK)
    monkeypatch.setattr(K.ResumableEmbedder, "__init__", small_chunks)
    monkeypatch.setattr(K, "embed_cache_dir", lambda cfg: str(cache))
    monkeypatch.setattr(K, "_corpus", lambda cfg: (iter(recs), None))
    return cfg, recs


def _build(cfg, recs, cache):
    emb = K.ResumableEmbedder(K.make_embedder(cfg, "cpu"), str(cache))
    parts = []
    for rec in recs:
        got, _, _ = K._paper_kus(rec, cfg)
        if got is not None:
            parts.append(got[:2])
    n = len(parts)
    blob = K.build_context_texts(cfg, emb, parts, ["f"] * n, ["s"] * n, (np.arange(n), [], []))
    return blob, emb


def test_prefill_finds_every_build_chunk(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    cfg, recs = _setup(tmp_path, monkeypatch, cache)
    blob, _ = _build(cfg, recs, cache)
    n_chunks = -(-len(blob["ctx_x"]) // CHUNK)
    assert n_chunks > 3
    built = set(os.listdir(cache))
    K.prefill_context_texts(cfg, 0, 1, start=n_chunks)                 # check only: nothing to encode
    out = capsys.readouterr().out
    assert f"{n_chunks} found in the cache, 0 not" in out
    assert set(os.listdir(cache)) == built


def test_prefill_writes_what_the_build_loads(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cfg, recs = _setup(tmp_path, monkeypatch, cache)
    for shard in range(2):                                             # two helpers, every chunk from 0
        K.prefill_context_texts(cfg, shard, 2, start=0)
    pre = set(os.listdir(cache))
    blob, emb = _build(cfg, recs, cache)
    assert emb.reused == len(blob["ctx_x"])                            # every context text loaded, none encoded
    assert set(os.listdir(cache)) == pre
    ref, _ = _build(cfg, recs, tmp_path / "fresh")                     # and the vectors are the build's own
    np.testing.assert_array_equal(blob["ctx_x"], ref["ctx_x"])
    import json                                                         # the dashboard row followed the encoding
    p = json.load(open(tmp_path / "infra" / "progress" / "KU_index_build.json"))
    assert p["task"] == "encoding paper texts" and p["done"] == p["total"] == len(ref["ctx_x"])
