"""Resuming a training run from its checkpoint gives exactly the run that was never interrupted.

Synthetic data, the tiny model, CPU, a few seconds of training plus the final evaluation (about a minute):
    python -m pytest -q tests/test_resume.py
"""
import os

import numpy as np
import pytest

from sci_jepa.config import SIZES, PaperAConfig
from sci_jepa.data.graph_index import get_index
from sci_jepa.training import train as T

STEPS = 12
FAST = dict(synthetic=True, embedder="hash", device="cpu", time_T=2016, test_open_ended=False, proxy_tau=0.6,
            steps=STEPS, epochs=0.0, batch_size=16, log_every=1, warmup=5, eval_instances=48, probe_papers=15,
            seeds=[0], eval_tasks=False, eval_worldview=False, save_model=False, prefetch=2, ckpt_every=5)


class Stop(Exception):
    pass


def _jepa_losses(m):
    return np.array([h["jepa"] for h in m["history"]])


@pytest.mark.slow
def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    cfg = PaperAConfig().override(**SIZES["tiny"], **FAST, out_dir=str(tmp_path))
    gi = get_index(cfg)
    full = T.train_one(cfg, gi, 0, log=lambda *_: None)                 # never interrupted, no checkpoints

    ckpt = str(tmp_path / "run.pt")
    calls, real = {"n": 0}, T.L.jepa_loss
    def dies_at_step_8(*a, **k):                                        # the job "hits its time limit"
        calls["n"] += 1
        if calls["n"] == 9:
            raise Stop()
        return real(*a, **k)
    monkeypatch.setattr(T.L, "jepa_loss", dies_at_step_8)
    with pytest.raises(Stop):
        T.train_one(cfg, gi, 0, log=lambda *_: None, ckpt_path=ckpt)
    assert os.path.exists(ckpt), "no checkpoint after step 4 (ckpt_every=5)"
    monkeypatch.setattr(T.L, "jepa_loss", real)

    resumed = T.train_one(cfg, gi, 0, log=lambda *_: None, ckpt_path=ckpt)   # the next job
    assert len(resumed["history"]) == len(full["history"]) == STEPS
    np.testing.assert_allclose(_jepa_losses(resumed), _jepa_losses(full), rtol=0, atol=1e-6)
    assert resumed["retrieval_balanced"]["MRR"] == pytest.approx(full["retrieval_balanced"]["MRR"], abs=1e-9)


@pytest.mark.slow
def test_checkpoint_of_other_settings_is_ignored(tmp_path):
    cfg = PaperAConfig().override(**SIZES["tiny"], **FAST, out_dir=str(tmp_path))
    gi = get_index(cfg)
    ckpt = str(tmp_path / "run.pt")
    T.train_one(cfg, gi, 0, log=lambda *_: None, ckpt_path=ckpt)       # leaves a "training finished" checkpoint
    other = T.train_one(cfg.override(lr=cfg.lr * 2), gi, 0, log=lambda *_: None, ckpt_path=ckpt)
    assert len(other["history"]) == STEPS                                # trained from the start, not resumed


def test_llm_arms_expand_one_per_model():
    cfg = PaperAConfig().override(llm_models=("Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"))
    arms = T.expand_llm_arms([("bm25", {"baseline": "bm25"}), ("LLM-Reasoning", {"baseline": "llm"})], cfg)
    assert [t for t, _ in arms] == ["bm25", "LLM-Reasoning@Qwen2.5-7B-Instruct", "LLM-Reasoning@Mistral-7B-Instruct-v0.3"]
    assert arms[2][1]["llm_model"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert T.expand_llm_arms([("LLM-Reasoning", {"baseline": "llm"})], PaperAConfig()) == [("LLM-Reasoning", {"baseline": "llm"})]


@pytest.mark.slow
def test_batch_workers_independent_of_worker_count_and_resumable(tmp_path, monkeypatch):
    """Parallel batch building (batch_workers): the same run with 1 or 2 worker processes, and resumed from a
    checkpoint, trains on exactly the same batches."""
    cfg = PaperAConfig().override(**SIZES["tiny"], **FAST, out_dir=str(tmp_path), batch_workers=1)
    gi = get_index(cfg)
    one = T.train_one(cfg, gi, 0, log=lambda *_: None)
    two = T.train_one(cfg.override(batch_workers=2), gi, 0, log=lambda *_: None)
    np.testing.assert_allclose(_jepa_losses(two), _jepa_losses(one), rtol=0, atol=1e-6)

    ckpt = str(tmp_path / "run.pt")
    calls, real = {"n": 0}, T.L.jepa_loss
    def dies_at_step_8(*a, **k):
        calls["n"] += 1
        if calls["n"] == 9:
            raise Stop()
        return real(*a, **k)
    monkeypatch.setattr(T.L, "jepa_loss", dies_at_step_8)
    with pytest.raises(Stop):
        T.train_one(cfg.override(batch_workers=2), gi, 0, log=lambda *_: None, ckpt_path=ckpt)
    monkeypatch.setattr(T.L, "jepa_loss", real)
    resumed = T.train_one(cfg.override(batch_workers=2), gi, 0, log=lambda *_: None, ckpt_path=ckpt)
    np.testing.assert_allclose(_jepa_losses(resumed), _jepa_losses(one), rtol=0, atol=1e-6)


@pytest.mark.slow
def test_dynamic_shards_run_every_arm_once(tmp_path, monkeypatch):
    """Dynamic sharding (SCI_JEPA_CLAIM=1): two shards together run every arm x seed exactly once, and the merge
    (no shard) loads them all."""
    monkeypatch.setenv("SCI_JEPA_CLAIM", "1")
    cfg = PaperAConfig().override(**SIZES["tiny"], **dict(FAST, steps=4, seeds=[0, 1]), out_dir=str(tmp_path))
    gi = get_index(cfg)
    arms = [("a", {}), ("b", {"lr": cfg.lr * 2})]
    first, _ = T.run_experiment(cfg, gi, arms=arms, shard=(0, 2), name="X", log=lambda *_: None)
    second, _ = T.run_experiment(cfg, gi, arms=arms, shard=(1, 2), name="X", log=lambda *_: None)
    assert sum(len(v) for v in first.values()) == 4          # the first shard took every run ...
    claims = sorted(os.listdir(tmp_path / "X" / "claims"))
    assert claims == ["a_seed0", "a_seed1", "b_seed0", "b_seed1"]
    merged, summ = T.run_experiment(cfg, gi, arms=arms, name="X", log=lambda *_: None)
    assert {t: len(v) for t, v in merged.items()} == {"a": 2, "b": 2}
    assert second == {} or all(m.get("tag") for v in second.values() for m in v)   # ... the second only loaded
