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


def test_keep_steps_are_evenly_spaced_and_end_at_the_last_step():
    assert T.keep_steps(331720, 10) == [33171, 66343, 99515, 132687, 165859, 199031, 232203, 265375, 298547, 331719]
    assert T.keep_steps(5, 10) == [0, 1, 2, 3, 4]                       # fewer steps than kept checkpoints
    assert T.keep_steps(100, 0) == []
    assert "keep_ckpts" not in T._ckpt_key(PaperAConfig().override(keep_ckpts=10), 0)   # old checkpoints still resume


def test_checkpoint_keyed_with_a_now_ignored_setting_still_resumes(tmp_path):
    """Checkpoints written before "arms" joined _CKPT_IGNORE carry it in their key; they must still resume."""
    import torch
    key = T._ckpt_key(PaperAConfig(), 0)
    path = str(tmp_path / "old.pt")
    torch.save({"version": T.CKPT_VERSION, "key": {**key, "arms": "('reference',)", "keep_ckpts": "0"}}, path)
    assert T._load_ckpt(path, key, lambda *_: None) is not None
    torch.save({"version": T.CKPT_VERSION, "key": {**key, "lr": "1.0"}}, path)          # a real difference
    assert T._load_ckpt(path, key, lambda *_: None) is None


def test_resume_file_round_trips_every_array_kind(tmp_path):
    """_save_ckpt writes arrays as raw bytes outside pickle: every dtype, shape and sharing must come back exactly."""
    import random

    import torch
    shared = torch.randn(3, 4)
    payload = {"model": {"w": torch.randn(5, 7), "bf": torch.randn(4, 3).bfloat16(), "half": torch.randn(2).half(),
                         "i": torch.arange(6).reshape(2, 3), "mask": torch.tensor([True, False]),
                         "step": torch.tensor(17.0), "empty": torch.zeros(0, 4),
                         "strided": torch.randn(4, 6)[:, ::2]},                    # non-contiguous
               "a": shared, "b": shared,
               "rng": {"numpy": np.random.get_state(), "python": random.getstate(), "torch": torch.get_rng_state()},
               "sampler": {"queue": np.arange(10, dtype=np.int64)[::-1], "scalar": np.array(2.5),
                           "none": np.zeros((0,), np.float32), "obj": np.array(["x", None], dtype=object)},
               "history": [{"jepa": 0.5}], "version": T.CKPT_VERSION}
    path = str(tmp_path / "ck.pt")
    T._save_ckpt(path, payload)
    back = T._read_ckpt(path)
    for k, v in payload["model"].items():
        assert back["model"][k].dtype == v.dtype and torch.equal(back["model"][k], v), k
    assert back["a"] is back["b"] and torch.equal(back["a"], shared)
    assert torch.equal(back["rng"]["torch"], payload["rng"]["torch"]) and back["rng"]["python"] == payload["rng"]["python"]
    for x, y in zip(back["rng"]["numpy"], payload["rng"]["numpy"]):
        assert np.array_equal(x, y) if isinstance(y, np.ndarray) else x == y
    for k, v in payload["sampler"].items():
        assert back["sampler"][k].dtype == v.dtype and back["sampler"][k].shape == v.shape
        assert np.array_equal(back["sampler"][k], v), k
    assert back["history"] == payload["history"]
    with open(path, "r+b") as f:                                          # a write cut short: unreadable, not wrong
        f.truncate(os.path.getsize(path) - 8)
    assert T._load_ckpt(path, {}, lambda *_: None) is None


@pytest.mark.slow
def test_kept_weights_load_into_the_model(tmp_path):
    from safetensors import safe_open
    cfg = PaperAConfig().override(**SIZES["tiny"], **FAST, out_dir=str(tmp_path), keep_ckpts=4)
    gi = get_index(cfg)
    T.train_one(cfg, gi, 0, log=lambda *_: None, save_dir=str(tmp_path / "model"))
    kept = sorted(os.listdir(tmp_path / "model" / "checkpoints"))
    assert kept == ["step0000003.safetensors", "step0000006.safetensors", "step0000009.safetensors",
                    "step0000012.safetensors"]
    with safe_open(str(tmp_path / "model" / "checkpoints" / kept[-1]), "pt") as f:
        assert f.metadata()["step"] == "12" and set(f.keys()) == set(T.PaperAModel(cfg).state_dict())


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
