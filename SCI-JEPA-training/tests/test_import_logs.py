"""Unit tests of scripts/import_logs.py (earlier runs rebuilt from SLURM logs for the dashboard)."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "import_logs", os.path.join(os.path.dirname(__file__), "..", "scripts", "import_logs.py"))
il = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(il)

FIRST = """results : /nfs/x/runs/bench_S
context              N = 8 papers per training instance
KU graph   each paper -> ~22 KUs (ku_mode='section')
at most 2 epochs over 1000 hidden KUs = 200 steps of 10

[paper_a] BENCH | bm25 | seed 0
  -> balanced MRR 0.7000 (chance 0.0500) | standard MRR 0.4000 (chance 1.00e-03) | chi nan | probe nan | rel-sens nan | inv nan | audit flagged nan

[paper_a] BENCH | jepa+gine | seed 0
  step=0 epoch=0 phase=mixed jepa=1.3 inverse_probe=2.8 flagged=0.5 sec=7.2
  step=50 epoch=0.5 phase=mixed jepa=0.4 inverse_probe=0.3 flagged=0.4 sec=90
  [epoch 1] val balanced MRR 0.5000 (chance 0.0500) acc(sim>=0.80) 0.010 (top-1 0.010, false accept 0.000) train jepa 0.5 types learned 9/10 reasoning_index 3.000
"""
RESUMED = """results : /nfs/x/runs/bench_S
at most 2 epochs over 1000 hidden KUs = 200 steps of 10
[paper_a] BENCH | jepa+gine | seed 0
  [resume] /nfs/x/ckpt.pt: continuing after step 99 of 200 (epoch 1, 1 h trained so far)
  step=150 epoch=1.5 phase=mixed jepa=0.2 inverse_probe=0.1 flagged=0.4 sec=200
  [epoch 2] val balanced MRR 0.8000 (chance 0.0500) acc(sim>=0.80) 0.020 (top-1 0.020, false accept 0.000) train jepa 0.2 types learned 10/10 reasoning_index 8.000
  [best] testing the weights of epoch 2 (validation reasoning_index 8.000)
  -> balanced MRR 0.7900 (chance 0.0500) | standard MRR 0.5000 (chance 1.00e-03) | chi 0.600 | probe 0.800 | rel-sens 0.010 | inv 0.990 | audit flagged 0.40
[paper_a] BENCH | jepa+gps | seed 0  [worker gpu 1]
[paper_a] BENCH | jepa+gps | seed 0: already finished, loaded from /nfs/x
"""


def test_runs_rebuilt_and_resume_joined(tmp_path):
    logs = tmp_path / "log"
    logs.mkdir()
    (logs / "bench_S_38100_shard0.log").write_text(FIRST)
    (logs / "bench_S_38101_shard0.log").write_text(RESUMED)
    dash = tmp_path / "dash"
    assert il.main([str(logs), str(dash)]) == 0
    index = json.loads((dash / "runs_archive.json").read_text())
    assert [r["name"] for r in index] == ["log-bench_S__BENCH__bm25__seed0__job38100",
                                          "log-bench_S__BENCH__jepa+gine__seed0__job38100"]
    bm25 = json.loads((dash / index[0]["path"]).read_text())
    assert bm25["status"] == "done" and bm25["steps_total"] == 0 and bm25["config"]["baseline"] == "bm25"
    assert bm25["final"]["retrieval_balanced"] == {"MRR": 0.7, "chance_MRR": 0.05} and bm25["final"]["inverse_acc"] is None
    run = json.loads((dash / index[1]["path"]).read_text())
    assert run["status"] == "done" and run["imported"]["jobs"] == ["38100", "38101"]
    assert [h["step"] for h in run["history"]] == [0, 50, 150] and [e["epoch"] for e in run["epoch_results"]] == [1, 2]
    assert run["config"]["n_subgraphs"] == 8 and run["steps_total"] == 200 and run["final"]["best_epoch"] == 2
    assert run["evals"][-1]["bal_MRR"] == 0.8 and run["final"]["target_space"]["chi"] == 0.6

    assert il.main([str(logs), str(dash)]) == 0                 # re-import replaces, never duplicates
    assert len(json.loads((dash / "runs_archive.json").read_text())) == 2
