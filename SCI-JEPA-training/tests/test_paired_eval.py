"""Paired benchmark v2 (evaluation/paired_eval.py): mid-rank ties, AUC, the tuned threshold and the paired-bootstrap
summary with the best-JEPA-vs-strongest-comparison verdict.

    python -m pytest -q tests/test_paired_eval.py
"""
import numpy as np

from sci_jepa.evaluation import paired_eval as PE


def test_mid_rank_and_pair_win():
    assert PE.mid_rank(0.5, [0.9, 0.5, 0.1]) == 2.5          # one above, one tie (half)
    assert PE.mid_rank(0.5, [0.5] * 100) == 51.0            # a collapsed model ranks in the middle, not first
    assert PE.pair_win(0.5, [0.4, 0.5, 0.6]) == [1.0, 0.5, 0.0]
    assert abs(PE.chance_rr(1) - 1.0) < 1e-12


def test_auc_balacc_threshold():
    y = [0, 0, 1, 1]
    assert PE.auc(y, [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert PE.auc(y, [0.5] * 4) == 0.5 and PE.auc([1, 1], [0.1, 0.2]) is None
    assert PE.bal_acc([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400); s = y * 2.0 + rng.normal(0, 0.3, 400) + 5   # separable around 6
    t = PE.best_threshold(y, s)
    assert 5.3 < t < 6.7 and PE.bal_acc(y, (s > t).astype(int)) > 0.95


def _data(nq=60, npairs=80):
    Q = [{"i": i, "task": PE.RANK_TASKS[i % 3], "paper": i} for i in range(nq)]
    P = [{"y": int(k % 4 == 0), "paper": k // 2} for k in range(npairs)]
    return {"questions": Q, "pairs": P, "train_pairs": [], "interventions": [{"q": 0}, {"q": 3}]}


def test_summarise_paired_verdict():
    D = _data()
    rec = lambda rr, mp: [{"rr": rr, "chance": 0.05, "mp": mp, "mp_all": mp} for _ in D["questions"]]
    cons = lambda acc: [{"score": float(p["y"]), "yhat": p["y"] if k < acc * len(D["pairs"]) else 1 - p["y"]}
                        for k, p in enumerate(D["pairs"])]
    S = {"jepa_a": {"seeds": [0, 1], "rank": [rec(0.8, 0.6), rec(0.8, 0.6)], "cons": [cons(1.0), cons(1.0)],
                    "int": [[{"rr_before": 0.1, "rr_after": 0.3, "toward": 1.0}] * 2] * 2, "family": "gnn"},
         "jepa_b": {"seeds": [0], "rank": [rec(0.9, 0.6)], "cons": [cons(1.0)], "int": [], "family": "jepa"},
         "bm25": {"seeds": [0], "rank": [rec(0.5, 0.5)], "cons": [cons(1.0)], "int": [], "family": "comparison"}}
    out, cmp = PE.summarise(D, S, val_index={"jepa_a": 9.7, "jepa_b": 9.1})
    a = out["jepa_a"]
    assert abs(a["point"]["average"] - (0.8 * 3 + 1.0) / 4) < 1e-9 and a["n_seeds"] == 2
    assert a["ci"]["average"][0] <= a["point"]["average"] <= a["ci"]["average"][1]
    assert abs(a["intervention"]["gain"] - 0.2) < 1e-9
    assert cmp["best_jepa"] == "jepa_a"                      # chosen on validation, not on test (jepa_b is higher on test)
    assert cmp["strongest_comparison"] == "bm25" and cmp["metrics"]["average"]["verdict"] == "JEPA better"
