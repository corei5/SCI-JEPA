"""Safety checks (evaluation/safety.py): flag limits, paraphrases that keep the meaning, calibration error, and
the report from result files.

    python -m pytest -q tests/test_safety.py
"""
import json

import numpy as np

from sci_jepa.data.ku_schema import counterfactuals
from sci_jepa.evaluation import safety as S


def test_flags():
    assert S.flag("robustness", 0.70) == "pass" and S.flag("robustness", 0.60) == "warn"
    assert S.flag("robustness", 0.52) == "fail"
    assert S.flag("label_gap", 0.05) == "pass" and S.flag("label_gap", 0.90) == "fail"
    assert S.flag("ece", None) == "n/a"


def test_paraphrase_keeps_meaning_words():
    t = "The study found that accuracy increased by 4.2% and the authors utilized ResNet-50."
    p = S.paraphrase(t)
    assert p and p != t
    for w in ("increased", "4.2%", "ResNet-50"):                # direction, number and terms untouched
        assert w in p
    assert not any(e == p for _, e in counterfactuals(t))       # a paraphrase is never a counterfactual edit
    assert S.paraphrase("Nothing to reword here at all.") is None


def test_ece():
    ece, rel = S._ece([0.9] * 10, [1] * 9 + [0])
    assert abs(ece) < 1e-9 and rel[0]["n"] == 10
    ece, _ = S._ece([0.9] * 10, [0] * 10)
    assert abs(ece - 0.9) < 1e-9


def test_report_from_results(tmp_path):
    exp = tmp_path / "BENCH"; exp.mkdir()
    task = lambda r: {"mp_pair": r, "mp_by_kind": {"direction": r}, "ratio": 10.0, "n": 100,
                      "by_domain": {"A": {"ratio": 10.0, "n": 50}, "B": {"ratio": 6.0, "n": 50}},
                      "by_period": {"after_cutoff": {"ratio": 10.0, "n": 100}}, "worst_domain": {"domain": "B", "ratio": 6.0}}
    for s, r in ((0, 0.52), (1, 0.54)):
        m = {"tag": "jepa+rgat", "family": "gnn", "seed": s,
             "tasks": {k: task(r) for k in S.RETRIEVAL} | {"consistency": {"recall_supports": 0.9, "recall_contradicts": 0.8}},
             "partition_probe": {"probe_acc": 0.99}, "structure": {"with": 19.3, "without": 19.3},
             "similarity": {"false_accept": 0.0}}
        json.dump(m, open(exp / f"jepa+rgat_seed{s}.json", "w"))
    res = S.check_results(str(exp))
    r = res["jepa+rgat"]
    assert r["n"] == 2 and abs(r["robustness"] - 0.53) < 1e-9
    assert r["flags"]["robustness"] == "fail" and r["flags"]["identity"] == "fail" and r["flags"]["structure"] == "fail"
    assert r["contamination"] is None and r["flags"]["contamination"] == "n/a"     # every test paper is later
    assert abs(r["domain_gap"] - 0.25) < 1e-9
    rep = S.write(str(exp), res, {}, str(tmp_path))
    assert (exp / "safety.md").exists() and (tmp_path / "safety.json").exists() and rep["systems"]
    assert np.isfinite(r["label_gap"])
