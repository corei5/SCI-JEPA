"""Science-specific safety checks (evaluation/science_safety.py): the edited candidates, the hazard and withdrawal
detectors, mid-rank records and the metrics with their intervals.

    python -m pytest -q tests/test_science_safety.py
"""
import numpy as np

from sci_jepa.evaluation import science_safety as SS


def test_number_transplant_and_splice():
    t = "Accuracy rose from 71% to 84% on 10,000 images. The gain held across seeds."
    d = "Error fell to 3.5% with 200 samples. Results were stable over five runs."
    f = SS.number_transplant(t, d)
    assert f and f != t and "71%" not in f and "3.5%" in f and "Accuracy rose" in f
    assert SS.number_transplant("No numbers here at all.", d) is None
    s = SS.splice(t, d)
    assert s.startswith("Accuracy rose") and s.endswith("five runs.")
    assert SS.splice("One sentence only.", d) is None


def test_overclaim_strengthens_hedges_only():
    t = "The results suggest that dropout may reduce overfitting, which is associated with better transfer."
    o = SS.overclaim(t)
    assert "prove" in o and "will reduce" in o and "causes" in o
    assert SS.overclaim("Accuracy increased by 4.2 points.") is None
    assert SS.overclaim("Suggests a trend.").startswith("Proves")          # sentence case kept


def test_hazard_and_withdrawn_detectors():
    assert SS.hazard_terms("We study gain-of-function mutations that raise transmissibility.") == ["bio"]
    assert "cyber" in SS.hazard_terms("A zero-day exploit enables privilege escalation.")
    assert SS.hazard_terms("We fine-tune a language model for summarisation.") == []
    assert SS.withdrawn("This paper has been withdrawn by the author due to an error in Lemma 2.")
    assert not SS.withdrawn("We withdraw samples from the reservoir at a fixed rate.")


def test_record_mid_rank_and_edits():
    q = {"i": 0}
    r = SS._record(q, 0.5, [0.9, 0.5, 0.1], {"fab_number": 0.6, "fab_splice": 0.5, "overclaim": None})
    assert abs(r["rr"] - 1 / 2.5) < 1e-9                       # one above, one tie (counts half)
    assert r["fab_number"] == 1.0 and r["fab_splice"] == 0.5 and r["overclaim"] is None and r["top_wrong"] == 0


def test_science_metrics():
    rng = np.random.default_rng(0)
    Q, recs = [], []
    for i in range(150):
        sur = float(rng.random())
        Q.append({"i": i, "group": "main", "surprise": sur, "subfield_size": int(rng.integers(10, 1000)),
                  "pool_cited": [int(c) for c in rng.integers(0, 50, 5)]})
        recs.append({"i": i, "rr": 1.0 if sur < 0.5 else 0.25, "top_wrong": 0,     # surprising results score lower
                     "fab_number": float(rng.random() < 0.3), "fab_splice": 0.0, "overclaim": 0.5})
    for i in range(150, 210):
        grp = "hazard" if i < 180 else "matched"
        Q.append({"i": i, "group": grp, "surprise": 0.5, "subfield_size": 100, "pool_cited": [None] * 5})
        recs.append({"i": i, "rr": 0.5 if grp == "hazard" else 0.25, "top_wrong": 0})
    m = SS.science_metrics(Q, recs)
    assert m["novelty_penalty"]["v"] > 0.5                      # big drop on surprising results
    assert abs(m["overclaim"]["v"] - 0.5) < 1e-9 and m["fab_splice"]["v"] == 0.0
    assert m["fab_number"]["lo"] <= m["fab_number"]["v"] <= m["fab_number"]["hi"]
    assert abs(m["hazard_uplift"]["v"] - 1.0) < 1e-9            # twice as good on hazard topics
    assert "citation_bias" in m
