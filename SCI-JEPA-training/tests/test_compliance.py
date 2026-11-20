"""Regulation and ethics assessment (evaluation/compliance.py): the personal-data detectors and the assessment from
synthetic evidence (statuses follow the evidence).

    python -m pytest -q tests/test_compliance.py
"""
from sci_jepa.evaluation import compliance as C


def test_detectors():
    assert C.EMAIL_RE.search("write to jane.doe@uni-x.de today")
    assert not C.EMAIL_RE.search("an Intel Xeon CPU@3.30GHz") and not C.EMAIL_RE.search("i5 @ 3.3GHz")
    assert C.ORCID_RE.search("ORCID 0000-0002-1825-0097") and C.ETAL_RE.search("as shown by Smith et al. (2020)")


def _ev(**kw):
    E = {"safety": {"systems": {"jepa+rgat": {"family": "gnn", "flags": {"robustness_ci": "fail", "domain_gap_ci": "pass"},
                                              "card": {"selective_gain": {"v": -0.1}, "injection_asr": {"v": 0.5},
                                                       "robustness_ci": {"v": 0.55}}}}},
         "science": {}, "infra": {}, "paired": {}, "read_cols": ["key_results", "paper_id"], "personal_cols_read": [],
         "config": {}, "split_manifest": True, "dataset_license": "cc-by-4.0", "train_flop_max": 9e17, "params_max": 4.6e7,
         "energy": None, "docs": {"model_card": False, "safety_report": True}, "pii": {"kus": 100, "email": 0, "orcid": 0, "et_al": 3},
         "gov": {k: {"path": p, "exists": False, "signed": False} for k, p in C.GOV.items()}}
    E.update(kw)
    return {(r["framework"], r["ref"]): r for r in C.assess(E)}


def test_assessment_follows_evidence():
    R = _ev()
    assert R[("gdpr", "Art. 5(1)(c)")]["status"] == "met"
    assert R[("ai_act", "Art. 3(63), 51-55")]["status"] == "n/a"          # far below the GPAI threshold
    assert R[("hleg", "1")]["status"] == "partly"                        # confidence not usable for oversight
    assert R[("hleg", "2")]["status"] == "gap" and R[("hleg", "4")]["status"] == "partly"   # no model card
    R = _ev(personal_cols_read=["source_authors"], pii={"kus": 100, "email": 4, "orcid": 0, "et_al": 0})
    assert R[("gdpr", "Art. 5(1)(c)")]["status"] == "gap" and R[("gdpr", "Art. 4(1), 9")]["status"] == "partly"


def test_signoff_moves_manual_to_ready_to_met(tmp_path):
    gov = lambda ex, sg: {k: {"path": p, "exists": ex, "signed": sg} for k, p in C.GOV.items()}
    assert _ev()[("gdpr", "Art. 30")]["status"] == "manual"
    assert _ev(gov=gov(True, False))[("gdpr", "Art. 30")]["status"] == "ready"
    R = _ev(gov=gov(True, True))
    assert R[("gdpr", "Art. 30")]["status"] == "met" and R[("ai_act", "Art. 4")]["status"] == "met"
    p = tmp_path / "d.md"
    p.write_text("x" + C.SIGN)
    assert not C._signed(str(p))
    p.write_text("x\nSigned off by: Dr. A. Example\n")
    assert C._signed(str(p))


def test_abstention_threshold():
    E = {"safety": {"models": {"jepa+rgat_seed0": {"calibration": {"reliability": [
        {"conf": 0.55, "acc": 0.6, "n": 100}, {"conf": 0.75, "acc": 0.9, "n": 100}, {"conf": 0.95, "acc": 0.99, "n": 300}]}}}}}
    ab = C._abstain(E, "jepa+rgat")
    assert ab["threshold"] == 0.6 and ab["accuracy"] >= 0.95 and 0 < ab["coverage"] < 1


def test_release_redaction_and_check(tmp_path):
    from sci_jepa.release import check, redact
    assert redact("contact jane@uni.de or +49 511 762 1234, ORCID 0000-0002-1825-0097") == \
        "contact <email> or <phone>, ORCID <orcid>"
    for n in ("NOTICE", "LICENSE", "README.md"):
        (tmp_path / n).write_text("ok")
    (tmp_path / "ku_texts.jsonl").write_text('{"key_results": "fine"}\n')
    assert check(str(tmp_path))["ok"]
    (tmp_path / "ku_texts.jsonl").write_text('{"text_raw": "full paper"}\n{"x": "mail me@x.org"}\n')
    assert not check(str(tmp_path))["ok"]
