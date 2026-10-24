"""
CPU smoke test on the synthetic graph: the invariants checked directly, then a few
representative arms (one per experiment family, all three KU modes, the reasoning-task
matrix) train for a few steps and evaluate. About a minute.

    python -m sci_jepa.smoke_test           # small (default)
    python -m sci_jepa.smoke_test --full    # every arm of every experiment
"""
import math
import sys

import numpy as np
import torch

from sci_jepa.config import EXPERIMENTS, SIZES, PaperAConfig
from sci_jepa.data.graph_index import HUB, get_index
from sci_jepa.data.instances import Sampler, collate
from sci_jepa.models.model import PaperAModel, to_device
from sci_jepa.training.train import train_one

FAST = dict(synthetic=True, embedder="hash", device="cpu", time_T=2016, test_open_ended=False, proxy_tau=0.6, steps=12, batch_size=16, log_every=1000, warmup=5,
            eval_instances=48, probe_papers=15, seeds=[0], eval_tasks=False, task_eval_instances=32, eval_worldview=False, save_model=False,
            epochs=0.0)                                   # fixed 12 steps instead of the paper's 10 epochs
# the small test: one arm per experiment family (tag, with the reasoning tasks on or off)
SMALL = [("BENCH", "text_baseline", True), ("E10", "N=8", False), ("E11", "inv=0.3", False), ("E12", "reg=within", False),
         ("E4", "frozen+census", False), ("E5", "contrast=on", False), ("E15", "only=result", False),
         ("E17", "section", True), ("E17", "paper_masked", True), ("E18", "pm_only=claim", False),
         ("E19", "world_model", True), ("E20", "graph_jepa+world", True), ("APPX_EVAL", "reference_old_protocol", False), ("E1", "shacl=tight", False),
         ("SPEED", "fast+multi4", False), ("SPEED", "motif+world fast+multi4", True),
         ("BENCH", "bm25", True), ("BENCH", "supervised", True), ("BENCH", "jepa+rgat", True),
         ("BENCH", "jepa+gine", True), ("BENCH", "jepa+gps", True), ("BENCH", "graph_jepa_gine", False)]
# comparison systems outside BENCH (minilm: a small encoder already in the local cache, tests the encoder path)
EXTRA = [("baseline", "minilm", dict(baseline="minilm"), True),
         ("baseline", "LLM-Reasoning (tiny LM)", dict(baseline="llm", llm_model="sshleifer/tiny-gpt2", llm_allow_cpu=True,
                                                      llm_eval_instances=24, llm_batch=8), True)]


def check_invariants(gi, cfg):
    cfg = cfg.override(task_mask_p=0.0)                  # the invariants test the masking policy alone
    s = Sampler(gi, cfg.override(target_types=("claim",)), "train", 0)
    leaky = Sampler(gi, cfg.override(target_types=("claim",), mask_policy="token"), "train", 0)
    safe_i, leak_i = s.sample(64), leaky.sample(64)
    kids_vis = lambda insts: np.mean([np.isin(i["ctx"]["comp"], gi.children.get(i["target_comp"], [])).any()
                                      for i in insts])
    assert kids_vis(safe_i) == 0.0 and kids_vis(leak_i) > 0.9, (kids_vis(safe_i), kids_vis(leak_i))
    sim = lambda insts: np.mean([i["audit"]["max_ctx_sim"] for i in insts])
    print(f"  [ok] leak-safe masking: claim's evidence/implications visible in "
          f"{kids_vis(leak_i):.0%} (token only) -> {kids_vis(safe_i):.0%}; "
          f"max ctx sim {sim(leak_i):.3f} -> {sim(safe_i):.3f}")
    sib = Sampler(gi, cfg.override(target_types=("method",), mask_policy="token+edges+children+siblings"), "train", 0)
    proc = gi.comp_type == __import__("sci_jepa.data.ku_schema", fromlist=["x"]).COMP_TYPES.index("procedure")
    for i in sib.sample(16):
        own = i["ctx"]["slot"] == 0
        assert not np.any(proc[i["ctx"]["comp"][own & (i["ctx"]["comp"] >= 0)]]), "procedure visible"
    print("  [ok] siblings policy hides the masked method's procedure")
    assert gi.n_papers == 300 and len(gi.papers("test")) > 0
    names = set(gi.field_names)
    print(f"  [ok] KU graph from normalized-schema records: {gi.summary()['kus_per_paper']} KUs/paper, "
          f"{len(names)} fields, splits {gi.summary()['splits']}")
    out = gi.summary()["build"]["ku_outcomes"]
    assert out.get("contradicting_evidence:null_statement", 0) > 0 and out.get("figures:placeholder", 0) == 0
    print(f"  [ok] null statements filtered: {out.get('contradicting_evidence:null_statement')}, "
          f"placeholders: {out.get('contradicting_evidence:placeholder', 0)}")

    for inst in s.sample(32):
        c, m = inst["ctx"], inst["mask_pos"]
        assert np.all(c["x"][m] == 0), "masked token still carries features"
        assert not np.any(c["comp"][~inst["is_mask"]] == inst["target_comp"]), "target visible twice"
        assert set(np.unique(c["rel"][m])) <= {0, 1, 2}, "masked token kept a role edge"
        hubs = c["type"] == HUB
        assert np.all(c["x"][hubs].shape[0] == len(inst["papers"]))
    print("  [ok] masked token: no features, no role edges, not duplicated; one hub per paper")

    n1 = Sampler(gi, cfg.override(n_subgraphs=1), "train", 0).sample(16)
    n8 = Sampler(gi, cfg.override(n_subgraphs=8), "train", 0).sample(16)
    assert all(len(i["papers"]) == 1 for i in n1) and all(len(i["papers"]) == 8 for i in n8)
    tr = set(gi.papers("train").tolist())
    assert all(set(i["papers"]) <= tr for i in n8), "neighbourhood leaked across splits"
    print("  [ok] neighbourhood size N honoured and confined to the split")

    model = PaperAModel(cfg)
    batch = to_device(collate(n8), "cpu")
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["target"], model.encode_target(batch["tgt"], batch["target_pos"]))
    print("  [ok] target encoder is deterministic and gradient-free")

    # multi-target training: k - 1 more hidden KUs, one per neighbouring paper, each leak-safe masked
    mt = Sampler(gi, cfg.override(n_subgraphs=8, targets_per_instance=4, target_types=("claim", "method")), "train", 0)
    insts = mt.sample(32)
    n_extra = []
    for inst in insts:
        c, ex = inst["ctx"], [e for e in inst["extra"] if e[3] > 0]
        n_extra.append(len(ex))
        for cp, tp, ty, j in ex:
            comp = int(c["comp"][cp]); tv = inst["extra_views"][j - 1]
            assert inst["is_mask"][cp] and np.all(c["x"][cp] == 0), "extra target not hidden"
            assert c["slot"][cp] != 0, "extra target in the anchor's paper"
            assert set(np.unique(c["rel"][cp])) <= {0, 1, 2}, "extra target kept a role edge"
            assert tv["comp"][tp] == comp and gi.comp_type[comp] == ty, "extra target view mismatch"
            assert not np.isin(c["comp"], gi.children.get(comp, []) + gi.parents.get(comp, [])).any(), \
                "extra target's paraphrases visible"
    batch = to_device(collate(insts), "cpu")
    with torch.no_grad():
        out = model(batch)
    assert out["pred_extra"].shape[0] == sum(n_extra) == batch["extra"][0].numel()
    assert batch["tgt"]["x"].shape[0] == len(insts) + sum(n_extra)
    print(f"  [ok] multi-target: {np.mean(n_extra):.1f} extra hidden KUs per instance, leak-safe, own target views")


def check_domains(gi, cfg):
    """canonical domains, temperature quotas, macro averaging, domain-weighted training anchors."""
    from collections import Counter
    from sci_jepa.data.ku_graph import _temperature_quota
    from sci_jepa.data.ku_schema import DOMAINS, domain_of
    from sci_jepa.evaluation.baselines import text_baseline
    assert domain_of("Astronomy — Galaxies") == "Astrophysics" and domain_of("Machine Learning — NLP") == "Computer Science"
    assert domain_of("Statistical Mechanics — Spin glasses") == "Physics" and domain_of("Statistics — Bayesian") == "Statistics"
    q = _temperature_quota({"Physics": 3000, "CS": 2000, "Chemistry": 50}, 1000, 0.5)
    assert sum(q.values()) == 1000 and q["Chemistry"] == 50 and q["Physics"] < 600, q
    u = _temperature_quota({"Physics": 3000, "CS": 2000, "Chemistry": 50}, 1000, 1.0)
    assert abs(u["Physics"] - 594) <= 1 and sum(u.values()) == 1000, u   # natural shares at alpha 1
    assert hasattr(gi, "paper_domain") and len(gi.paper_domain) == gi.n_papers
    dist = Counter(gi.domain_names[d] for d in gi.paper_domain)
    m = text_baseline(cfg.override(baseline="text", min_domain_queries=1), gi, "test")
    mac = m["tasks"].get("macro") or {}
    assert mac.get("index") is not None and m["tasks"]["forward"].get("by_domain"), mac
    s = Sampler(gi, cfg.override(train_domain_alpha=0.0), "train", 0)
    a = s._draw("all", 2000)
    share = Counter(gi.domain_names[gi.paper_domain[gi.comp_paper[x]]] for x in a)
    print(f"  [ok] domains: {dict(dist)}; temperature quota (alpha 0.5) {q}; macro index {mac['index']:.2f} over "
          f"{len(m['tasks']['forward']['by_domain'])} domains; domain-weighted anchors (alpha 0): {dict(share)}")


def check_validity_and_findings(gi, cfg):
    """validity.py (surface-cue audit, filter, annotation round trip) and findings.py (bootstrap CIs)."""
    import csv
    import json
    import os
    import tempfile
    from sci_jepa.evaluation import validity as V
    from sci_jepa.evaluation.findings import write as write_findings
    from sci_jepa.training.train import run_experiment
    a = V.audit(gi, cfg, save=False)
    assert 0 <= a["detector_accuracy"] <= 1 and set(a["by_kind"]) <= {"direction", "negation", "number", "substitution"}
    print(f"  [ok] validity audit: context-free detector names the original in {a['detector_accuracy']:.0%} of "
          f"{a['n']} counterfactuals; {a['flagged_share']:.0%} flagged (detectable from wording alone)")
    d = tempfile.mkdtemp()
    e = V.export(gi, cfg, d, n=30)
    key = {k["item_id"]: k for k in json.load(open(os.path.join(d, "key_do_not_share.json")))}
    for name, noise in (("A", 0.0), ("B", 0.2)):          # simulated annotators: A always right, B 80% right
        rows = list(csv.DictReader(open(os.path.join(d, f"annotator_{name}.csv"))))
        rng = np.random.default_rng(1 if name == "B" else 0)
        for r in rows:
            if r["kind"] == "counterfactual":
                right = key[r["item_id"]]["original"]
                r["Q1_which_is_the_papers_A_or_B"] = right if rng.random() >= noise else ("A" if right == "B" else "B")
                r["Q2_edited_option_is_fluent_1to5"] = "4"; r["Q3_edit_changes_the_meaning_yes_no"] = "yes"
        with open(os.path.join(d, f"annotator_{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    g = V.agreement([os.path.join(d, "annotator_A.csv"), os.path.join(d, "annotator_B.csv")])
    acc = g["per_annotator"]["annotator_A.csv"]["human_accuracy"]
    assert acc == 1.0 and g["kappa"], g
    print(f"  [ok] annotation round trip: {e['items']} items exported; simulated annotators -> human accuracy "
          f"{acc:.0%} / {g['per_annotator']['annotator_B.csv']['human_accuracy']:.0%}, kappa computed")
    out = tempfile.mkdtemp()
    arms = [("text_baseline", dict(baseline="text")), ("reference", dict()), ("jepa+gine", dict(encoder="gine")),
            ("world_model", dict(wm_p_action=0.25))]
    run_experiment(cfg.override(out_dir=out, eval_tasks=True, eval_worldview=True), gi, None, arms=arms,
                   log=lambda *_: None, name="FINDINGS_TEST")
    f = json.load(open(os.path.join(out, "FINDINGS_TEST", "findings.json")))
    h1 = f["hypotheses"][0]
    assert h1.get("ci95") and h1["ci95"][0] is not None and f["conclusion"], h1
    print(f"  [ok] findings: H1 {h1['outcome']} (difference {h1['difference']:+.2f}, 95% CI "
          f"{h1['ci95'][0]:+.2f} .. {h1['ci95'][1]:+.2f}); conclusion written")


def main():
    cfg = PaperAConfig().override(**SIZES["tiny"], **FAST)
    gi = get_index(cfg)
    check_invariants(gi, cfg)
    quiet = cfg.override(eval_tasks=False, eval_worldview=False)
    a, b = train_one(quiet, gi, 0, log=lambda *_: None), train_one(quiet.override(prefetch=2), gi, 0, log=lambda *_: None)
    assert a["history"][-1]["jepa"] == b["history"][-1]["jepa"], "prefetching changed training"
    print(f"  [ok] prefetching batches in a background thread gives the identical run "
          f"(final loss {a['history'][-1]['jepa']:.6f} both)")
    # early stopping on the validation reasoning index, testing the best epoch
    es = train_one(cfg.override(epochs=4.0, patience=1, eval_tasks=True, eval_worldview=False, neuron_report=False,
                                epoch_eval_instances=48, task_eval_instances=32), gi, 0, log=lambda *_: None)
    rl = es["run_length"]
    assert rl["best_epoch"] is not None and rl["tested_weights"] == "best" and len(es["epoch_results"]) >= 2, rl
    print(f"  [ok] early stopping: best epoch {rl['best_epoch']}, trained {rl['epochs_trained']} of {rl['max_epochs']:.0f} "
          f"epochs (stopped early: {rl['stopped_early_at']}), tested the best epoch's weights")
    from sci_jepa.data.ku_schema import MP_KINDS, DESIGN_TYPES, COMP_TYPES as CT
    kinds = {MP_KINDS[k]: int((gi.mp_kind == k).sum()) for k in np.unique(gi.mp_kind)}
    design = sum(1 for c in gi.mp_comp if CT[gi.comp_type[c]] in DESIGN_TYPES)
    assert design > 0 and kinds.get("substitution", 0) > 0, kinds
    print(f"  [ok] counterfactuals incl. experiment design: {kinds}; {design} on methods / procedures")
    check_domains(gi, cfg)
    check_validity_and_findings(gi, cfg)
    failures = []
    if "--full" in sys.argv:
        todo = [(e, t, o, False) for e, arms in EXPERIMENTS.items() for t, o in arms]
    elif "--all" in sys.argv:                          # the ALL leaderboard: every distinct method, tasks on
        todo = [("ALL", t, o, True) for t, o in EXPERIMENTS["ALL"]]
    else:
        todo = [(e, t, dict(EXPERIMENTS[e])[t], tasks) for e, t, tasks in SMALL] + EXTRA
    for exp, tag, over, tasks in todo:
        if True:
            try:
                acfg = cfg.override(**over, eval_tasks=tasks, eval_worldview=tasks)
                agi = get_index(acfg) if acfg.ku_mode != cfg.ku_mode else gi   # E16: its own KU graph
                if acfg.baseline:
                    from sci_jepa.evaluation.baselines import text_baseline
                    m = text_baseline(acfg, agi, "test")
                    print(f"  [ok] {exp:<5s} {tag:<22s} balanced MRR {m['retrieval_balanced']['MRR']:.3f} "
                          f"(chance {m['retrieval_balanced']['chance_MRR']:.3f}) forward {m['tasks']['forward']['ratio']:.2f}x")
                    continue
                m = train_one(acfg, agi, 0, log=lambda *_: None)
                bm = m["retrieval_balanced"]["MRR"]
                assert math.isfinite(bm) and math.isfinite(m["history"][-1]["jepa"])
                print(f"  [ok] {exp:<4s} {tag:<22s} balanced MRR {bm:.3f} chi {m['target_space']['chi']:.3f} "
                      f"probe {m['partition_probe']['probe_acc']:.3f} inv {m['inverse_acc']:.3f} "
                      f"det {m.get('detector', {}).get('discrimination', float('nan')):+.3f}")
                if tasks:
                    s_, a_ = m["structure"], m["actions"]
                    print(f"       world: x chance with edges {s_['with']:.2f} / without {s_['without']:.2f}; "
                          f"action direction swap {a_['swap']['direction_cos']} (n={a_['swap']['n']}) "
                          f"flip {a_['flip']['direction_cos']} (n={a_['flip']['n']})")
                    T = m["tasks"]
                    assert all(k in T for k in ("forward", "inverse", "abduction", "consistency", "intervention")), T
                    print(f"       tasks: forward {T['forward']['ratio']:.2f}x  inverse {T['inverse']['ratio']:.2f}x  "
                          f"abduction {T['abduction']['ratio']:.2f}x  consistency {T['consistency']['bal_acc']}  "
                          f"intervention n={T['intervention']['condition']['n']}")
            except Exception as e:                       # noqa: BLE001
                failures.append((exp, tag, repr(e)))
                print(f"  [FAIL] {exp} {tag}: {e!r}")
    print(f"\n{'ALL ARMS RAN' if not failures else f'{len(failures)} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
