"""
Test a finished run with its best epoch chosen by validation retrieval MRR instead of the configured select_metric
(the paper's default is the validation reasoning index). The 2-epoch runs report both: the run's own result
(reasoning-index selection) and this one (retrieval selection).

For each result JSON: the epoch with the highest validation balanced MRR (epoch_results) is taken.
  - it is the epoch the run already tested: its result is copied (no GPU work);
  - it is the last epoch: the end-of-training weights (<model_dir>/model.safetensors) are tested;
  - otherwise: the weights kept at that epoch's last step (<model_dir>/checkpoints/step*.safetensors).
The test is the one train_one runs: evaluate() on the test split, plus the world-model and task evaluations
when the run's config has them, with the same adversarial counterfactual filter.

    PYTHONPATH=src python scaling/test_by_mrr.py RESULTS/BENCH/jepa+gine_seed0.json [...] --out RESULTS/BENCH_mrr
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file

from sci_jepa.config import PaperAConfig, resolve_device
from sci_jepa.data.graph_index import get_index
from sci_jepa.evaluation.diagnostics import evaluate
from sci_jepa.models.model import PaperAModel
from sci_jepa.training.train import set_seed

KEEP = ("retrieval_balanced", "retrieval_standard", "structure", "actions", "tasks", "relation_sensitivity",
        "inverse_acc", "target_space", "context_space", "partition_probe", "per_type", "n_eval")


def pick(res):
    ep = [e for e in res.get("epoch_results") or [] if isinstance(e.get("bal_MRR"), (int, float))]
    if not ep:
        raise SystemExit(f"{res.get('tag')} seed {res.get('seed')}: no epoch_results")
    by_mrr = max(ep, key=lambda e: e["bal_MRR"])
    by_rule = max(ep, key=lambda e: e.get("select_score", float("-inf")))
    return ep, by_mrr, by_rule


def weights_for(model_dir, epoch, ep):
    if epoch == ep[-1]["epoch"]:
        return os.path.join(model_dir, "model.safetensors")
    step = next(e["step"] for e in ep if e["epoch"] == epoch) + 1
    return os.path.join(model_dir, "checkpoints", f"step{step:07d}.safetensors")


def test(model_dir, weights, seed, log=print):
    raw = json.load(open(os.path.join(model_dir, "config.json")))
    base = PaperAConfig()
    cfg = base.override(**{k: v for k, v in raw.items() if hasattr(base, k)})
    set_seed(int(seed or 0))
    dev = torch.device(resolve_device(cfg.device))
    gi = get_index(cfg)
    if cfg.minimal_pairs and cfg.mp_filter:            # the same filtered counterfactuals as train_one
        from sci_jepa.evaluation.validity import apply_filter
        apply_filter(gi, cfg, log=log)
    if gi.comp_x.shape[1] != cfg.feat_dim:
        cfg = cfg.override(feat_dim=int(gi.comp_x.shape[1]))
    model = PaperAModel(cfg)
    missing, unexpected = model.load_state_dict(load_file(weights), strict=False)
    if [k for k in missing if not k.startswith("shapes.")] or unexpected:
        raise RuntimeError(f"weights do not match the model: missing {missing[:5]} unexpected {unexpected[:5]}")
    model = model.to(dev)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    m = evaluate(model, gi, cfg, dev, None)
    if cfg.eval_worldview:
        from sci_jepa.evaluation import worldview as W
        m["structure"] = W.structure_reliance(model, gi, cfg, dev, "test", base=m)
        m["actions"] = W.action_eval(model, gi, cfg, dev, "test")
    if cfg.eval_tasks:
        from sci_jepa.evaluation.tasks import task_eval
        m["tasks"] = task_eval(model, gi, cfg, dev, "test")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result JSONs of finished runs (RESULTS/BENCH/<arm>_seed<s>.json)")
    ap.add_argument("--out", required=True, help="folder for the retrieval-selected results")
    ap.add_argument("--check", action="store_true", help="first re-test each run's best.safetensors: it must "
                    "reproduce the run's own test MRR, which validates this script's setup")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.results:
        res = json.load(open(f))
        tag, seed = res.get("tag"), res.get("seed")
        ep, by_mrr, by_rule = pick(res)
        out = os.path.join(a.out, os.path.basename(f))
        info = {"tag": tag, "seed": seed, "select": "validation balanced MRR", "epoch": by_mrr["epoch"],
                "val_bal_MRR": by_mrr["bal_MRR"], "rule_epoch": by_rule["epoch"],
                "val_by_epoch": [{"epoch": e["epoch"], "bal_MRR": e["bal_MRR"], "select_score": e.get("select_score")}
                                 for e in ep]}
        t0 = time.time()
        if a.check:
            c = test(res["model_dir"], os.path.join(res["model_dir"], "best.safetensors"), seed)
            info["check_best_MRR"] = c["retrieval_balanced"]["MRR"]
            print(f"  [check] best.safetensors re-tested: balanced MRR {c['retrieval_balanced']['MRR']:.4f} vs "
                  f"{res['retrieval_balanced']['MRR']:.4f} in the run", flush=True)
        if by_mrr["epoch"] == by_rule["epoch"]:
            print(f"{tag} seed {seed}: MRR and the run's rule both choose epoch {by_mrr['epoch']}: result copied")
            m = {k: res[k] for k in KEEP if k in res}
            info["weights"] = "same as the run's own test"
        else:
            w = weights_for(res["model_dir"], by_mrr["epoch"], ep)
            print(f"{tag} seed {seed}: testing epoch {by_mrr['epoch']} (val MRR {by_mrr['bal_MRR']:.4f}; the run "
                  f"tested epoch {by_rule['epoch']}) from {w}", flush=True)
            m = {k: v for k, v in test(res["model_dir"], w, seed).items() if k in KEEP}
            info["weights"] = w
        info["test_sec"] = round(time.time() - t0, 1)
        rb = m.get("retrieval_balanced", {})
        print(f"  -> balanced MRR {rb.get('MRR', float('nan')):.4f} (the run's own test: "
              f"{res['retrieval_balanced']['MRR']:.4f})", flush=True)
        json.dump({**info, **m}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
