"""
Paper A runner.

    # one configuration
    python -m sci_jepa.training.train --size S --set n_subgraphs=8 lambda_inv=0.3

    # an ablation from the plan (all arms x all seeds, paired tests vs the first arm)
    python -m sci_jepa.training.train --exp E10 --size S

    # Theorem A2 tightness audit of the shape vocabulary (no training)
    python -m sci_jepa.training.train --audit-shapes [--set shape_spec=path.json]

    # smoke test on a synthetic graph, CPU, ~1 min
    python -m sci_jepa.training.train --synthetic --size tiny --exp E10 --set steps=60 seeds=[0,1]

Outputs (cfg.out_dir/<exp>/): one JSON per arm x seed, and summary.json with
mean / std per metric and paired t-tests against the reference arm.
"""
import argparse
import ast
import json
import math
import os
import time

import numpy as np
import torch

from sci_jepa.models import losses as L
from sci_jepa.config import EXPERIMENTS, SIZES, PaperAConfig, resolve_device
from sci_jepa.evaluation.diagnostics import evaluate
from sci_jepa.data.graph_index import get_index
from sci_jepa.data.instances import Sampler, collate, minimal_pair
from sci_jepa.models.model import PaperAModel, to_device
from sci_jepa.evaluation.neurons import neuron_report
from sci_jepa.models.shapes import ShapeBank, SlotLabels, audit_tightness

# Metrics compared across arms in summary.json; (path, higher_is_better).
KEY_METRICS = [
    (("retrieval_balanced", "MRR"), True),
    (("similarity", "acc"), True),
    (("similarity", "acc_top1"), True),
    (("similarity", "false_accept"), False),
    (("retrieval_standard", "MRR"), True),
    (("retrieval_standard", "Hits@1"), True),
    (("target_space", "chi"), True),
    (("target_space", "chi_raw"), True),
    (("context_space", "chi"), True),
    (("partition_probe", "between_subgraph_var_share"), False),
    (("partition_probe", "probe_acc"), False),
    (("relation_sensitivity",), True),
    (("inverse_acc",), True),
    (("detector", "discrimination"), True),
]


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _grad_norms(model):
    groups = {"encoder": model.context, "predictor": model.predictor, "inverse": model.inverse}
    if model.shapes is not None:
        groups["shapes"] = model.shapes
    out = {}
    for k, mod in groups.items():
        sq = [p.grad.detach().float().pow(2).sum() for p in mod.parameters() if p.grad is not None]
        out[k] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
    return out


@torch.no_grad()
def _step_stats(out, batch, cfg):
    """Per-step learning signals: loss per target KU type, prediction/target
    agreement, and whether the predictions are collapsing (std across the batch)."""
    pred, tgt, tt = out["pred"].float(), out["target"].float(), batch["target_type"]
    per = (pred - tgt).pow(2).mean(-1)
    cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)
    rec = {"cos_pred_target": float(cos.mean()),
           "pred_std": float(pred.std(0).mean()), "target_std": float(tgt.std(0).mean()),
           "pred_pairwise_cos": float(torch.nn.functional.normalize(pred - pred.mean(0), dim=-1)
                                      .matmul(torch.nn.functional.normalize(pred - pred.mean(0), dim=-1).T).mean()),
           "tokens_per_instance": float((~batch["ctx"]["pad"]).sum(1).float().mean())}
    from sci_jepa.data.ku_schema import COMP_TYPES
    rec["loss_by_type"] = {COMP_TYPES[int(t)]: float(per[tt == t].mean()) for t in tt.unique()}
    return rec


def mask_plan(cfg, sampler):
    """-> (n_steps, plan_fn, info). plan_fn(step) -> ({type: count} or None, phase label).

      mixed        every target type, in proportion to its KU count (None = one shared pool)
      balanced     equal numbers of each type in every batch
      round_robin  batch k is all one type, cycling through the types
      sequential   one phase per type, in target_types order: a full pass over every KU
                   of that type before moving on; replay_frac of each batch re-draws the
                   types of earlier phases (0 = pure sequential, prone to forgetting)
    With epochs > 0 the run is `epochs` full passes over all target KUs; sequential phases
    are sized by each type's KU count, so each type is masked `epochs` times over."""
    types = [t for t in cfg.target_types if sampler.n_anchors(t) > 0]
    missing = [t for t in cfg.target_types if t not in types]
    if missing:
        print(f"[paper_a] no KUs of type {missing} in this split: skipped")
    counts = {t: sampler.n_anchors(t) for t in types}
    B = cfg.batch_size
    per_step = B * max(cfg.targets_per_instance, 1)    # predicted KUs per step
    steps = int(math.ceil(sum(counts.values()) * cfg.epochs / per_step)) if cfg.epochs > 0 else cfg.steps
    sched = cfg.mask_schedule
    bounds, acc = [], 0.0
    for t in types:                                    # sequential phase ends
        acc += counts[t] / max(sum(counts.values()), 1) * steps
        bounds.append((t, int(round(acc))))

    def split(n, keys):
        base, rem = divmod(n, len(keys))
        return {k: base + (i < rem) for i, k in enumerate(keys)}

    def plan(step):
        if sched == "mixed":
            return None, "mixed"
        if sched == "balanced":
            return split(B, types), "balanced"
        if sched == "round_robin":
            t = types[step % len(types)]
            return {t: B}, t
        if sched == "sequential":
            k = next(i for i, (_, end) in enumerate(bounds) if step < end or i == len(bounds) - 1)
            t, prev = types[k], types[:k]
            r = int(round(cfg.replay_frac * B)) if prev else 0
            out = {t: B - r}
            for pt, n in split(r, prev).items() if r else []:
                out[pt] = out.get(pt, 0) + n
            return out, t
        raise ValueError(f"unknown mask_schedule {sched!r}")

    info = {"schedule": sched, "steps": steps, "types": types, "anchors_per_type": counts,
            "phases": [{"type": t, "end_step": e} for t, e in bounds] if sched == "sequential" else None,
            "replay_frac": cfg.replay_frac if sched == "sequential" else None}
    return steps, plan, info


def epoch_checkpoint(model, gi, cfg, dev, epoch, step, train_jepa, sec, live=None):
    """Validation + neuron summary at the end of one epoch (one row of the
    dashboard's Epochs tab)."""
    m = evaluate(model, gi, cfg.override(eval_instances=cfg.epoch_eval_instances,
                                         probe_papers=min(cfg.probe_papers, 80)), dev, None, split="val")
    rb, pt = m["retrieval_balanced"], m.get("per_type", {})
    rec = {"epoch": epoch, "step": step, "sec": round(sec, 1), "train_jepa": train_jepa,
           "bal_MRR": rb["MRR"], "bal_chance": rb.get("chance_MRR"), "bal_H1": rb["Hits@1"], "bal_H10": rb["Hits@10"],
           "acc_sim": (m.get("similarity") or {}).get("acc"), "sim_mean": (m.get("similarity") or {}).get("mean"),
           "acc_sim_top1": (m.get("similarity") or {}).get("acc_top1"),
           "false_accept": (m.get("similarity") or {}).get("false_accept"),
           "std_MRR": m["retrieval_standard"]["MRR"], "std_chance": m["chance_MRR_standard"],
           "chi": m["target_space"]["chi"], "eff_rank": m["target_space"]["eff_rank"],
           "probe_acc": m["partition_probe"].get("probe_acc"), "probe_chance": m["partition_probe"].get("probe_chance"),
           "between_var": m["partition_probe"].get("between_subgraph_var_share"),
           "inverse_acc": m["inverse_acc"], "rel_sens": m["relation_sensitivity"],
           "flagged": m["audit"]["flagged_frac"],
           "per_type": {k: {"MRR": v["MRR_bal"], "chance": v["chance_bal"], "n": v["n"],
                            "same_paper": v["top1_same_paper"]} for k, v in pt.items()},
           "types_learned": sum(v["MRR_bal"] >= 1.5 * v["chance_bal"] for v in pt.values())}
    if cfg.eval_tasks:
        from sci_jepa.evaluation.tasks import task_eval
        rec["tasks"] = task_eval(model, gi, cfg, dev, "val")
    if cfg.eval_worldview and cfg.epoch_worldview:
        from sci_jepa.evaluation.worldview import action_eval, structure_reliance
        rec["structure"] = structure_reliance(model, gi, cfg.override(eval_instances=cfg.epoch_eval_instances,
                                                                      probe_papers=min(cfg.probe_papers, 80)),
                                              dev, "val", base=m)
        rec["actions"] = action_eval(model, gi, cfg, dev, "val", n=min(200, cfg.epoch_eval_instances))
        if live is not None:                          # the World model tab updates every epoch
            from sci_jepa.evaluation.worldview import examples
            live.worldview(step, epoch, examples(model, gi, cfg, dev, "val"))
    if cfg.neuron_report:
        rep = neuron_report(model, gi, cfg, dev, "val", cfg.neuron_instances)
        out = rep["layers"][-1]
        n = max(sum(out["counts"].values()), 1)
        rec["neurons"] = {"identity": out["counts"]["identity"] / n, "content": out["counts"]["content"] / n,
                          "type": out["counts"]["type"] / n, "dead": out["counts"]["dead"] / n,
                          "leak": out["counts"]["leak"] / n,
                          "pred_learned": rep["predictor"]["learned"], "pred_n": rep["predictor"]["n"]}
        if live is not None:
            live.neurons(step, epoch, rep)
    if live is not None:
        live.epoch_end(rec)
    return rec


def save_model(model, cfg, save_dir, metrics=None, best_state=None, gi=None):
    """Write a reusable model folder:
       model.safetensors  weights at the end of training (all modules, incl. the EMA target encoder)
       best.safetensors   weights at the best end-of-epoch validation score (if tracked)
       config.json        the full PaperAConfig, so inference rebuilds the identical model
       summary.json       test metrics, reasoning tasks, data fingerprint, text encoder
    Load with sci_jepa.inference.SciJEPA.from_pretrained(save_dir)."""
    from safetensors.torch import save_file
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(save_dir, "model.safetensors"))
    if best_state is not None:
        save_file({k: v.contiguous() for k, v in best_state["state"].items()}, os.path.join(save_dir, "best.safetensors"))
    json.dump(cfg.as_dict(), open(os.path.join(save_dir, "config.json"), "w"), indent=2, default=str)
    summ = {"best_epoch": best_state["epoch"] if best_state else None,
            "best_val_score": best_state["score"] if best_state else None, "select_metric": cfg.select_metric,
            "text_encoder": cfg.embed_model, "feat_dim": cfg.feat_dim, "ku_mode": cfg.ku_mode,
            "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    if gi is not None:
        from sci_jepa.data.ku_graph import data_fingerprint
        summ["data_fingerprint"] = data_fingerprint(cfg)
    if metrics:
        rb = metrics.get("retrieval_balanced", {})
        summ["test"] = {"balanced_MRR": rb.get("MRR"), "chance": rb.get("chance_MRR"),
                        "tasks": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}
                                  for k, v in (metrics.get("tasks") or {}).items() if isinstance(v, dict)}}
    json.dump(summ, open(os.path.join(save_dir, "summary.json"), "w"), indent=2, default=str)
    return save_dir


class _StepPlan:
    """Parallel batch building, main-process side (a DataLoader batch sampler): draws every step's anchors from
    the sampler's epoch pools, in order, and attaches the sampler state where a checkpoint needs it."""

    def __init__(self, sampler, plan_fn, n_steps, batch_size, start, snap):
        self.sampler, self.plan_fn, self.n_steps, self.batch_size = sampler, plan_fn, n_steps, batch_size
        self.start, self.snap = start, snap

    def __len__(self):
        return max(self.n_steps - self.start, 0)

    def __iter__(self):
        for step in range(self.start, self.n_steps):
            plan, phase = self.plan_fn(step)
            anchors = self.sampler.draw(self.batch_size, plan)
            state = self.sampler.get_state() if self.snap is not None and self.snap(step) else None
            yield [(step, phase, anchors, state)]


class _StepBuilder(torch.utils.data.Dataset):
    """Parallel batch building, worker side: the instances of one step, from that step's own random stream."""

    def __init__(self, sampler, seed):
        self.sampler, self.seed = sampler, seed

    def __len__(self):
        return 0

    def __getitem__(self, item):
        step, phase, anchors, state = item
        insts = self.sampler.build(anchors, np.random.default_rng([int(self.seed), int(step)]))
        return step, phase, len(insts), (collate(insts) if insts else None), state


def _first(items):
    return items[0]


def lr_factor(cfg, n_steps):
    """Learning-rate multiplier per step: linear warm-up over cfg.warmup steps, then constant or (cfg.lr_schedule
    "cosine") a cosine decay from cfg.lr to cfg.lr_min at the last step of the run's budget (max epochs)."""
    w, floor = max(cfg.warmup, 1), cfg.lr_min / max(cfg.lr, 1e-12)

    def f(s):
        if s < w:
            return (s + 1) / w
        if cfg.lr_schedule != "cosine":
            return 1.0
        t = min((s - w) / max(n_steps - w, 1), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def _batches(sampler, plan_fn, n_steps, batch_size, prefetch, start=0, snap=None, workers=0, seed=0):
    """-> (step, phase, instances, CPU batch, sampler state or None) for every step from `start`. prefetch > 0:
    a background thread builds the next batches while the GPU trains on this one. It is the only user of
    the sampler, so the batches and their order are exactly those without prefetching. snap(step) True: the
    sampler's state right after drawing that step's batch is attached (for a checkpoint at that step; the
    thread runs ahead, so the state must be taken here, not in the training loop).
    workers > 0: `workers` processes build the batches (instances = their number then, not the list); the
    anchors are still drawn here, in order, and step s is built from the random stream (seed, s)."""
    if workers > 0:
        loader = torch.utils.data.DataLoader(
            _StepBuilder(sampler, seed), batch_sampler=_StepPlan(sampler, plan_fn, n_steps, batch_size, start, snap),
            num_workers=workers, collate_fn=_first, prefetch_factor=max(prefetch, 4), pin_memory=torch.cuda.is_available(),
            multiprocessing_context="fork")
        it = iter(loader)
        try:
            yield from it
        finally:
            del it
        return
    def make(step):
        plan, phase = plan_fn(step)
        insts = sampler.sample(batch_size, plan)
        state = sampler.get_state() if snap is not None and snap(step) else None
        return step, phase, insts, (collate(insts) if insts else None), state
    if prefetch <= 0:
        for step in range(start, n_steps):
            yield make(step)
        return
    import queue
    import threading
    q, stop = queue.Queue(maxsize=prefetch), threading.Event()

    def work():
        try:
            for step in range(start, n_steps):
                if stop.is_set():
                    return
                q.put(make(step))
        except BaseException as e:                      # surface sampler errors in the training loop
            q.put(e)
    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        for _ in range(start, n_steps):
            item = q.get()
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while th.is_alive():                            # unblock a worker waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                th.join(0.05)


def _autocast(cfg, dev):
    """Mixed precision for the model's forward pass (cfg.amp); a no-op when off."""
    import contextlib
    if cfg.amp == "off" or dev.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(dev.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[cfg.amp])


def _select_score(rec, cfg):
    """Validation score of one epoch for early stopping / best-epoch selection."""
    if cfg.select_metric == "reasoning_index" and rec.get("tasks"):
        from sci_jepa.evaluation.tasks import add_domains, reasoning_index
        r = reasoning_index(rec["tasks"], cfg.primary_average)
        if r is not None:
            return r
    return rec["bal_MRR"] / max(rec.get("bal_chance") or 1e-9, 1e-9)   # retrieval x chance


def _fp32(out):
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}


# ---- resume a training run across jobs (cfg.resume_runs; run_experiment passes ckpt_path) --------------------
CKPT_VERSION = 1
_CKPT_IGNORE = ("out_dir", "model_dir", "index_dir", "device", "resume_runs", "ckpt_every", "llm_models",
                "batch_workers", "tf32", "fused_adamw", "trainer", "devices")   # speed settings: a run may resume with other values


def _ckpt_key(cfg, seed):
    """What must match for a checkpoint to belong to this run: the seed and every setting except where files
    live and how often checkpoints are written."""
    return {"seed": int(seed), **{k: repr(v) for k, v in sorted(cfg.as_dict().items()) if k not in _CKPT_IGNORE}}


def _rng_states():
    import random
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_states(s):
    import random
    torch.set_rng_state(s["torch"]); np.random.set_state(s["numpy"]); random.setstate(s["python"])
    if s.get("cuda") is not None and torch.cuda.is_available() and len(s["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(s["cuda"])


def _save_ckpt(path, payload):
    """Atomic: a job killed while writing leaves the previous checkpoint intact."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_ckpt(path, key, log):
    if not path or not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                              # noqa: BLE001  (a damaged file: start over)
        log(f"  [resume] cannot read {path} ({e}): starting this run from the beginning")
        return None
    if ck.get("version") != CKPT_VERSION or ck.get("key") != key:
        log(f"  [resume] {path} belongs to other settings: starting this run from the beginning")
        return None
    return ck


def train_one(cfg, gi, seed, log=print, live=None, save_dir=None, ckpt_path=None):
    """live: optional telemetry.LiveRun -- streams every log step and every
    cfg.eval_every steps a quick validation pass to the dashboard.
    save_dir: write the trained model there (see save_model); None = do not save.
    ckpt_path: resume file. If it exists and belongs to this run (same settings and seed), training continues
    from it; while training, a checkpoint is written every cfg.ckpt_every steps and at every epoch end, and
    once more when training ends (so a job stopped during the final evaluation does not retrain). The caller
    deletes it when the result is saved. None = no checkpoints (the original behaviour)."""
    set_seed(seed)
    dev = torch.device(resolve_device(cfg.device))
    if cfg.minimal_pairs and cfg.mp_filter:            # adversarially filtered counterfactuals (validity.py)
        from sci_jepa.evaluation.validity import apply_filter
        apply_filter(gi, cfg, log=log)
    if gi.comp_x.shape[1] != cfg.feat_dim:            # node features come from the index's text encoder
        log(f"  [model] feat_dim {cfg.feat_dim} -> {gi.comp_x.shape[1]} (dimension of the KU index's text encoder)")
        cfg = cfg.override(feat_dim=int(gi.comp_x.shape[1]))
    model = PaperAModel(cfg)
    # optional modules draw from their own seed AFTER the shared model, so every
    # arm of an ablation starts from identical encoder / predictor weights
    shapes = None
    if cfg.lambda_shacl > 0 or cfg.shape_spec:
        torch.manual_seed(seed + 10_007)
        shapes = ShapeBank(cfg.shape_spec, SlotLabels(gi), cfg.d_model, cfg)
        model.shapes = shapes
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if dev.type == "cuda":                             # TF32 for the fp32 matrix products (cfg.tf32; Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    fused = bool(cfg.fused_adamw) and dev.type == "cuda"
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, **({"fused": True} if fused else {}))
    sampler = Sampler(gi, cfg, "train", seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    history, t0 = [], time.time()
    n_steps, plan_fn, sched_info = mask_plan(cfg, sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(cfg, n_steps))   # warm-up, then constant / cosine
    log(f"  [lr] {cfg.lr_schedule}: warm-up {cfg.warmup} steps to {cfg.lr:g}"
        + (f", cosine to {cfg.lr_min:g} at step {n_steps}" if cfg.lr_schedule == "cosine" else ""))
    log(f"  [mask] schedule={sched_info['schedule']} steps={n_steps} anchors={sched_info['anchors_per_type']}"
        + (f" phases={sched_info['phases']}" if sched_info["phases"] else "")
        + (f" targets/instance={cfg.targets_per_instance}" if cfg.targets_per_instance > 1 else ""))
    total_anchors = max(sum(sched_info["anchors_per_type"].values()), 1)
    per_step = cfg.batch_size * max(cfg.targets_per_instance, 1)        # predicted KUs per step
    epoch_of = lambda st: round((st + 1) * per_step / total_anchors, 4)   # passes over all target KUs
    if live is not None:
        live.state["steps_total"] = n_steps
        live.state["schedule"] = sched_info
        live.state["epochs_total"] = round(n_steps * per_step / total_anchors, 3)
        live.state["steps_per_epoch"] = total_anchors / per_step
    # end-of-epoch checkpoints: the last step of every full pass over the target KUs
    epoch_ends = {}
    if cfg.eval_at_epochs and cfg.epochs > 0:
        for e in range(1, int(math.floor(cfg.epochs + 1e-9)) + 1):
            epoch_ends[min(int(math.ceil(e * total_anchors / per_step)) - 1, n_steps - 1)] = e
    epoch_results, ep_sum, ep_n = [], 0.0, 0
    best_state = None                                  # weights at the best end-of-epoch validation
    track_best = bool(save_dir) or cfg.select_best or cfg.patience > 0
    stopped_epoch, phase = None, None
    scaler = torch.amp.GradScaler(dev.type) if cfg.amp == "fp16" and dev.type != "cpu" else None
    # the contrast loss (E5) draws from the sampler's random stream inside the step: no prefetching
    prefetch = cfg.prefetch if cfg.lambda_contrast <= 0 else 0
    # parallel batch building: not with the contrast loss (above), nor with neighbourhoods frozen at their first
    # draw (E4: the frozen cache would differ between the worker processes)
    workers = cfg.batch_workers if cfg.lambda_contrast <= 0 and cfg.resample_each_epoch else 0
    if cfg.amp != "off" or prefetch:
        log(f"  [speed] amp={cfg.amp} prefetch={prefetch} attention={cfg.attn_impl} batch workers={workers} "
            f"tf32={cfg.tf32} fused AdamW={fused}")

    # ---- resume ------------------------------------------------------------------------------------------
    key = _ckpt_key(cfg, seed) if ckpt_path else None
    start, train_done, step = 0, False, -1
    ck = _load_ckpt(ckpt_path, key, log)
    if ck is not None:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["gen"])
        _set_rng_states(ck["rng"])
        if ck.get("sampler") is not None:
            sampler.set_state(ck["sampler"])
        history, epoch_results, best_state = ck["history"], ck["epoch_results"], ck["best_state"]
        ep_sum, ep_n, phase, stopped_epoch = ck["ep_sum"], ck["ep_n"], ck["phase"], ck["stopped_epoch"]
        t0 = time.time() - ck["elapsed"]
        step, start, train_done = ck["step"], ck["step"] + 1, ck["done"]
        log(f"  [resume] {ckpt_path}: continuing after step {step} of {n_steps} (epoch {epoch_of(step)}, "
            f"{ck['elapsed'] / 3600:.2f} h trained so far)" + (" -- training finished, evaluating" if train_done else ""))
        if live is not None:
            live.restore(step)
    # progress for the infrastructure monitor (sci_jepa.infra): a tiny file about every 5 s, never fatal
    prog_name = live.name if live is not None else f"run_seed{seed}"
    prog = {"t": time.time(), "step": step}

    def report(status, at):
        try:
            from sci_jepa.infra.monitor import write_progress
            now = time.time()
            rate = (at - prog["step"]) / max(now - prog["t"], 1e-9) if status == "training" else None
            prog.update(t=now, step=at)
            mem = {}
            if dev.type == "cuda":
                mem = {"device": torch.cuda.get_device_name(dev),
                       "gpu_mem_alloc_gb": round(torch.cuda.memory_allocated(dev) / 1e9, 2),
                       "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved(dev) / 1e9, 2),
                       "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)}
            write_progress(cfg.out_dir, prog_name, status=status, step=int(at), n_steps=int(n_steps),
                           epoch=epoch_of(at) if at >= 0 else 0.0, max_epochs=cfg.epochs,
                           elapsed_s=round(now - t0), steps_per_s=round(rate, 3) if rate else None,
                           instances_per_s=round(rate * cfg.batch_size, 1) if rate else None,
                           eta_max_s=round((n_steps - at - 1) / rate) if rate else None,
                           best_epoch=best_state["epoch"] if best_state else None,
                           batch_size=cfg.batch_size, params_trainable=int(sum(p.numel() for p in params)), **mem)
        except Exception:                               # noqa: BLE001  (monitoring never stops training)
            pass
    report("starting", step)
    ck_every = cfg.ckpt_every if ckpt_path else 0
    snap = (lambda st: (ck_every and (st + 1) % ck_every == 0) or st in epoch_ends) if ckpt_path else None

    def checkpoint(at, sampler_state, done):
        _save_ckpt(ckpt_path, {
            "version": CKPT_VERSION, "key": key, "step": at, "done": done,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "gen": gen.get_state(), "rng": _rng_states(), "sampler": sampler_state,
            "history": history, "epoch_results": epoch_results, "best_state": best_state,
            "ep_sum": float(ep_sum), "ep_n": ep_n, "phase": phase, "stopped_epoch": stopped_epoch,
            "elapsed": time.time() - t0, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job": os.environ.get("SLURM_JOB_ID")})

    use_lightning = cfg.trainer == "lightning" and not train_done
    if use_lightning:                                  # the same loop on PyTorch Lightning (training/lightning_loop.py)
        from sci_jepa.training import lightning_loop as LL
        ok, why = LL.supported(cfg, ckpt_path)
        if not ok:
            log(f"  [lightning] not used ({why}): native training loop")
            use_lightning = False
    if use_lightning:
        res = LL.fit_lightning(model=model, cfg=cfg, gi=gi, dev=dev, seed=seed, log=log, live=live, opt=opt,
                               sched=sched, scaler=scaler, params=params, sampler=sampler, plan_fn=plan_fn,
                               n_steps=n_steps, prefetch=prefetch, workers=workers, epoch_ends=epoch_ends,
                               epoch_of=epoch_of, track_best=track_best, shapes=shapes, gen=gen, t0=t0,
                               history=history, epoch_results=epoch_results, report=report, prog=prog)
        if res is None:                                # another rank of a multi-GPU run: rank 0 evaluates
            return None
        best_state, stopped_epoch, step, phase = res
        batches = iter(())
    else:
        batches = iter(()) if train_done else _batches(sampler, plan_fn, n_steps, cfg.batch_size, prefetch, start,
                                                       snap, workers, seed)
    for step, phase_now, insts, cpu_batch, sampler_state in batches:
        if phase_now != phase and cfg.mask_schedule == "sequential":
            log(f"  [mask] step {step}: phase -> masking '{phase_now}'"
                + (f" (+{cfg.replay_frac:.0%} replay)" if cfg.replay_frac and step else ""))
        phase = phase_now
        if not insts:
            continue
        batch = to_device(cpu_batch, dev)
        with _autocast(cfg, dev):
            out = model(batch)
        out = _fp32(out)                                # every loss is computed in fp32
        if "pred_extra" in out:                         # E20: every hidden motif node counts equally
            parts = {"jepa": L.jepa_loss(torch.cat([out["pred"], out["pred_extra"]]),
                                         torch.cat([out["target"], out["target_extra"]]), cfg.jepa_loss)}
        else:
            parts = {"jepa": L.jepa_loss(out["pred"], out["target"], cfg.jepa_loss)}
        if "inv_logits" in out:
            if cfg.lambda_inv > 0:                      # E11: the IDM-analogue barrier
                parts["inverse"] = cfg.lambda_inv * L.inverse_loss(out["inv_logits"], out["inv_labels"])
            else:                                       # detached probe: head learns, encoder does not
                b, i, j, _ = batch["probe"]
                h = out["h"].detach()
                parts["inverse_probe"] = L.inverse_loss(model.inverse(h[b, i], h[b, j]), out["inv_labels"])
        if cfg.lambda_reg > 0:
            c = batch["ctx"]
            parts["reg"] = cfg.lambda_reg * L.grouping_regulariser(
                out["h"], c["slot"], c["type"], c["pad"], cfg.reg_grouping, cfg.reg_projections, gen)
        if cfg.lambda_align > 0:
            parts["align"] = cfg.lambda_align * L.align_loss(out["pred"], out["target"],
                                                             batch["target_type"], cfg.align_tau)
        if cfg.lambda_shacl > 0 and shapes is not None:
            parts["shacl"] = cfg.lambda_shacl * shapes.loss(out["pred"], batch["target_comp"],
                                                            batch["target_type"])
        if cfg.lambda_contrast > 0:
            pairs = [minimal_pair(gi, i, sampler.rng, sampler.papers) for i in insts]
            keep = [k for k, p in enumerate(pairs) if p is not None]
            if keep:
                pb = to_device(collate([pairs[k] for k in keep]), dev)
                with _autocast(cfg, dev):
                    p2, _ = model.predict(pb)
                parts["contrast"] = cfg.lambda_contrast * L.contrast_margin(
                    out["pred"][keep], p2.float(), cfg.contrast_margin)
        loss = sum(parts.values())
        ep_sum, ep_n = ep_sum + parts["jepa"].detach(), ep_n + 1
        opt.zero_grad(set_to_none=True)
        if scaler is not None:                          # fp16: scale the loss so small gradients survive
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        logging_now = step % cfg.log_every == 0 or step == n_steps - 1
        gnorm = _grad_norms(model) if (logging_now and live is not None) else None
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        if scaler is not None:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        sched.step()
        m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * step / max(n_steps - 1, 1)
        model.ema(m)
        if time.time() - prog["t"] >= 5:
            report("training", step)

        if logging_now:
            rec = {"step": step, "epoch": epoch_of(step), "phase": phase,
                   **{k: float(v.detach()) for k, v in parts.items()},
                   "flagged": float(np.mean([a["flagged"] for a in batch["audit"]])),
                   "sec": round(time.time() - t0, 1)}
            history.append(rec)
            log("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()))
            if live is not None:
                live.relbias(model)
                live.step({**rec, "total": float(loss.detach()), "lr": float(sched.get_last_lr()[0]), "ema": m,
                           "grad_norm": gnorm, **_step_stats(out, batch, cfg)})
        if step in epoch_ends:
            rec = epoch_checkpoint(model, gi, cfg, dev, epoch_ends[step], step, float(ep_sum) / max(ep_n, 1),
                                   time.time() - t0, live)
            epoch_results.append(rec)
            ep_sum, ep_n = 0.0, 0
            rec["select_score"] = _select_score(rec, cfg)
            if track_best and (best_state is None or rec["select_score"] > best_state["score"]):
                best_state = {"epoch": rec["epoch"], "score": rec["select_score"],
                              "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            log(f"  [epoch {rec['epoch']}] val balanced MRR {rec['bal_MRR']:.4f} (chance {rec['bal_chance']:.4f}) "
                f"acc(sim>={cfg.sim_threshold:.2f}) {rec['acc_sim'] if rec.get('acc_sim') is not None else float('nan'):.3f} "
                f"(top-1 {rec['acc_sim_top1'] if rec.get('acc_sim_top1') is not None else float('nan'):.3f}, "
                f"false accept {rec['false_accept'] if rec.get('false_accept') is not None else float('nan'):.3f}) "
                f"train jepa {rec['train_jepa']:.4g} types learned {rec['types_learned']}/{len(rec['per_type'])} "
                f"{cfg.select_metric} {rec['select_score']:.3f}")
            if (cfg.patience > 0 and best_state is not None and rec["epoch"] >= cfg.min_epochs
                    and rec["epoch"] - best_state["epoch"] >= cfg.patience):
                stopped_epoch = rec["epoch"]
                log(f"  [early stop] no better validation {cfg.select_metric} for {cfg.patience} epochs: "
                    f"stopped after epoch {stopped_epoch}, best epoch {best_state['epoch']}")
                break
        if live is not None and cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            quick = cfg.override(eval_instances=cfg.live_eval_instances, probe_papers=min(cfg.probe_papers, 40))
            live.eval(step, evaluate(model, gi, quick, dev, None, split="val"), epoch=epoch_of(step))
            if cfg.neuron_report and step not in epoch_ends:
                live.neurons(step, epoch_of(step), neuron_report(model, gi, cfg, dev, "val", cfg.neuron_instances))
        if sampler_state is not None:                   # every ckpt_every steps and at every epoch end
            checkpoint(step, sampler_state, done=False)

    last_step = step
    report("evaluating", last_step)
    if ckpt_path and not train_done:                    # training over: a job stopped during the final
        checkpoint(last_step, None, done=True)          #   evaluation resumes straight into evaluation
    if cfg.select_best and best_state is not None:     # test the best validation epoch, not the last one
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if save_dir else None
        model.load_state_dict(best_state["state"])
        log(f"  [best] testing the weights of epoch {best_state['epoch']} "
            f"(validation {cfg.select_metric} {best_state['score']:.3f})")
    else:
        final_state = None
    n_steps = last_step + 1                            # steps actually trained (early stopping)
    metrics = evaluate(model, gi, cfg, dev, shapes)
    if cfg.eval_worldview:
        from sci_jepa.evaluation import worldview as W
        metrics["structure"] = W.structure_reliance(model, gi, cfg, dev, "test", base=metrics)
        metrics["actions"] = W.action_eval(model, gi, cfg, dev, "test")
        ex = W.examples(model, gi, cfg, dev, "test")
        if live is not None:
            live.worldview(n_steps, epoch_of(n_steps - 1), ex, final=True)
        s_, a_ = metrics["structure"], metrics["actions"]
        log(f"  [world] x chance with typed edges {s_['with']:.2f} / without {s_['without']:.2f} | action direction "
            f"swap {a_['swap']['direction_cos']} flip {a_['flip']['direction_cos']}")
    if cfg.eval_tasks:
        from sci_jepa.evaluation.tasks import task_eval
        metrics["tasks"] = task_eval(model, gi, cfg, dev, "test")
        log("  [tasks] " + " | ".join(f"{k} {v['ratio']:.2f}x chance" for k, v in metrics["tasks"].items()
                                      if isinstance(v, dict) and v.get("ratio") is not None))
    if live is not None:
        live.relbias(model)
        live.eval(n_steps, evaluate(model, gi, cfg.override(eval_instances=cfg.live_eval_instances,
                                                              probe_papers=min(cfg.probe_papers, 40)),
                                      dev, None, split="val"), epoch=epoch_of(n_steps - 1))
        if cfg.neuron_report:
            live.neurons(n_steps, epoch_of(n_steps - 1),
                         neuron_report(model, gi, cfg, dev, "test", cfg.neuron_instances * 2), final=True)
    metrics["history"] = history
    metrics["epoch_results"] = epoch_results
    metrics["train_sec"] = round(time.time() - t0, 1)
    metrics["params_trainable"] = int(sum(p.numel() for p in params))
    metrics["samples_seen"] = int(n_steps * cfg.batch_size)
    metrics["schedule"] = {**sched_info, "passes_completed": dict(getattr(sampler, "passes", {}))}
    metrics["run_length"] = {"max_epochs": cfg.epochs, "epochs_trained": epoch_of(last_step),
                             "stopped_early_at": stopped_epoch, "best_epoch": best_state["epoch"] if best_state else None,
                             "best_val_score": best_state["score"] if best_state else None,
                             "select_metric": cfg.select_metric, "tested_weights": "best" if cfg.select_best and best_state
                             else "last", "steps": int(last_step + 1)}
    if save_dir:
        if final_state is not None:                    # model.safetensors = end of training, best.safetensors = best
            best_now = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(final_state)
            metrics["model_dir"] = save_model(model, cfg, save_dir, metrics, best_state, gi)
            model.load_state_dict(best_now)
        else:
            metrics["model_dir"] = save_model(model, cfg, save_dir, metrics, best_state, gi)
        log(f"  [model] saved to {save_dir}" + (f" (best epoch {best_state['epoch']})" if best_state else ""))
    if live is not None:
        live.finish(metrics)
    report("done", last_step)
    return metrics


def summarise(results, ref_tag):
    from scipy import stats
    tags = list(results)
    summary = {}
    for tag in tags:
        row = {}
        for path, _ in KEY_METRICS:
            vals = [_get(r, path) for r in results[tag]]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                continue
            row[".".join(path)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            ref = [_get(r, path) for r in results[ref_tag]]
            if tag != ref_tag and len(vals) == len(ref) >= 2 and None not in ref:
                t, p = stats.ttest_rel(vals, ref)
                row[".".join(path)]["paired_vs_ref"] = {"t": float(t), "p": float(p),
                                                       "delta": float(np.mean(vals) - np.mean(ref))}
        summary[tag] = row
    return summary


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser("paper_a")
    ap.add_argument("--exp", default=None, help=f"one of {list(EXPERIMENTS)}")
    ap.add_argument("--size", default="S", choices=list(SIZES))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--audit-shapes", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run every n-th arm x seed of --exp, from i "
                                                  "(n machines sharing one results folder)")
    a = ap.parse_args()

    cfg = PaperAConfig().override(**SIZES[a.size], **parse_overrides(a.set))
    if a.synthetic:
        cfg = cfg.override(synthetic=True, out_dir=os.environ.get("PAPER_A_OUT", "./paper_a_out"))
    gi = get_index(cfg)
    S = gi.summary()
    print(f"[paper_a] index: {S['papers']} papers, {S['kus']} KUs ({S['kus_per_paper']}/paper), "
          f"splits {S['splits']}, citations {S['paper_edges_cites']}, proxy edges {S['paper_edges_proxy']}")

    if a.audit_shapes:
        rows = audit_tightness(cfg.shape_spec, SlotLabels(gi), gi)
        for r in rows:
            print("  " + json.dumps(r))
        os.makedirs(cfg.out_dir, exist_ok=True)
        json.dump(rows, open(os.path.join(cfg.out_dir, "shape_tightness.json"), "w"), indent=2)
        return

    run_experiment(cfg, gi, a.exp, shard=tuple(int(x) for x in a.shard.split("/")) if a.shard else None)


def _resumable(path, acfg):
    """A finished result for the same arm, seed and run settings (papers, epochs, size)."""
    if not os.path.exists(path):
        return None
    try:
        m = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    c = m.get("config", {})
    same = all(c.get(k, getattr(PaperAConfig(), k)) == getattr(acfg, k)
               for k in ("max_papers", "epochs", "d_model", "enc_layers", "ku_mode", "time_T", "split_dir", "n_subgraphs",
                         "batch_size", "patience", "select_best", "baseline", "encoder", "lr_schedule"))
    return m if same else None


def llm_short_name(model_id):
    """'mistralai/Mistral-7B-Instruct-v0.3' -> 'Mistral-7B-Instruct-v0.3' (the arm suffix)."""
    return model_id.rstrip("/").split("/")[-1]


def expand_llm_arms(arms, cfg):
    """With cfg.llm_models, the LLM-Reasoning arm (baseline="llm" without its own llm_model) becomes one arm per
    model: "LLM-Reasoning@<model name>". Without it, the arms are returned unchanged."""
    if not cfg.llm_models:
        return list(arms)
    out = []
    for tag, over in arms:
        if over.get("baseline") == "llm" and "llm_model" not in over:
            out += [(f"{tag}@{llm_short_name(m)}", {**over, "llm_model": m}) for m in cfg.llm_models]
        else:
            out.append((tag, over))
    return out


def _skipped(acfg, over):
    """Arms run_experiment skips (identical arms in paper mode; the LLM without a CUDA GPU)."""
    if acfg.ku_mode == "paper" and over.keys() & {"target_types", "mask_schedule", "replay_frac"}:
        return True
    return acfg.baseline == "llm" and resolve_device(acfg.device) != "cuda" and not acfg.llm_allow_cpu


def missing_runs(cfg, exp, name=None):
    """arm x seed runs of an experiment without a finished result file (with the same settings) in
    cfg.out_dir/<name>; [] = the experiment is complete."""
    out_dir = os.path.join(cfg.out_dir, name or exp)
    miss = []
    for tag, over in expand_llm_arms(EXPERIMENTS[exp], cfg):
        acfg = cfg.override(**over)
        if _skipped(acfg, over):
            continue
        miss += [(tag, s) for s in acfg.seeds if _resumable(os.path.join(out_dir, f"{tag}_seed{s}.json"), acfg) is None]
    return miss


def _claim(out_dir, tag, seed):
    """Dynamic sharding (env SCI_JEPA_CLAIM=1, set by cluster/grete/run_paper_a.sh dist): a shard takes an arm x seed
    run by creating its claim file atomically; True = this process got it. The job script clears the claims when a
    job starts, so a run interrupted by the time limit is claimed again (and resumed) by the next job."""
    d = os.path.join(out_dir, "claims")
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, f"{tag}_seed{seed}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"job {os.environ.get('SLURM_JOB_ID', '-')} shard {os.environ.get('SCI_JEPA_SHARD', '-')} "
                 f"pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
    os.close(fd)
    return True


def run_experiment(cfg, gi, exp=None, arms=None, log=print, live_root=None, resume=True, shard=None, name=None):
    """Train + evaluate every arm x seed of an experiment, write per-run JSON and
    summary.json, return (results, summary). exp: a key of EXPERIMENTS, or None
    for a single run of `cfg` (or pass `arms` = [(tag, overrides), ...]).
    resume: an arm x seed whose result file already exists with the same settings is
    loaded instead of retrained (so an interrupted benchmark continues where it stopped).
    shard = (i, n): run only every n-th arm x seed, starting at i, so n machines (or n jobs)
    share one experiment through a shared out_dir; a last call without shard loads every
    finished run and writes the summary.
    With env SCI_JEPA_CLAIM=1 the shards instead claim runs dynamically (_claim): every shard takes the next run
    nobody has taken: first the quick comparison systems (results within hours), then the trained arms (the
    longest), then the LLMs, so the GPUs finish at about the same time."""
    arms = expand_llm_arms(arms or (EXPERIMENTS[exp] if exp else [("single", {})]), cfg)
    dynamic = shard is not None and os.environ.get("SCI_JEPA_CLAIM") == "1"
    if dynamic:                                         # quick comparison systems, trained arms, then the LLMs
        rank = lambda o: 1 if not o.get("baseline") else 2 if o.get("baseline") == "llm" else 0
        arms = sorted(arms, key=lambda a: rank(a[1]))
    name = name or exp or "single"          # name: folder / dashboard label (e.g. ALL_smoke for exp ALL)
    from sci_jepa.config import describe, method_family
    out_dir = os.path.join(cfg.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    model_root = os.path.join(cfg.model_dir, name) if cfg.model_dir else os.path.join(out_dir, "models")
    results, base_gi, k_run = {}, gi, -1
    for tag, over in arms:
        acfg = cfg.override(**over)
        if acfg.ku_mode == "paper" and over.keys() & {"target_types", "mask_schedule", "replay_frac"}:
            log(f"\n[paper_a] {name} | {tag}: skipped -- with ku_mode='paper' there is one KU type, "
                f"so arms that vary the masked type / schedule are identical")
            continue
        if acfg.baseline == "llm" and resolve_device(acfg.device) != "cuda" and not acfg.llm_allow_cpu:
            log(f"\n[paper_a] {name} | {tag}: skipped -- no CUDA GPU; set llm_allow_cpu=True (notebook: LLM_ON_MAC) "
                f"to run it here with a small model (<= ~2B parameters)")
            continue
        results[tag] = []
        done = {s: _resumable(os.path.join(out_dir, f"{tag}_seed{s}.json"), acfg) for s in acfg.seeds} if resume else {}
        for s, m in done.items():
            if m is not None:
                log(f"\n[paper_a] {name} | {tag} | seed {s}: already finished, loaded from {out_dir}")
                results[tag].append(m)
        mine = set()
        for s in acfg.seeds:                            # this machine's share of the arm x seed runs
            k_run += 1
            if shard is None or dynamic or k_run % shard[1] == shard[0]:
                mine.add(s)
        todo = [s for s in acfg.seeds if done.get(s) is None and s in mine]
        if not todo:
            continue
        # an arm that changes the KU graph itself (E16: ku_mode) gets its own index (loaded once a run needs it)
        own_index, gi = "ku_mode" in over and acfg.ku_mode != cfg.ku_mode, base_gi
        for seed in todo:
            if dynamic:                                 # one run at a time, so long arms spread over the shards
                if not _claim(out_dir, tag, seed):      # another shard has it
                    continue
                m = _resumable(os.path.join(out_dir, f"{tag}_seed{seed}.json"), acfg)
                if m is not None:                       # finished by another shard since this one looked
                    results[tag].append(m); continue
            if own_index and gi is base_gi:
                gi = get_index(acfg)
            log(f"\n[paper_a] {name} | {tag} | seed {seed}")
            live = None
            if live_root:
                from sci_jepa.dashboard.telemetry import LiveRun
                live = LiveRun(live_root, f"{name}__{tag}__seed{seed}".replace("=", "-"), acfg, gi,
                               extra={"experiment": name, "arm": tag, "seed": seed,
                                      "about": describe({**over, "llm_model": acfg.llm_model}),
                                      "family": method_family(over), "overrides": {k: str(v) for k, v in over.items()}})
            if acfg.baseline:                            # BENCH: a no-training reference row
                from sci_jepa.evaluation.baselines import run_isolated, text_baseline
                if acfg.baseline == "llm" and not acfg.synthetic:   # own process: GPU memory fully isolated
                    m = run_isolated(acfg, "test", seed=1234 + seed)
                else:
                    m = text_baseline(acfg, gi, "test", seed=1234 + seed)
                if live is not None:
                    live.state["steps_total"] = 0
                    live.finish(m)
            else:
                ckpt = os.path.join(out_dir, "checkpoints", f"{tag}_seed{seed}.pt") if acfg.resume_runs else None
                m = train_one(acfg, gi, seed, log=log, live=live, ckpt_path=ckpt,
                              save_dir=os.path.join(model_root, f"{tag}_seed{seed}") if acfg.save_model else None)
            import gc                                  # free GPU memory between arms (encoders, LLMs, models)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            m["config"] = acfg.as_dict(); m["tag"] = tag; m["seed"] = seed
            m["about"], m["family"] = describe({**over, "llm_model": acfg.llm_model}), method_family(over)
            m["overrides"] = {k: str(v) for k, v in over.items()}
            json.dump(m, open(os.path.join(out_dir, f"{tag}_seed{seed}.json"), "w"), indent=2, default=str)
            ckpt = os.path.join(out_dir, "checkpoints", f"{tag}_seed{seed}.pt")
            if os.path.exists(ckpt):                     # result saved: the resume file is no longer needed
                os.remove(ckpt)
            results[tag].append(m)
            log(f"  -> balanced MRR {m['retrieval_balanced']['MRR']:.4f} "
                f"(chance {m['retrieval_balanced'].get('chance_MRR', float('nan')):.4f}) | "
                f"standard MRR {m['retrieval_standard']['MRR']:.4f} (chance {m['chance_MRR_standard']:.2e}) | "
                f"chi {m['target_space']['chi']:.3f} | probe {(m['partition_probe'].get('probe_acc') or float('nan')):.3f} | "
                f"rel-sens {m['relation_sensitivity']:.3f} | inv {m['inverse_acc']:.3f} | "
                f"audit flagged {m['audit']['flagged_frac']:.2f}")
    results = {t: r for t, r in results.items() if r}
    if not results:
        log(f"[paper_a] {name}: no arm ran"); return results, {}
    if shard is not None:
        log(f"\n[paper_a] {name}: shard {shard[0]}/{shard[1]} done; run once more without a shard "
            f"(everything finished is loaded) to write the summary")
        return results, {}
    ref = next(iter(results))
    summary = summarise(results, ref)
    json.dump({"experiment": name, "reference_arm": ref, "summary": summary,
               "index": base_gi.summary(), "note": "grounded edges are PROXIES"},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=str)
    log(f"\n[paper_a] summary -> {os.path.join(out_dir, 'summary.json')}")
    if len(results) > 1:                               # pre-registered hypotheses + conclusion (findings.py)
        try:
            from sci_jepa.evaluation.findings import write as write_findings
            f = write_findings(out_dir, dashboard=live_root)
            log(f"[paper_a] findings -> {os.path.join(out_dir, 'findings.json')}\n  {f.get('conclusion', '')}")
        except Exception as e:                         # noqa: BLE001 -- findings never break a finished run
            log(f"[paper_a] findings not computed: {e!r}")
        try:                                           # paper figures (PDF + PNG) and LaTeX / CSV tables
            from sci_jepa.reporting.figures import export as export_figures
            made = export_figures(out_dir)
            log(f"[paper_a] {len(made)} figures -> {os.path.join(out_dir, 'figures')}")
        except Exception as e:                         # noqa: BLE001
            log(f"[paper_a] figures not written: {e!r}")
    for tag, row in summary.items():
        bm = row.get("retrieval_balanced.MRR", {})
        ch = row.get("target_space.chi", {})
        log(f"  {tag:<22s} balanced MRR {bm.get('mean', float('nan')):.4f}±{bm.get('std', 0):.4f}  "
            f"chi {ch.get('mean', float('nan')):.3f}  "
            f"{'p=%.3g' % bm['paired_vs_ref']['p'] if 'paired_vs_ref' in bm else ''}")
    return results, summary


if __name__ == "__main__":
    main()
