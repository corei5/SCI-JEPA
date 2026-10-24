"""
Where does a training step spend its time? Times, on real data and the run's own settings:
  1. building the KU index (text encoder throughput),
  2. building one training batch on the CPU (sampler + collate),
  3. one optimisation step on the GPU (forward, loss, backward, AdamW),
  4. one end-of-epoch validation,
for several arms, and extrapolates to a full corpus. Prints a table and writes profile.json.

    sbatch scaling/profile.sbatch before 2000   (or: python scaling/profile_training.py --split <stored split> --n-papers 2000 --out <dir>)
"""
import argparse
import inspect
import json
import os
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--split", required=True)
p.add_argument("--n-papers", type=int, default=2000)
p.add_argument("--full-papers", type=int, default=70000, help="corpus size to extrapolate to (train+val+test)")
p.add_argument("--size", default="local")
p.add_argument("--batches", type=int, default=40)
p.add_argument("--arms", default="reference,jepa+rgat,jepa+gps,paper_as_text")
p.add_argument("--out", required=True)
p.add_argument("--workers", default="", help="e.g. 0,3,7: also time the batch pipeline with that many worker processes")
A = p.parse_args()
os.makedirs(A.out, exist_ok=True)
os.environ.setdefault("PAPER_A_INDEX_DIR", os.path.join(A.out, "index"))

from sci_jepa.config import PaperAConfig, SIZES, FAST, EXPERIMENTS  # noqa: E402
from sci_jepa.data.graph_index import get_index  # noqa: E402
from sci_jepa.data.instances import Sampler, collate  # noqa: E402
from sci_jepa.models import losses as L  # noqa: E402
from sci_jepa.models.model import PaperAModel, to_device  # noqa: E402
from sci_jepa.training.train import mask_plan, _autocast, _fp32, epoch_checkpoint, _batches  # noqa: E402

base = PaperAConfig().override(**SIZES[A.size], max_papers=A.n_papers, split_dir=A.split, epochs=1.0, device="auto",
                               batch_size=64 if A.size in ("tiny", "local") else 128, rows_per_shard=100,
                               index_dir=os.environ["PAPER_A_INDEX_DIR"], out_dir=A.out, **FAST)
dev = torch.device("cuda")
print("gpu:", torch.cuda.get_device_name(0), "| cpu threads:", torch.get_num_threads(), flush=True)
t = time.time()
gi = get_index(base)
S = gi.summary()
t_index = time.time() - t
print(f"index: {S['papers']} papers, {S['kus']} KUs in {t_index:.0f}s ({S['kus'] / max(t_index, 1e-9):.0f} KUs/s)",
      flush=True)
arms = dict(EXPERIMENTS["BENCH"])
res = {"gpu": torch.cuda.get_device_name(0), "papers": S["papers"], "kus": S["kus"], "index_s": t_index, "arms": {}}

for arm in A.arms.split(","):
    cfg = base.override(**arms[arm])
    if cfg.ku_mode != base.ku_mode:                     # paper_as_text needs its own index
        t = time.time(); g = get_index(cfg); t_ix = time.time() - t
    else:
        g, t_ix = gi, 0.0
    if g.comp_x.shape[1] != cfg.feat_dim:
        cfg = cfg.override(feat_dim=int(g.comp_x.shape[1]))
    torch.manual_seed(0)
    sampler = Sampler(g, cfg, "train", 0)
    n_steps, plan_fn, info = mask_plan(cfg, sampler)
    # CPU: batch building, split into the neighbourhood search and the rest (instance building + collate)
    nb_time = [0.0]
    orig_nb = sampler.neighbourhood

    def timed_nb(p0):
        t1 = time.time(); r = orig_nb(p0); nb_time[0] += time.time() - t1
        return r
    sampler.neighbourhood = timed_nb
    t = time.time(); cpu = []
    for s in range(A.batches):
        plan, _ = plan_fn(s)
        cpu.append(collate(sampler.sample(cfg.batch_size, plan)))
    t_cpu = (time.time() - t) / A.batches
    t_nb = nb_time[0] / A.batches
    sampler.neighbourhood = orig_nb
    # batches per second of the whole batch pipeline (thread or worker processes), where the code has workers
    pipe = {}
    if A.workers and "workers" in inspect.signature(_batches).parameters:
        for w in [int(x) for x in A.workers.split(",")]:
            smp = Sampler(g, cfg, "train", 1)
            it = _batches(smp, plan_fn, n_steps, cfg.batch_size, 2, 0, None, w, 1)
            for _ in range(8):
                next(it)                                # start-up (worker processes, first caches)
            t = time.time(); k = 0
            for _ in range(A.batches):
                try:
                    next(it); k += 1
                except StopIteration:
                    break
            pipe[f"workers={w}"] = (time.time() - t) / max(k, 1)
            it.close()
            print(f"    batch pipeline, {w} workers: {pipe[f'workers={w}'] * 1e3:7.1f} ms per batch", flush=True)

    # GPU: optimisation step on prebuilt batches, for the current settings and the A100 candidates
    def gpu_step_time(vcfg, tf32=False, fused=False):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32
        torch.manual_seed(0)
        model = PaperAModel(vcfg).to(dev)
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=vcfg.lr, fused=fused)
        scaler = torch.amp.GradScaler("cuda") if vcfg.amp == "fp16" else None

        def step(b):
            b = to_device(b, dev)
            with _autocast(vcfg, dev):
                out = model(b)
            out = _fp32(out)
            loss = L.jepa_loss(out["pred"], out["target"], vcfg.jepa_loss)
            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            model.ema(0.996)
        for b in cpu[:5]:
            step(b)                                     # warm-up
        torch.cuda.synchronize(); t1 = time.time()
        for b in cpu[5:]:
            step(b)
        torch.cuda.synchronize()
        return (time.time() - t1) / max(len(cpu) - 5, 1), model
    variants = {"current (fp16, manual attention)": (cfg, False, False),
                "bf16": (cfg.override(amp="bf16"), False, False),
                "bf16 + sdpa": (cfg.override(amp="bf16", attn_impl="sdpa"), False, False),
                "bf16 + sdpa + tf32 + fused AdamW": (cfg.override(amp="bf16", attn_impl="sdpa"), True, True)}
    gpu_var = {}
    for vname, (vcfg, tf32, fused) in variants.items():
        try:
            gpu_var[vname], model = gpu_step_time(vcfg, tf32, fused)
        except Exception as e:                          # noqa: BLE001
            gpu_var[vname] = float("nan"); print(f"  {vname} failed: {e!r}"[:200])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    t_gpu, model = gpu_step_time(cfg)                   # the evaluation below uses the current settings
    # one end-of-epoch validation
    t = time.time()
    try:
        epoch_checkpoint(model, g, cfg, dev, 1, 0, 0.0, 0.0)
        t_eval = time.time() - t
    except Exception as e:                              # noqa: BLE001
        t_eval = float("nan"); print("  eval failed:", repr(e)[:200])
    anchors = sum(info["anchors_per_type"].values())
    per_paper = anchors / max(len(g.papers("train")), 1)
    r = {"batch": cfg.batch_size, "anchors": anchors, "steps_per_epoch": n_steps, "cpu_batch_s": t_cpu,
         "cpu_neighbourhood_s": t_nb, "batch_pipeline_s": pipe, "gpu_step_s": t_gpu, "gpu_step_variants_s": gpu_var, "epoch_eval_s": t_eval,
         "index_s": t_ix, "anchors_per_train_paper": per_paper,
         "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["arms"][arm] = r
    print(f"{arm:16s} batch {cfg.batch_size}: CPU batch {t_cpu * 1e3:7.1f} ms (neighbourhood {t_nb * 1e3:.1f} ms) | "
          f"GPU step {t_gpu * 1e3:7.1f} ms | epoch eval {t_eval:6.1f} s | {n_steps} steps/epoch at {S['papers']} "
          f"papers | peak {r['gpu_mem_peak_gb']:.1f} GB", flush=True)
    for vname, v in gpu_var.items():
        print(f"    GPU step, {vname:34s} {v * 1e3:7.1f} ms", flush=True)
    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# extrapolation: steps per epoch grow with the train papers; the per-epoch validation is fixed size
scale = A.full_papers / max(S["papers"], 1)
print(f"\nextrapolated to {A.full_papers} papers (x{scale:.0f}), one epoch:")
for arm, r in res["arms"].items():
    steps = r["steps_per_epoch"] * scale
    serial = steps * (r["cpu_batch_s"] + r["gpu_step_s"])
    overlap = steps * max(r["cpu_batch_s"], r["gpu_step_s"])
    r["full_epoch_h_serial"], r["full_epoch_h_overlap"] = serial / 3600, overlap / 3600
    print(f"  {arm:16s} {steps:9.0f} steps: {serial / 3600:5.2f} h now (CPU and GPU in turn), "
          f"{overlap / 3600:5.2f} h if batch building fully overlaps the GPU, + eval {r['epoch_eval_s']:.0f} s")
res["index_full_h_est"] = t_index * scale / 3600
print(f"  KU index (text encoder): ~{res['index_full_h_est']:.1f} h for {A.full_papers} papers")
json.dump(res, open(os.path.join(A.out, "profile.json"), "w"), indent=1)
