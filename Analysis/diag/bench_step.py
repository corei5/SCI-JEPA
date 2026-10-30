"""Speed benchmark of one Graph-JEPA training step (papers, Part 1), split into its two halves.

1. Data: batches per second the DataLoader delivers (online METIS transform + collate) for several
   worker counts, without the GPU.
2. GPU: forward + backward + optimizer + EMA on batches already on the GPU, for each speed switch
   (core/config.py `train.*`) on its own and all together, in fp32 and bf16 autocast.

    cd Analysis && python -m diag.bench_step --steps 30 --workers 8,16,30 --out bench.json

The switches do not change the loss; the "loss" column shows it per variant on the same batches.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from core.config import cfg as base_cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from core.objectives import vicreg_step
from core.trainer import k_fold

OFF = dict(reuse_online_pass=False, foreach_ema=False, fused_optim=False)
VARIANTS = [
    ("baseline (all switches off)", OFF, False),
    ("+ reuse online pass", {**OFF, "reuse_online_pass": True}, False),
    ("+ foreach EMA", {**OFF, "foreach_ema": True}, False),
    ("+ fused Adam", {**OFF, "fused_optim": True}, False),
    ("all switches", {}, False),
    ("all switches + bf16 autocast", {}, True),
]


def loader_rate(train_set, batch_size, workers, n_batches):
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=True, persistent_workers=False,
                        **({"prefetch_factor": 4} if workers > 0 else {}))
    it = iter(loader)
    next(it)                                            # worker start-up
    t = time.perf_counter()
    k = 0
    for _ in range(n_batches):
        try:
            next(it)
            k += 1
        except StopIteration:
            break
    rate = k / (time.perf_counter() - t)
    del it, loader
    return rate


def gpu_step_time(cfg, batches, overrides, bf16, steps, warmup=5):
    c = cfg.clone()
    for key, value in overrides.items():
        setattr(c.train, key, value)
    torch.manual_seed(0)
    model = create_model(c).cuda().train()
    fused = {"fused": True} if c.train.fused_optim else {}
    opt = torch.optim.Adam(model.parameters(), lr=float(c.train.lr), weight_decay=float(c.train.wd), **fused)
    def step(batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = vicreg_step(model, batch, c, training=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(c.train.grad_clip or 1.0))
        opt.step()
        model.update_target(0.996)
        return out["loss"]

    first = float(step(batches[0]).detach())             # same weights and batch for every variant
    for i in range(1, warmup):
        step(batches[i % len(batches)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for i in range(steps):
        step(batches[i % len(batches)])
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - t) / steps
    return seconds, torch.cuda.max_memory_allocated() / 1e9, first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="papers.yaml")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batches", type=int, default=10, help="distinct batches cycled in the GPU timing")
    p.add_argument("--workers", default="8,16,30")
    p.add_argument("--out", default="")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = update_cfg(base_cfg, f"--config {a.config} " + " ".join(a.opts))
    torch.set_float32_matmul_precision("high")          # as the Lightning engine sets it

    dataset, transform, _ = create_dataset(cfg)
    train_idx, _ = k_fold(dataset, 2)
    train_set = dataset[train_idx[0]]
    train_set.transform = transform
    bs = int(cfg.train.batch_size)
    print(f"gpu {torch.cuda.get_device_name(0)} | {len(train_set)} train graphs, batch {bs}, "
          f"{len(train_set) // bs} steps/epoch", flush=True)
    res = {"gpu": torch.cuda.get_device_name(0), "batch_size": bs, "train_graphs": len(train_set),
           "loader_batches_per_s": {}, "gpu_step": {}}

    for w in [int(x) for x in a.workers.split(",") if x != ""]:
        rate = loader_rate(train_set, bs, w, n_batches=min(20, len(train_set) // bs - 1))
        res["loader_batches_per_s"][w] = rate
        print(f"data: {w:2d} workers -> {rate:6.2f} batches/s ({1e3 / rate:7.1f} ms per batch)", flush=True)

    np.random.seed(0)
    torch.manual_seed(0)
    it = iter(DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=8))
    batches = [next(it).cuda() for _ in range(a.batches)]
    for name, overrides, bf16 in VARIANTS:
        sec, mem, loss = gpu_step_time(cfg, batches, overrides, bf16, a.steps)
        res["gpu_step"][name] = {"ms": sec * 1e3, "peak_gb": mem, "first_loss": loss}
        print(f"gpu: {name:30s} {sec * 1e3:7.1f} ms/step | peak {mem:5.2f} GB | first-step loss {loss:.4f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
