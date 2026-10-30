"""
Operator-level trace of one SCI-JEPA training step (torch.profiler): which operations take the GPU and CPU time.
Uses the index cache of a profile run (scaling/runs/profile_<tag>/index). Prints the batch shapes and the top
operations, and writes a Chrome trace (open in chrome://tracing or https://ui.perfetto.dev).

    python scaling/trace_step.py --split <stored split> --index <dir> --out <dir> [--arm reference]
"""
import argparse
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

p = argparse.ArgumentParser()
p.add_argument("--split", required=True)
p.add_argument("--index", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n-papers", type=int, default=2000)
p.add_argument("--arm", default="reference")
p.add_argument("--speed", default="fast", choices=["fast", "gpu"])
p.add_argument("--size", default="local", help="model size (config.SIZES); batch 64 for tiny/local, else 128")
p.add_argument("--label", default="", help="code version label: also writes ops_<label>_<arm>_<speed>_<size>.json")
A = p.parse_args()
os.makedirs(A.out, exist_ok=True)

from sci_jepa import config as C  # noqa: E402
from sci_jepa.data.graph_index import get_index  # noqa: E402
from sci_jepa.data.instances import Sampler, collate  # noqa: E402
from sci_jepa.models import losses as L  # noqa: E402
from sci_jepa.models.model import PaperAModel, to_device  # noqa: E402
from sci_jepa.training.train import _autocast, _fp32  # noqa: E402

speed = C.FAST if A.speed == "fast" else C.FAST_GPU
cfg = C.PaperAConfig().override(**C.SIZES[A.size], max_papers=A.n_papers, split_dir=A.split, epochs=1.0, device="auto",
                                batch_size=64 if A.size in ("tiny", "local") else 128, rows_per_shard=100, index_dir=A.index, out_dir=A.out, **speed)
cfg = cfg.override(**dict(C.EXPERIMENTS["BENCH"])[A.arm])
gi = get_index(cfg)
if gi.comp_x.shape[1] != cfg.feat_dim:
    cfg = cfg.override(feat_dim=int(gi.comp_x.shape[1]))
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(getattr(cfg, "tf32", False))
smp = Sampler(gi, cfg, "train", 0)
batches = [collate(smp.sample(cfg.batch_size)) for _ in range(16)]


def shapes(d, pre=""):
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"  {pre}{k}: {tuple(v.shape)} {v.dtype} {v.numel() * v.element_size() / 1e6:.1f} MB")
        elif isinstance(v, dict):
            shapes(v, pre + k + ".")
        elif isinstance(v, tuple) and v and torch.is_tensor(v[0]):
            print(f"  {pre}{k}: {len(v)} x {tuple(v[0].shape)}")
print("batch:"); shapes(batches[0])

model = PaperAModel(cfg).to(dev)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=cfg.lr,
                        **({"fused": True} if getattr(cfg, "fused_adamw", False) else {}))
scaler = torch.amp.GradScaler("cuda") if cfg.amp == "fp16" else None


def step(b):
    with record_function("to_device"):
        b = to_device(b, dev)
    with record_function("forward"):
        with _autocast(cfg, dev):
            out = model(b)
        out = _fp32(out)
        loss = L.jepa_loss(out["pred"], out["target"], cfg.jepa_loss)
        if "inv_logits" in out:
            bb, i, j, _ = b["probe"]
            h = out["h"].detach()
            loss = loss + L.inverse_loss(model.inverse(h[bb, i], h[bb, j]), out["inv_labels"])
    with record_function("backward"):
        opt.zero_grad(set_to_none=True)
        (scaler.scale(loss) if scaler else loss).backward()
    with record_function("optimizer"):
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], cfg.grad_clip)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
    with record_function("ema"):
        model.ema(0.996)


for b in batches[:4]:
    step(b)
torch.cuda.synchronize(); t = time.time()
for b in batches[4:]:
    step(b)
torch.cuda.synchronize(); print(f"step: {(time.time() - t) / (len(batches) - 4) * 1e3:.1f} ms ({A.arm}, {A.speed}, {A.size})")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for b in batches[4:10]:
        step(b)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=50))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, max_name_column_width=50))
prof.export_chrome_trace(os.path.join(A.out, f"trace_{A.arm}_{A.speed}_{A.size}.json"))
if A.label:                                             # per-operation device time, for before/after breakdowns
    import json
    ops = [{"name": e.key, "self_device_us": getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)),
            "device_us": getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)), "calls": e.count,
            "self_cpu_us": e.self_cpu_time_total} for e in prof.key_averages()]
    json.dump({"label": A.label, "arm": A.arm, "speed": A.speed, "size": A.size, "steps": 6,
               "gpu": torch.cuda.get_device_name(0), "ops": ops},
              open(os.path.join(A.out, f"ops_{A.label}_{A.arm}_{A.speed}_{A.size}.json"), "w"), indent=1)
