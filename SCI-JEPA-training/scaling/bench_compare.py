"""
Efficiency of one Paper A training run on the MAIN pipeline (sci_jepa.training.train.train_one): one full run
(training epochs + end-of-epoch validation + final test evaluation) on a stored split, measured from outside.
The code version is whatever `sci_jepa` resolves to, so the same script times the previous and the optimized
code (PYTHONPATH=<that version's src>); scaling/bench_kit.sh does this for both and every configuration.

    python scaling/bench_compare.py --label optimized --size S --batch 128 --epochs 1 \
        --split <stored split> --index-dir <dir> --out <dir>

Writes <out>/<label>_<size>_b<batch>_<speed>.json with
  time        steady-state steps/s, instances/s, predicted KUs/s (from the run's own step log), training and
              evaluation wall time, extrapolation to a D0 epoch (60k train papers)
  compute     NVML GPU utilisation (time share with a kernel running), memory-controller utilisation, SM
              clock, CPU utilisation; model FLOPs per step (torch FlopCounterMode) and MFU vs the GPU's peak
  memory      peak allocated / reserved by PyTorch during training, NVML device memory used
  data I/O    bytes copied host->device per step (the collated batch), NVML PCIe rx/tx, batch pipeline ms
  energy      NVML energy counter: J per step, J per 1k predicted KUs, mean / max power, kWh per D0 epoch
  quality     the run's training loss curve and final validation / test metrics (same seed in both versions)
"""
import argparse
import json
import os
import platform
import re
import tempfile
import threading
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--label", required=True, help="code version label, e.g. previous / optimized")
p.add_argument("--size", default="local")
p.add_argument("--batch", type=int, default=0, help="0 = 64 for tiny/local, 128 otherwise (as sci-jepa run)")
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--speed", default="gpu", choices=["gpu", "fast"], help="SPEED preset (config.FAST_GPU / FAST)")
p.add_argument("--workers", type=int, default=7, help="batch worker processes (cfg.batch_workers)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split", required=True)
p.add_argument("--index-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--d0-train-papers", type=int, default=60000, help="corpus to extrapolate one epoch to")
p.add_argument("--tag", default="", help="extra suffix of the result file (e.g. concurrency run id)")
p.add_argument("--trainer", default="native", choices=["native", "lightning"], help="training loop (cfg.trainer)")
p.add_argument("--devices", type=int, default=1, help="GPUs for one run (trainer=lightning, DDP)")
A = p.parse_args()
os.makedirs(A.out, exist_ok=True)

import pynvml  # noqa: E402

from sci_jepa import config as C  # noqa: E402
from sci_jepa.data.graph_index import get_index  # noqa: E402
from sci_jepa.data.instances import Sampler, collate  # noqa: E402
from sci_jepa.models import losses as L  # noqa: E402
from sci_jepa.models.model import PaperAModel, to_device  # noqa: E402
from sci_jepa.training import train as T  # noqa: E402

batch = A.batch or (64 if A.size in ("tiny", "local") else 128)
speed = C.FAST_GPU if A.speed == "gpu" else C.FAST
work = tempfile.mkdtemp(prefix="bench_")
# parquet_dirs is part of the index fingerprint; a fixed value lets the bundled index load on any machine
cfg = C.PaperAConfig().override(**C.SIZES[A.size], **speed, batch_size=batch, max_papers=6000, split_dir=A.split,
                                parquet_dirs=("sci-jepa-mini/normalized/arxiv",), index_dir=A.index_dir,
                                out_dir=work, device="cuda", epochs=A.epochs, batch_workers=A.workers,
                                log_every=25, rows_per_shard=100,
                                **({} if A.trainer == "native" and A.devices == 1 else   # the previous code has no
                                   {"trainer": A.trainer, "devices": A.devices}))        # trainer / devices keys
variant = A.label + ("" if A.trainer == "native" else "+lightning" + (f"-ddp{A.devices}" if A.devices > 1 else ""))
RANK = int(os.environ.get("LOCAL_RANK", 0))
gi = get_index(cfg)
if gi.comp_x.shape[1] != cfg.feat_dim:
    cfg = cfg.override(feat_dim=int(gi.comp_x.shape[1]))

# ---- GPU identity (NVML handle of the GPU this process uses, also under CUDA_VISIBLE_DEVICES) -------------------
pynvml.nvmlInit()
props = torch.cuda.get_device_properties(0)
uuid = str(props.uuid)
H = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
HS = [H]
for k in range(1, A.devices):                          # the other GPUs of a multi-GPU run (energy, utilisation)
    u = str(torch.cuda.get_device_properties(k).uuid)
    HS.append(pynvml.nvmlDeviceGetHandleByUUID(u if u.startswith("GPU-") else "GPU-" + u))


def _nv(fn, *a, default=None):
    try:
        return fn(*a)
    except pynvml.NVMLError:
        return default


gpu = {"name": torch.cuda.get_device_name(0), "memory_gb": props.total_memory / 1e9, "sms": props.multi_processor_count,
       "power_limit_w": (_nv(pynvml.nvmlDeviceGetEnforcedPowerLimit, H, default=0) or 0) / 1e3,
       "driver": _nv(pynvml.nvmlSystemGetDriverVersion), "torch": torch.__version__, "cuda": torch.version.cuda,
       "host": platform.node(), "cpus": os.cpu_count()}
def energy():                                          # mJ since driver load, summed over the run's GPUs
    vals = [_nv(pynvml.nvmlDeviceGetTotalEnergyConsumption, h, default=None) for h in HS]
    return None if any(v is None for v in vals) else sum(vals)


# ---- NVML sampler (background thread, 10 Hz) -----------------------------------------------------------------
class Probe(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()
        try:
            import psutil
            self.ps = psutil
            psutil.cpu_percent(None)
        except ImportError:
            self.ps = None

    def run(self):
        while not self.stop.is_set():
            us = [_nv(pynvml.nvmlDeviceGetUtilizationRates, h) for h in HS]
            u = type("U", (), {"gpu": float(np.mean([x.gpu for x in us if x])), "memory": float(np.mean(
                [x.memory for x in us if x]))})() if any(us) else None
            m = _nv(pynvml.nvmlDeviceGetMemoryInfo, H)
            self.rows.append({
                "t": time.perf_counter(), "power_w": sum((_nv(pynvml.nvmlDeviceGetPowerUsage, h, default=0) or 0)
                                                         for h in HS) / 1e3,
                "util_gpu": u.gpu if u else None, "util_mem": u.memory if u else None,
                "mem_used_gb": m.used / 1e9 if m else None,
                "sm_clock_mhz": _nv(pynvml.nvmlDeviceGetClockInfo, H, pynvml.NVML_CLOCK_SM),
                "pcie_rx_mbs": (_nv(pynvml.nvmlDeviceGetPcieThroughput, H, pynvml.NVML_PCIE_UTIL_RX_BYTES) or 0) / 1e3,
                "pcie_tx_mbs": (_nv(pynvml.nvmlDeviceGetPcieThroughput, H, pynvml.NVML_PCIE_UTIL_TX_BYTES) or 0) / 1e3,
                "cpu_pct": self.ps.cpu_percent(None) if self.ps else None})
            self.stop.wait(0.1)


# ---- markers from the run's own log: step lines, the step count, the epoch-end validation ---------------------
marks, info = [], {"n_steps": None, "train_end": None, "lines": []}


def log(line):
    info["lines"].append(line)
    s = re.search(r"\bstep=(\d+)\b", line)
    if "[mask]" in line:
        m = re.search(r"steps=(\d+)", line)
        info["n_steps"] = int(m.group(1)) if m else None
    elif s:
        step = int(s.group(1))
        torch.cuda.synchronize()
        now = time.perf_counter()
        if not marks:
            torch.cuda.reset_peak_memory_stats()
        marks.append((step, now, energy()))
        if info["n_steps"] is not None and step == info["n_steps"] - 1:
            info["train_end"] = (now, energy(), torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    print(line, flush=True)


probe = Probe()
probe.start()
t_start, e_start = time.perf_counter(), energy()
metrics = T.train_one(cfg, gi, A.seed, log=log)
if metrics is None:                                    # rank > 0 of a multi-GPU run: rank 0 measures and reports
    probe.stop.set()
    raise SystemExit(0)
torch.cuda.synchronize()
t_end, e_end = time.perf_counter(), energy()
probe.stop.set()
probe.join()

# ---- time and energy from the step marks (steady state: from step 50 on) ----------------------------------------
steady = [m for m in marks if m[0] >= 50] or marks
(s0, t0, e0), (s1, t1, e1) = steady[0], steady[-1]
steps_per_s = (s1 - s0) / max(t1 - t0, 1e-9)
win = [(b[0] - a[0]) / (b[1] - a[1]) for a, b in zip(steady, steady[1:]) if b[1] > a[1]]
train_end_t, train_end_e, peak_alloc, peak_res = info["train_end"] or (t1, e1, torch.cuda.max_memory_allocated(),
                                                                        torch.cuda.max_memory_reserved())
first_t = marks[0][1]
in_train = [r for r in probe.rows if first_t <= r["t"] <= train_end_t]
in_eval = [r for r in probe.rows if r["t"] > train_end_t]
mean = lambda rows, k: float(np.mean([r[k] for r in rows if r[k] is not None])) if rows else None
pmax = lambda rows, k: float(np.max([r[k] for r in rows if r[k] is not None])) if rows else None
j_per_step = (e1 - e0) / 1e3 / max(s1 - s0, 1) if e0 is not None and e1 is not None else None

# ---- one batch: bytes host->device, FLOPs per training step, batch pipeline speed -------------------------------
torch.manual_seed(A.seed)
smp = Sampler(gi, cfg, "train", A.seed)
cpu_batch = collate(smp.sample(cfg.batch_size))


def nbytes(o):
    if torch.is_tensor(o):
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(nbytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(nbytes(v) for v in o)
    return 0


h2d = nbytes(cpu_batch)
rel_bytes = nbytes(cpu_batch["ctx"]["rel"]) + nbytes(cpu_batch["tgt"]["rel"])
dev = torch.device("cuda")
model = PaperAModel(cfg).to(dev)
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
b = to_device(cpu_batch, dev)
with FlopCounterMode(display=False) as fc:
    with T._autocast(cfg, dev):
        out = model(b)
    out = T._fp32(out)
    L.jepa_loss(out["pred"], out["target"], cfg.jepa_loss).backward()
flops_step = int(fc.get_total_flops())
del model, b, out
n_steps_plan, plan_fn, _ = T.mask_plan(cfg, smp)
it = T._batches(Sampler(gi, cfg, "train", A.seed + 1), plan_fn, n_steps_plan, cfg.batch_size, 2, 0, None,
                A.workers, A.seed + 1)
for _ in range(8):
    next(it)
tb = time.perf_counter()
k = 0
for _ in range(30):
    try:
        next(it)
        k += 1
    except StopIteration:
        break
pipeline_ms = (time.perf_counter() - tb) / max(k, 1) * 1e3
it.close()

# ---- peaks (dense, no sparsity) of the GPUs we compare; vendor datasheets ---------------------------------------
PEAK_BF16_TFLOPS = {"H100": 989.0 if "SXM" in gpu["name"] or "HBM3" in gpu["name"] or gpu["power_limit_w"] > 500
                    else 756.0, "A100": 312.0, "L40S": 362.0}
peak = next((v for k_, v in PEAK_BF16_TFLOPS.items() if k_ in gpu["name"]), None)
n_train = len(gi.papers("train"))
steps_epoch = info["n_steps"] / max(A.epochs, 1e-9) if info["n_steps"] else None
d0_steps = steps_epoch * A.d0_train_papers / max(n_train, 1) if steps_epoch else None
kus_per_step = cfg.batch_size * max(cfg.targets_per_instance, 1)
final = {k_: v for k_, v in metrics.items() if isinstance(v, (int, float, str)) and k_ not in ("model_dir",)}
for k_ in ("retrieval_balanced", "retrieval_standard"):          # test MRR / Hits (the paper's headline metric)
    if isinstance(metrics.get(k_), dict):
        final[k_] = {m: v for m, v in metrics[k_].items() if isinstance(v, (int, float))}
if isinstance(metrics.get("tasks"), dict):                       # the five reasoning tasks, MRR / chance
    final["tasks"] = {t: {m: v for m, v in d.items() if isinstance(v, (int, float))}
                      for t, d in metrics["tasks"].items() if isinstance(d, dict)}
res = {
    "label": variant, "code": A.label, "trainer": A.trainer, "devices": A.devices, "size": A.size, "batch": cfg.batch_size, "speed": A.speed, "epochs": A.epochs,
    "workers": A.workers, "seed": A.seed, "gpu": gpu, "train_papers": n_train, "n_steps": info["n_steps"],
    "time": {"steps_per_s": steps_per_s, "window_steps_per_s_p10_p50_p90": list(np.percentile(win, [10, 50, 90]))
             if win else None, "ms_per_step": 1e3 / steps_per_s, "instances_per_s": steps_per_s * cfg.batch_size,
             "predicted_kus_per_s": steps_per_s * kus_per_step, "train_wall_s": train_end_t - first_t,
             "eval_wall_s": t_end - train_end_t, "run_wall_s": t_end - t_start,
             "d0_epoch_steps": d0_steps, "d0_epoch_h": d0_steps / steps_per_s / 3600 if d0_steps else None},
    "compute": {"gpu_util_pct": mean(in_train, "util_gpu"), "mem_ctrl_util_pct": mean(in_train, "util_mem"),
                "sm_clock_mhz": mean(in_train, "sm_clock_mhz"), "cpu_util_pct": mean(in_train, "cpu_pct"),
                "flops_per_step": flops_step, "achieved_tflops": flops_step * steps_per_s / 1e12,
                "peak_bf16_tflops": peak * A.devices if peak else None, "devices": A.devices,
                "mfu_pct": 100 * flops_step * steps_per_s / 1e12 / (peak * A.devices) if peak else None,
                "eval_gpu_util_pct": mean(in_eval, "util_gpu")},
    "memory": {"peak_allocated_gb": peak_alloc / 1e9, "peak_reserved_gb": peak_res / 1e9,
               "nvml_used_max_gb": pmax(in_train, "mem_used_gb"), "device_gb": gpu["memory_gb"]},
    "io": {"h2d_mb_per_step": h2d / 1e6, "rel_mb_per_step": rel_bytes / 1e6,
           "h2d_gb_per_s_needed": h2d * steps_per_s / 1e9, "pcie_rx_mb_s": mean(in_train, "pcie_rx_mbs"),
           "pcie_tx_mb_s": mean(in_train, "pcie_tx_mbs"), "batch_pipeline_ms": pipeline_ms},
    "energy": {"train_j_per_step": j_per_step,
               "train_j_per_1k_kus": j_per_step / kus_per_step * 1e3 if j_per_step else None,
               "train_power_w_mean": (e1 - e0) / 1e3 / max(t1 - t0, 1e-9) if e0 is not None else None,
               "train_power_w_max": pmax(in_train, "power_w"),
               "eval_j": (e_end - train_end_e) / 1e3 if e_end is not None and train_end_e is not None else None,
               "run_j": (e_end - e_start) / 1e3 if e_end is not None and e_start is not None else None,
               "d0_epoch_kwh": j_per_step * d0_steps / 3.6e6 if j_per_step and d0_steps else None},
    "quality": {"history": [{k_: h[k_] for k_ in ("step", "jepa", "inverse_probe") if k_ in h}
                            for h in metrics.get("history", [])],
                "final": final, "epoch_results": [{k_: v for k_, v in r.items() if isinstance(v, (int, float))}
                                                  for r in metrics.get("epoch_results", [])]},
    "samples": probe.rows[::5],
}
name = f"{variant}_{A.size}_b{cfg.batch_size}_{A.speed}{('_' + A.tag) if A.tag else ''}.json"
json.dump(res, open(os.path.join(A.out, name), "w"), indent=1, default=float)
print(f"[bench] {name}: {res['time']['ms_per_step']:.1f} ms/step | {res['time']['predicted_kus_per_s']:.0f} KUs/s | "
      f"GPU util {res['compute']['gpu_util_pct']} % | MFU {res['compute']['mfu_pct']} % | peak "
      f"{res['memory']['peak_allocated_gb']:.2f} GB | {res['energy']['train_j_per_step']} J/step", flush=True)
