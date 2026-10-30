"""
Tables and figures from scaling/bench_kit.sh results: previous vs optimized code, per GPU type.

    python scaling/bench_report.py <results dir with one sub-folder per GPU> <out dir>

Writes summary.csv, summary.md (one row per GPU x configuration x code version, every metric) and the
figures used in the slides (PNG, 200 dpi).
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# colours: neutral grey = previous code, blue = optimized; GPUs take the first three categorical slots
PREV, OPT = "#8a8984", "#2a78d6"
GPU_COLOR = {"H100": "#2a78d6", "A100": "#eb6834", "L40S": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def gpu_short(name):
    return next((k for k in GPU_COLOR if k in name), name)


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
    r = json.load(open(f))
    r["gpu_short"] = gpu_short(r["gpu"]["name"])
    r["config"] = f"{r['size']} b{r['batch']} {r['speed']}"
    r["concurrent"] = "concurrent" in os.path.basename(f)
    rows.append(r)
if not rows:
    sys.exit(f"no results under {ROOT}")
single = [r for r in rows if not r["concurrent"]]
gpus = [g for g in GPU_COLOR if any(r["gpu_short"] == g for r in rows)] + \
       sorted({r["gpu_short"] for r in rows} - set(GPU_COLOR))
CONFIG_ORDER = ["local b64 fast", "local b64 gpu", "S b128 gpu", "S b256 gpu", "M b128 gpu"]
configs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in single)]


def get(gpu, config, label, path):
    for r in single:
        if r["gpu_short"] == gpu and r["config"] == config and r["label"] == label:
            v = r
            for k in path.split("."):
                v = v.get(k) if isinstance(v, dict) else None
            return v
    return None


METRICS = [  # (key, label, unit, lower is better)
    ("time.ms_per_step", "Step time", "ms", True),
    ("time.predicted_kus_per_s", "Throughput", "predicted KUs/s", False),
    ("time.train_wall_s", "Training wall time (1 epoch, mini split)", "s", True),
    ("time.eval_wall_s", "Evaluation wall time (epoch-end + final test)", "s", True),
    ("time.d0_epoch_h", "One D0 epoch (60k papers), extrapolated", "GPU-h", True),
    ("compute.gpu_util_pct", "GPU utilisation (kernel running)", "%", False),
    ("compute.mem_ctrl_util_pct", "Memory-controller utilisation", "%", None),
    ("compute.achieved_tflops", "Achieved model TFLOP/s", "TFLOP/s", False),
    ("compute.mfu_pct", "Model FLOPs utilisation (MFU)", "%", False),
    ("compute.flops_per_step", "Model FLOPs per step", "FLOP", None),
    ("compute.cpu_util_pct", "Host CPU utilisation", "%", None),
    ("memory.peak_allocated_gb", "Peak GPU memory allocated", "GB", True),
    ("memory.peak_reserved_gb", "Peak GPU memory reserved", "GB", True),
    ("io.h2d_mb_per_step", "Host-to-device data per step", "MB", True),
    ("io.rel_mb_per_step", "  of which relation matrices", "MB", True),
    ("io.batch_pipeline_ms", "Batch pipeline (7 workers)", "ms/batch", True),
    ("io.pcie_rx_mb_s", "PCIe host-to-device traffic (NVML)", "MB/s", None),
    ("energy.train_j_per_step", "GPU energy per step", "J", True),
    ("energy.train_j_per_1k_kus", "GPU energy per 1k predicted KUs", "J", True),
    ("energy.train_power_w_mean", "Mean GPU power, training", "W", None),
    ("energy.d0_epoch_kwh", "GPU energy, one D0 epoch", "kWh", True),
]

# ---- tables -------------------------------------------------------------------------------------------------
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["gpu", "config", "metric", "unit", "previous", "optimized", "optimized/previous"])
    for g in gpus:
        for c in configs:
            for key, name, unit, _ in METRICS:
                a, b = get(g, c, "previous", key), get(g, c, "optimized", key)
                if a is None and b is None:
                    continue
                w.writerow([g, c, name, unit, a, b, (b / a) if a and b else None])
LABELS = ["previous", "optimized", "optimized+lightning"] + sorted({r["label"] for r in rows if "-ddp" in r["label"]})
NICE = {"previous": "Previous (own loop)", "optimized": "Optimized (own loop)",
        "optimized+lightning": "Optimized + Lightning, 1 GPU",
        **{l: f"Optimized + Lightning, {l.split('-ddp')[1]} GPUs" for l in LABELS if "-ddp" in l}}
md = ["# Paper A training: previous vs optimized code, own loop vs PyTorch Lightning\n",
      "Change columns are relative to the previous code. For the 4-GPU run, energy and power are the sum over "
      "the 4 GPUs and MFU is relative to 4x the peak.\n"]
for g in gpus:
    gname = next(r["gpu"]["name"] for r in rows if r["gpu_short"] == g)
    md.append(f"\n## {gname}\n")
    for c in configs:
        labs = [l for l in LABELS if get(g, c, l, "time.ms_per_step") is not None]
        md.append(f"\n### {c}\n\n| Metric | Unit | " + " | ".join(NICE[l] for l in labs) + " |\n|---|---|"
                  + "---|" * len(labs))
        for key, name, unit, lower in METRICS:
            vals = [get(g, c, l, key) for l in labs]
            if all(v is None for v in vals):
                continue
            a = vals[0]
            cells = []
            for l, v in zip(labs, vals):
                if v is None:
                    cells.append("–"); continue
                txt = f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}"
                if l != "previous" and a:
                    txt += f" ({(v / a - 1) * 100:+.1f}%)"
                cells.append(txt)
            md.append(f"| {name} | {unit} | " + " | ".join(cells) + " |")
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")


# ---- figures --------------------------------------------------------------------------------------------------
def paired_bars(key, title, unit, fname, fmt="{:.0f}", ratio=True):
    n = len(gpus)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False, sharey=False)
    for ax, g in zip(axes[0], gpus):
        cs = [c for c in configs if get(g, c, "previous", key) is not None]
        x = np.arange(len(cs))
        a = [get(g, c, "previous", key) or 0 for c in cs]
        b = [get(g, c, "optimized", key) or 0 for c in cs]
        ax.bar(x - 0.19, a, 0.36, color=PREV, label="previous")
        ax.bar(x + 0.19, b, 0.36, color=OPT, label="optimized")
        top = max(a + b) if (a + b) else 1
        for xi, (ai, bi) in enumerate(zip(a, b)):
            ax.text(xi - 0.19, ai + top * 0.02, fmt.format(ai), ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.text(xi + 0.19, bi + top * 0.02, fmt.format(bi), ha="center", va="bottom", fontsize=7.5, color=INK)
            if ratio and ai and bi:
                ax.text(xi, top * 1.13, f"{ai / bi:.2f}x" if key.startswith(("time.ms", "energy", "memory", "io"))
                        else f"{bi / ai:.2f}x", ha="center", fontsize=8.5, fontweight="bold", color=INK)
        ax.set_ylim(0, top * 1.25)
        ax.set_xticks(x, cs, rotation=0, fontsize=8)
        ax.set_title(g)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


paired_bars("time.ms_per_step", "Training step time (lower is better)", "ms per step", "fig_step_time.png", "{:.1f}")
paired_bars("energy.train_j_per_1k_kus", "GPU energy per 1,000 predicted KUs (lower is better)", "J",
            "fig_energy.png", "{:.0f}")
paired_bars("memory.peak_allocated_gb", "Peak GPU memory allocated during training", "GB", "fig_memory.png",
            "{:.1f}", ratio=False)
paired_bars("compute.gpu_util_pct", "GPU utilisation during training (share of time a kernel runs)", "%",
            "fig_util.png", "{:.0f}", ratio=False)
paired_bars("compute.mfu_pct", "Model FLOPs utilisation (MFU, % of dense bf16 peak)", "%", "fig_mfu.png",
            "{:.1f}", ratio=False)
paired_bars("io.h2d_mb_per_step", "Data copied host -> GPU per training step", "MB", "fig_io.png", "{:.1f}")

# GPU comparison: the paper's model (S, batch 128, SPEED=gpu), both code versions
cfg_cmp = "S b128 gpu"
panels = [("time.ms_per_step", "Step time", "ms"), ("energy.train_j_per_1k_kus", "Energy / 1k KUs", "J"),
          ("time.d0_epoch_h", "One D0 epoch", "GPU-hours"), ("energy.d0_epoch_kwh", "Energy, one D0 epoch", "kWh")]
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0))
for ax, (key, name, unit) in zip(axes, panels):
    x = np.arange(len(gpus))
    a = [get(g, cfg_cmp, "previous", key) or 0 for g in gpus]
    b = [get(g, cfg_cmp, "optimized", key) or 0 for g in gpus]
    ax.bar(x - 0.19, a, 0.36, color=PREV, label="previous")
    ax.bar(x + 0.19, b, 0.36, color=[GPU_COLOR.get(g, OPT) for g in gpus], label="optimized")
    top = max(a + b) if (a + b) else 1
    for xi, bi in enumerate(b):
        ax.text(xi + 0.19, bi + top * 0.02, f"{bi:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, gpus)
    ax.set_title(name)
    ax.set_ylabel(unit)
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("GPU comparison, S model (paper size), batch 128, SPEED=gpu: grey = previous code",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_gpu_compare.png"))
plt.close(fig)

# quality: training loss curves of both versions (same seed), per GPU, S b128
fig, axes = plt.subplots(1, len(gpus), figsize=(4.2 * len(gpus), 3.0), squeeze=False)
for ax, g in zip(axes[0], gpus):
    for label, col, lw in (("previous", PREV, 3.0), ("optimized", OPT, 1.4)):
        h = get(g, cfg_cmp, label, "quality.history") or []
        if h:
            ax.plot([r["step"] for r in h], [r["jepa"] for r in h], color=col, lw=lw, label=label,
                    solid_capstyle="round")
    ax.set_title(g)
    ax.set_xlabel("training step")
    ax.set_ylabel("JEPA loss")
axes[0][0].legend(frameon=False, fontsize=8)
fig.suptitle("Training loss, same seed: previous vs optimized (S, batch 128)", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_quality.png"))
plt.close(fig)

# scalability: model size and batch size (optimized / previous throughput), and 4 runs per node
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
sc = [c for c in ["local b64 gpu", "S b128 gpu", "S b256 gpu", "M b128 gpu"] if c in configs]
x = np.arange(len(sc))
width = 0.8 / max(len(gpus), 1)
for i, g in enumerate(gpus):
    sp = [(get(g, c, "previous", "time.ms_per_step") or np.nan) / (get(g, c, "optimized", "time.ms_per_step") or np.nan)
          for c in sc]
    axes[0].bar(x + (i - (len(gpus) - 1) / 2) * width, sp, width * 0.9, color=GPU_COLOR.get(g, OPT), label=g)
    for xi, v in enumerate(sp):
        if np.isfinite(v):
            axes[0].text(xi + (i - (len(gpus) - 1) / 2) * width, v + 0.03, f"{v:.2f}x", ha="center", fontsize=7.5,
                         color=INK)
axes[0].axhline(1.0, color=INK2, lw=0.8)
axes[0].set_xticks(x, sc, fontsize=8)
axes[0].set_ylabel("speed-up (previous / optimized step time)")
axes[0].set_title("Speed-up by model and batch size")
axes[0].grid(axis="x", visible=False)
axes[0].legend(frameon=False, fontsize=8)
conc = [r for r in rows if r["concurrent"]]
ax = axes[1]
for i, g in enumerate(gpus):
    for j, (label, col) in enumerate((("previous", PREV), ("optimized", GPU_COLOR.get(g, OPT)))):
        rs = [r for r in conc if r["gpu_short"] == g and r["label"] == label]
        one = get(g, "S b128 gpu", label, "time.predicted_kus_per_s")
        if rs and one:
            node = sum(r["time"]["predicted_kus_per_s"] for r in rs)
            ax.bar(i * 2.4 + j * 1.0, node, 0.9, color=col)
            ax.text(i * 2.4 + j * 1.0, node * 1.02, f"{node / one:.2f}x of 1 GPU", ha="center", fontsize=7.5,
                    color=INK)
ax.set_xticks([i * 2.4 + 0.5 for i in range(len(gpus))], gpus)
ax.set_ylabel("predicted KUs/s per node")
ax.set_title("4 trainings at once, one per GPU (grey = previous)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_scaling.png"))
plt.close(fig)
# ---- data for the slide deck (native charts): every metric per GPU x configuration x version -------------------
deck = {"gpus": gpus, "configs": configs, "gpu_info": {}, "metrics": {}, "history": {}, "final": {}, "concurrent": {}}
for g in gpus:
    r0 = next(r for r in rows if r["gpu_short"] == g)
    deck["gpu_info"][g] = r0["gpu"]
    for c in configs:
        for label in LABELS:
            deck["metrics"][f"{g}|{c}|{label}"] = {key: get(g, c, label, key) for key, *_ in METRICS}
            deck["final"][f"{g}|{c}|{label}"] = get(g, c, label, "quality.final")
            deck["history"][f"{g}|{c}|{label}"] = get(g, c, label, "quality.history")
    for label in ("previous", "optimized"):
        rs = [r for r in rows if r["concurrent"] and r["gpu_short"] == g and r["label"] == label]
        if rs:
            deck["concurrent"][f"{g}|{label}"] = {"runs": len(rs),
                                                  "kus_per_s": [r["time"]["predicted_kus_per_s"] for r in rs],
                                                  "ms_per_step": [r["time"]["ms_per_step"] for r in rs]}
json.dump(deck, open(os.path.join(OUT, "deck_data.json"), "w"), indent=1, default=float)
print(f"wrote {OUT}: summary.csv, summary.md, deck_data.json, figures for {gpus}")
