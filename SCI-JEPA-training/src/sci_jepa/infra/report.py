"""Training-infrastructure report of one experiment: hardware / software, compute per method, timelines.

    from sci_jepa.infra.report import write_all
    write_all("runs/paper_a_results/BENCH")      # infrastructure.{json,md,tex} + figures/infra_*.{png,pdf}

Also called by reporting.report_pdf (PDF pages) and reporting.figures (the LLM comparison figure).
Compute is counted per run from its own timer (train_sec: training + evaluation of a model, or the whole evaluation
of a comparison system) on ONE GPU or GPU slice; a MIG slice gx.yyGB is converted to A100-equivalents as x / 7.
"""
import glob
import json
import os
import re
import time

import numpy as np

FAMILY = {"jepa": "JEPA (graph transformer)", "gnn": "JEPA + GNN", "comparison": "comparison system"}
FAMCOL = {"jepa": "#2a78d6", "gnn": "#eb6834", "comparison": "#1baf7a"}


def _results_dir(exp_dir):
    return os.path.dirname(os.path.abspath(exp_dir.rstrip("/")))


def load_systems(results_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "infra", "*", "system.json"))):
        try:
            out.append(json.load(open(f)))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def load_timeseries(results_dir, jobs=None):
    import pandas as pd
    frames = []
    for f in sorted(glob.glob(os.path.join(results_dir, "infra", "*", "timeseries*.csv"))):
        job = os.path.basename(os.path.dirname(f))
        if jobs and job not in jobs:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:                               # noqa: BLE001
            continue
        d["job"] = job
        frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).sort_values("t")


def _mig_fraction(name):
    """'NVIDIA A100-SXM4-80GB MIG 3g.40gb' -> 3/7; a full GPU -> 1."""
    m = re.search(r"(\d)g\.\d+gb", name or "")
    return int(m.group(1)) / 7 if m else 1.0


def hardware(systems):
    """One table of what the jobs ran on (values that differ between jobs are listed together)."""
    if not systems:
        return {}
    def vals(f):
        v = []
        for s in systems:
            try:
                x = f(s)
            except Exception:                           # noqa: BLE001
                x = None
            if x not in (None, "", []) and x not in v:
                v.append(x)
        return ", ".join(str(x) for x in v) if v else "–"
    gpu_phys = vals(lambda s: "; ".join(sorted({g["name"] for g in s["gpus"]["physical"]})))
    slices = vals(lambda s: "; ".join(sorted({m["name"].split("MIG ")[-1] for m in s["gpus"]["mig"]})) or None)
    return {
        "GPU (physical)": gpu_phys,
        "GPU slices (MIG) per job": slices,
        "GPUs per job": vals(lambda s: len(s["torch"]["cuda_devices_visible"]) or len(s["gpus"]["mig"]) or None),
        "GPU memory per device (GB)": vals(lambda s: "; ".join(str(m["memory_gb"]) for m in s["gpus"]["mig"])
                                           or "; ".join(str(g["memory_gb"]) for g in s["gpus"]["physical"])),
        "NVIDIA driver / CUDA driver": vals(lambda s: f"{s['gpus']['driver']} / {s['gpus']['cuda_driver']}"
                                            if s["gpus"].get("driver") else None),
        "CPU": vals(lambda s: s["cpu"]["model"]),
        "CPU cores per job": vals(lambda s: s["cpu"]["cores_allocated"]),
        "Node RAM (GB)": vals(lambda s: s["cpu"]["ram_node_gb"]),
        "Job memory (MB)": vals(lambda s: s["slurm"].get("SLURM_MEM_PER_NODE")),
        "Cluster / partition": vals(lambda s: f"GWDG Grete, {s['slurm'].get('SLURM_JOB_PARTITION')}"
                                    if s["slurm"].get("SLURM_JOB_PARTITION") else None),
        "Job time limit": vals(lambda s: s["slurm"].get("TimeLimit")),
        "OS": vals(lambda s: s["os"]),
        "Python": vals(lambda s: s["versions"]["python"]),
        "PyTorch / CUDA build / cuDNN": vals(lambda s: f"{s['torch']['torch']} / {s['torch']['cuda_build']} / {s['torch']['cudnn']}"),
        "transformers / sentence-transformers": vals(lambda s: f"{s['versions'].get('transformers')} / "
                                                     f"{s['versions'].get('sentence-transformers')}"),
        "sci-jepa": vals(lambda s: s["versions"].get("sci-jepa")),
        "Jobs": len(systems),
    }


def compute(runs, systems=None):
    """Per method: runs, time per run, epochs, steps, parameters, GPU(-slice) hours and A100-equivalent hours."""
    frac = None
    if systems:
        fr = {_mig_fraction(m["name"]) for s in systems for m in s["gpus"]["mig"]} or \
             {1.0 for s in systems if s["gpus"]["physical"]}
        frac = fr.pop() if len(fr) == 1 else None
    rows = []
    for tag, ms in runs.items():
        secs = [m.get("train_sec") or 0 for m in ms]
        rl = [m.get("run_length") or {} for m in ms]
        fam = ms[0].get("family", "jepa")
        cfg = ms[0].get("config") or {}
        rows.append({"tag": tag, "family": fam, "runs": len(ms),
                     "minutes_mean": float(np.mean(secs)) / 60, "minutes_min": min(secs) / 60, "minutes_max": max(secs) / 60,
                     "epochs_mean": float(np.mean([r.get("epochs_trained") or 0 for r in rl])) if fam != "comparison" else None,
                     "best_epoch_mean": float(np.mean([r.get("best_epoch") or 0 for r in rl])) if fam != "comparison" else None,
                     "steps_mean": float(np.mean([r.get("steps") or 0 for r in rl])) if fam != "comparison" else None,
                     "params": ms[0].get("params_trainable"), "batch_size": cfg.get("batch_size"),
                     "llm_model": cfg.get("llm_model") if cfg.get("baseline") == "llm" else None,
                     "gpu_hours": sum(secs) / 3600, "a100_equiv_hours": sum(secs) / 3600 * frac if frac else None})
    rows.sort(key=lambda r: -r["gpu_hours"])
    total = {"runs": sum(r["runs"] for r in rows), "gpu_hours": sum(r["gpu_hours"] for r in rows),
             "a100_equiv_hours": sum(r["a100_equiv_hours"] for r in rows) if frac else None, "slice_fraction": frac}
    return rows, total


def monitoring(ts):
    """Peaks and means of the per-second samples (None without samples)."""
    if ts is None or not len(ts):
        return None
    out = {"samples": int(len(ts)), "jobs": int(ts["job"].nunique()), "hours_monitored": float(len(ts)) / 3600}
    for col in ts.columns:
        if col.endswith(("_mem_used_gb", "_power_w", "_temp_c", "_util_pct")) or col.startswith("cpu_") \
                or col == "steps_per_s_total":
            x = ts[col].dropna()
            if len(x):
                out[col] = {"mean": float(x.mean()), "max": float(x.max())}
    return out


def summarize(exp_dir, runs=None):
    if runs is None:
        from sci_jepa.reporting.report_pdf import load
        _, runs = load(exp_dir)
    res = _results_dir(exp_dir)
    systems = load_systems(res)
    rows, total = compute(runs, systems)
    return {"experiment": os.path.basename(exp_dir.rstrip("/")), "generated": time.strftime("%Y-%m-%d %H:%M"),
            "hardware": hardware(systems), "compute": rows, "total": total,
            "monitoring": monitoring(load_timeseries(res)), "systems": systems}


def _name(tag):
    arm = tag.split("__", 1)[1] if "__" in tag else tag
    return f"LLM-Reasoning ({arm.split('@', 1)[1]})" if arm.startswith("LLM-Reasoning@") else arm


def _tex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%") \
        .replace("#", r"\#").replace("@", "@")


def write_tables(summary, exp_dir):
    """infrastructure.json (everything), .md (readable), .tex (booktabs tables for the paper)."""
    json.dump(summary, open(os.path.join(exp_dir, "infrastructure.json"), "w"), indent=1, default=str)
    hw, rows, tot = summary["hardware"], summary["compute"], summary["total"]
    eq = tot.get("a100_equiv_hours")
    md = [f"# Training infrastructure: {summary['experiment']}", "", f"Generated {summary['generated']}.", "",
          "## Hardware and software", "", "| Item | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in hw.items()]
    md += ["", "## Compute per method", "",
           f"Total: {tot['runs']} runs, {tot['gpu_hours']:.1f} GPU-hours on one GPU or slice each"
           + (f" ({eq:.1f} A100-equivalent hours; one slice = {tot['slice_fraction']:.2f} A100)" if eq else "") + ".", "",
           "| Method | Family | Runs | Minutes per run (mean, min–max) | Epochs | Best epoch | Steps | Parameters | GPU-hours |",
           "|---|---|---|---|---|---|---|---|---|"]
    f1 = lambda v: "–" if v is None else f"{v:.1f}"
    for r in rows:
        steps = "–" if r["steps_mean"] is None else f"{r['steps_mean']:,.0f}"
        params = "–" if not r["params"] else f"{r['params'] / 1e6:.2f} M"
        md.append(f"| {_name(r['tag'])} | {FAMILY.get(r['family'], r['family'])} | {r['runs']} | "
                  f"{r['minutes_mean']:.1f} ({r['minutes_min']:.1f}–{r['minutes_max']:.1f}) | {f1(r['epochs_mean'])} | "
                  f"{f1(r['best_epoch_mean'])} | {steps} | {params} | {r['gpu_hours']:.2f} |")
    mon = summary.get("monitoring")
    if mon:
        md += ["", "## Measured during the runs (one sample per second)", "",
               f"{mon['samples']:,} samples over {mon['hours_monitored']:.1f} h in {mon['jobs']} job(s).", "",
               "| Quantity | Mean | Peak |", "|---|---|---|"]
        md += [f"| {k} | {v['mean']:.2f} | {v['max']:.2f} |" for k, v in mon.items() if isinstance(v, dict)]
        md += ["", "Power and temperature belong to the whole physical GPU, which other jobs' MIG slices share."]
    open(os.path.join(exp_dir, "infrastructure.md"), "w").write("\n".join(md) + "\n")

    tex = [r"% Training infrastructure (generated by sci_jepa.infra.report); needs \usepackage{booktabs}",
           r"\begin{table}[t]\centering\small", r"\caption{Training infrastructure.}\label{tab:infrastructure}",
           r"\begin{tabular}{@{}ll@{}}", r"\toprule"]
    tex += [f"{_tex_escape(k)} & {_tex_escape(v)} \\\\" for k, v in hw.items() if k not in ("Jobs",)]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
            r"\begin{table}[t]\centering\small",
            r"\caption{Compute per method: mean time per run on one GPU (slice), epochs until early stopping, and total GPU-hours.}",
            r"\label{tab:compute}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
            r"Method & Runs & Min./run & Epochs & GPU-h \\", r"\midrule"]
    for r in rows:
        ep = "--" if r["epochs_mean"] is None else f"{r['epochs_mean']:.1f}"
        tex.append(f"{_tex_escape(_name(r['tag']))} & {r['runs']} & {r['minutes_mean']:.1f} & {ep} & {r['gpu_hours']:.2f} \\\\")
    tex += [r"\midrule", f"Total & {tot['runs']} & & & {tot['gpu_hours']:.1f} \\\\", r"\bottomrule", r"\end{tabular}",
            r"\end{table}"]
    open(os.path.join(exp_dir, "infrastructure.tex"), "w").write("\n".join(tex) + "\n")


# ---- figures (PNG + PDF, 300 dpi, the paper figures' style) --------------------------------------------------
def figures(exp_dir, runs=None, out=None):
    """figures/infra_compute.{png,pdf} and, with monitoring data, infra_gpu_memory / infra_throughput /
    infra_gpu_power / infra_cpu_ram. Returns the names written."""
    from sci_jepa.reporting.figures import INK2, _save, _style
    plt = _style()
    summ = summarize(exp_dir, runs)
    out = out or os.path.join(exp_dir, "figures")
    made = []
    rows = summ["compute"]
    if rows:                                            # compute per method: GPU-hours, minutes per run
        rr = rows[::-1]
        fig, ax = plt.subplots(1, 2, figsize=(9.2, max(3.0, 0.24 * len(rr) + 1.2)), sharey=True)
        y = np.arange(len(rr))
        col = [FAMCOL.get(r["family"], "#888") for r in rr]
        ax[0].barh(y, [r["gpu_hours"] for r in rr], color=col, height=0.75)
        ax[0].set_xlabel("GPU-hours, all seeds (one GPU or slice per run)")
        ax[1].barh(y, [r["minutes_mean"] for r in rr], color=col, height=0.75,
                   xerr=[[r["minutes_mean"] - r["minutes_min"] for r in rr], [r["minutes_max"] - r["minutes_mean"] for r in rr]],
                   error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK2})
        ax[1].set_xlabel("minutes per run (mean, min–max over seeds)")
        ax[0].set_yticks(y)
        ax[0].set_yticklabels([_name(r["tag"]) for r in rr], fontsize=7)
        for a in ax:
            a.grid(axis="y", visible=False)
        tot = summ["total"]
        fig.suptitle(f"Compute per method: {tot['gpu_hours']:.1f} GPU-hours in {tot['runs']} runs"
                     + (f" ({tot['a100_equiv_hours']:.1f} A100-equivalent)" if tot.get("a100_equiv_hours") else ""),
                     x=0.01, ha="left", fontsize=9.5)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=FAMCOL[f], label=FAMILY[f]) for f in FAMILY if any(r["family"] == f for r in rr)],
                   loc="upper right", ncol=3, fontsize=7.5, bbox_to_anchor=(1.0, 1.0))
        made.append(_save(fig, out, "infra_compute"))
    ts = load_timeseries(_results_dir(exp_dir))
    if ts is not None and len(ts) > 1:
        t_h = (ts["t"] - ts["t"].min()) / 3600
        def series(suffix):
            return [c for c in ts.columns if c.endswith(suffix)]
        def timeline(cols, ylab, title, name, labels=None):
            if not cols:
                return
            fig, ax = plt.subplots(figsize=(7.2, 2.8))
            for i, c in enumerate(cols):
                ax.plot(t_h, ts[c], lw=1, label=(labels or {}).get(c, c.split("_")[0]))
            ax.set_xlabel("hours since the first job started (gaps = between jobs)")
            ax.set_ylabel(ylab)
            ax.set_title(title, loc="left")
            if len(cols) > 1:
                ax.legend(fontsize=7, ncol=min(len(cols), 4))
            made.append(_save(fig, out, name))
        timeline(series("_mem_used_gb"), "GB", "GPU memory in use per device (NVML)", "infra_gpu_memory")
        timeline([c for c in ["steps_per_s_total"] if c in ts], "training steps / s",
                 "Training throughput, all runs of the job together", "infra_throughput")
        timeline(series("_power_w"), "W", "Power of the physical GPU (shared by all MIG slices on it)", "infra_gpu_power")
        timeline([c for c in ("cpu_cpu_pct_of_allocated", "cpu_node_mem_used_pct") if c in ts], "%",
                 "CPU use of the job (of its allocated cores) and node memory", "infra_cpu_ram",
                 {"cpu_cpu_pct_of_allocated": "CPU (% of allocated cores)", "cpu_node_mem_used_pct": "node RAM used (%)"})
    return made


def write_all(exp_dir, runs=None):
    """Tables + figures of the infrastructure of one finished (or partly finished) experiment."""
    summ = summarize(exp_dir, runs)
    write_tables(summ, exp_dir)
    made = figures(exp_dir, runs)
    print(f"[infra] {exp_dir}: infrastructure.json / .md / .tex, figures {made}", flush=True)
    return summ


# ---- LLM comparison (several LLM-Reasoning arms) -----------------------------------------------------------
def _llm_rows(rows):
    return [r for r in rows if (r["name"].startswith("LLM-Reasoning"))]


def llm_figure(rows, runs, out):
    """fig13_llms: every LLM next to the best JEPA model and the strongest other comparison system, on the
    reasoning index, counterfactual robustness and consistency, plus evaluation time. Needs >= 2 LLM arms."""
    from sci_jepa.reporting.figures import INK2, _save, _style
    llms = _llm_rows(rows)
    if len(llms) < 2:
        return None
    plt = _style()
    best_j = max((r for r in rows if r["family"] in ("jepa", "gnn") and r.get("index") is not None),
                 key=lambda r: r["index"], default=None)
    best_c = max((r for r in rows if r["family"] == "comparison" and not r["name"].startswith("LLM-Reasoning")
                  and r.get("index") is not None), key=lambda r: r["index"], default=None)
    show = sorted(llms, key=lambda r: -(r.get("index") or 0)) + [r for r in (best_j, best_c) if r]
    labels = [r["name"].replace("LLM-Reasoning (", "").rstrip(")") if r["name"].startswith("LLM-Reasoning")
              else f"{r['name']} (best {'JEPA' if r['family'] != 'comparison' else 'other system'})" for r in show]
    col = ["#7a5cc4" if r["name"].startswith("LLM-Reasoning") else FAMCOL[r["family"]] for r in show]
    mins = {}
    for r in show:
        ms = runs.get(r["tag"]) or []
        mins[r["tag"]] = np.mean([m.get("train_sec") or 0 for m in ms]) / 60 if ms else None
    panels = [("index", "reasoning index (× chance)"), ("cf", "counterfactual robustness (chance 0.5)"),
              ("consistency", "consistency, balanced acc. (chance 0.5)")]
    panels = [(k, l) for k, l in panels if any(r.get(k) is not None for r in show)]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.0 * (len(panels) + 1), 0.32 * len(show) + 1.6), sharey=True)
    y = np.arange(len(show))[::-1]
    for ax, (k, lab) in zip(axes, panels):
        v = [r.get(k) or 0 for r in show]
        ax.barh(y, v, color=col, height=0.72)
        for yi, vi in zip(y, v):
            ax.text(vi, yi, f" {vi:.2f}", va="center", fontsize=6.5, color=INK2)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    ax = axes[-1]
    v = [mins[r["tag"]] or 0 for r in show]
    ax.barh(y, v, color=col, height=0.72)
    ax.set_xlabel("minutes per evaluation / training run")
    ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    fig.suptitle("LLM-Reasoning across model families, against the best JEPA model and the strongest other system",
                 x=0.01, ha="left", fontsize=9.5)
    return _save(fig, out, "fig13_llms")


def llm_pages(pdf, plt, rows, A4, page_title):
    llms = _llm_rows(rows)
    if len(llms) < 2:
        return
    fig = plt.figure(figsize=A4)
    page_title(fig, "LLM-Reasoning across model families",
               "every LLM ranks the same candidates (calibrated log-likelihood), on the same queries as every system")
    cols = ["Model", "Reasoning index", "Forward", "Design", "Abduction", "Consistency", "Robustness"]
    keys = ["index", "forward", "inverse", "abduction", "consistency", "cf"]
    y = 0.86
    xs = [0.04, 0.42, 0.54, 0.64, 0.74, 0.84, 0.93]
    for x, c in zip(xs, cols):
        fig.text(x, y, c, fontsize=8.5, weight="bold")
    for r in sorted(llms, key=lambda r: -(r.get("index") or 0)):
        y -= 0.035
        fig.text(xs[0], y, r["name"].replace("LLM-Reasoning (", "").rstrip(")"), fontsize=8.5)
        for x, k in zip(xs[1:], keys):
            v = r.get(k)
            fig.text(x, y, "–" if v is None else f"{v:.2f}", fontsize=8.5)
    pdf.savefig(fig)
    plt.close(fig)


# ---- PDF pages ---------------------------------------------------------------------------------------------
def infra_pages(pdf, plt, exp_dir, runs, A4, page_title):
    summ = summarize(exp_dir, runs)
    hw, rows, tot = summ["hardware"], summ["compute"], summ["total"]
    fig = plt.figure(figsize=A4)
    page_title(fig, "Training infrastructure", "what the runs ran on (one record per job: sci_jepa.infra.sysinfo)")
    y = 0.87
    for k, v in (hw.items() if hw else [("note", "no job records yet (infra/<job>/system.json)")]):
        fig.text(0.04, y, k, fontsize=8.5, weight="bold")
        fig.text(0.33, y, str(v)[:150], fontsize=8.5)
        y -= 0.032
    eq = tot.get("a100_equiv_hours")
    fig.text(0.04, y - 0.02, f"Compute: {tot['runs']} runs, {tot['gpu_hours']:.1f} GPU-hours (one GPU or slice per run)"
             + (f" = {eq:.1f} A100-equivalent hours" if eq else ""), fontsize=10, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)

    per = 26
    for p0 in range(0, len(rows), per):
        fig = plt.figure(figsize=A4)
        page_title(fig, "Compute per method", "time per run includes the run's own validation and final evaluation")
        hdr = ["Method", "Family", "Runs", "Min./run", "Epochs", "Best ep.", "Steps", "Params", "GPU-h"]
        xs = [0.04, 0.34, 0.50, 0.56, 0.64, 0.71, 0.78, 0.86, 0.93]
        y = 0.87
        for x, h in zip(xs, hdr):
            fig.text(x, y, h, fontsize=8.5, weight="bold")
        for r in rows[p0:p0 + per]:
            y -= 0.029
            vals = [_name(r["tag"]), FAMILY.get(r["family"], r["family"]), r["runs"], f"{r['minutes_mean']:.1f}",
                    "–" if r["epochs_mean"] is None else f"{r['epochs_mean']:.1f}",
                    "–" if r["best_epoch_mean"] is None else f"{r['best_epoch_mean']:.1f}",
                    "–" if r["steps_mean"] is None else f"{r['steps_mean']:,.0f}",
                    "–" if not r["params"] else f"{r['params'] / 1e6:.1f} M", f"{r['gpu_hours']:.2f}"]
            for x, v in zip(xs, vals):
                fig.text(x, y, str(v)[:44], fontsize=7.8, color=FAMCOL.get(r["family"], "#000") if x == xs[0] else "#000")
        pdf.savefig(fig)
        plt.close(fig)

    ts = load_timeseries(_results_dir(exp_dir))
    if ts is not None and len(ts) > 1:
        fig, axes = plt.subplots(2, 2, figsize=A4)
        page_title(fig, "Measured during the runs", "one sample per second; gaps are the time between chained jobs")
        t_h = (ts["t"] - ts["t"].min()) / 3600
        panels = [([c for c in ts.columns if c.endswith("_mem_used_gb")], "GPU memory in use (GB)"),
                  ([c for c in ["steps_per_s_total"] if c in ts], "training steps / s (all runs of a job)"),
                  ([c for c in ts.columns if c.endswith("_power_w")], "physical-GPU power (W, shared by MIG slices)"),
                  ([c for c in ("cpu_cpu_pct_of_allocated", "cpu_node_mem_used_pct") if c in ts], "CPU / node RAM (%)")]
        for ax, (cols, lab) in zip(axes.flat, panels):
            for c in cols:
                ax.plot(t_h, ts[c], lw=0.8, label=c)
            ax.set_title(lab, fontsize=9, loc="left")
            ax.set_xlabel("hours")
            if 1 < len(cols) <= 6:
                ax.legend(fontsize=6)
        fig.subplots_adjust(top=0.86, hspace=0.45)
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    write_all(sys.argv[1])
