"""
Paper figures of one finished experiment (BENCH, ALL, ...): every chart of the dashboard's Benchmark tab and
of every run, as vector PDF (for LaTeX) and 300-dpi PNG, plus the leaderboard as CSV and as a LaTeX table.

    python -m sci_jepa.reporting.figures paper_a_results/BENCH       # -> paper_a_results/BENCH/figures/

Written automatically when an experiment finishes (run_experiment). Colours: the three families take the
first three slots of the validated categorical palette (blue = JEPA with the graph transformer, orange =
JEPA + GNN, aqua = comparison systems; all-pairs colour-blind safe), heatmaps the sequential blue ramp, and
every bar carries a visible label (aqua is below 3:1 contrast on white).

Figures (figures/):
  fig01_reasoning_index      reasoning index per system with 95% bootstrap intervals, chance line
  fig02_tasks                the five tasks as small multiples (one panel per task)
  fig03_counterfactuals      share of counterfactual edits ranked below the truth, per task and per kind of edit
  fig04_methods_x_tasks      heatmap: every system x every task (value printed, colour = rank in the column)
  fig05_learning_reasoning   validation reasoning index per epoch, one panel per family
  fig06_learning_retrieval   validation retrieval (x chance) per epoch, one panel per family
  fig07_contamination        every task before vs after the encoders' pretraining cutoff (dumbbells)
  fig08_hypotheses           pre-registered hypotheses: difference with 95% interval (forest plot)
  fig09_world_model          intervention response and action direction per system (two panels, one scale each)
  fig10_families             reasoning index of every system by family
  fig11_ku_types             heatmap: every system x masked KU type (test, x chance)
  fig12_domains              per canonical research domain: test queries and reasoning index of every system
  runs/<arm>_seed<s>_*       per run: training losses, per-type validation per epoch, tasks per epoch
  leaderboard.csv / leaderboard.tex
"""
import csv
import glob
import json
import os
import sys

import numpy as np

FAMCOL = {"jepa": "#2a78d6", "gnn": "#eb6834", "comparison": "#1baf7a"}
FAMILY = {"jepa": "JEPA (graph transformer)", "gnn": "JEPA + GNN", "comparison": "comparison system"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential blue 100-700
TYPES = ["context", "question", "method", "procedure", "result", "interpretation", "limitation", "claim",
         "evidence", "implication"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "text.color": INK,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42})              # editable text in the PDF
    return plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return name


def _legend(ax, fams, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAMCOL[f], label=FAMILY[f]) for f in FAMILY if f in fams], loc=loc, fontsize=7.5)


def _rows(exp_dir):
    from sci_jepa.reporting.report_pdf import load
    rows, runs = load(exp_dir)
    return rows, runs


OLD = "† evaluated with the earlier, easier protocol (subfield candidates, no counterfactual distractors): an ablation, not a competitor"


def _old(r):
    return "old evaluation" in (r.get("about") or "")


def _label(r):
    s = r["name"] + (f" ({r['source']})" if r["source"] not in ("BENCH",) and "__" in r["tag"] else "")
    return s + (" †" if _old(r) else "")


def _footnote(fig, rows, y=-0.01):
    if any(_old(r) for r in rows):
        fig.text(0.01, y, OLD, fontsize=6.8, color=INK2, ha="left", va="top")


def _fig_legend(fig, fams, y=-0.035):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=FAMCOL[f], label=FAMILY[f]) for f in FAMILY if f in fams], loc="upper center",
               ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, y))


def _hbar(ax, rows, key, label, chance=None, scale=1.0, fmt="{:.2f}", err=None, show_values=True, fs=None):
    rr = [r for r in rows if r.get(key) is not None][::-1]
    y = np.arange(len(rr))
    v = [r[key] / scale for r in rr]
    ax.barh(y, v, height=0.78, color=[FAMCOL[r["family"]] for r in rr], edgecolor="white", linewidth=1)
    if err:
        lo = [max(0.0, (r[key] - err[r["tag"]][0]) / scale) if err.get(r["tag"]) else 0 for r in rr]
        hi = [max(0.0, (err[r["tag"]][1] - r[key]) / scale) if err.get(r["tag"]) else 0 for r in rr]
        ax.errorbar(v, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    fs = fs or (7 if len(rr) <= 24 else 5.5)
    if show_values and len(rr) <= 30:
        for yi, vi in zip(y, v):
            ax.text(vi, yi, " " + fmt.format(vi), va="center", fontsize=min(fs, 6.5), color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(r) for r in rr], fontsize=fs)
    ax.grid(axis="y", visible=False)
    if chance is not None:
        ax.axvline(chance, color=MUTED, ls="--", lw=1)
        ax.text(chance, len(rr) - 0.4, " chance", color=MUTED, fontsize=6.5, va="bottom")
    ax.set_xlabel(label)
    return rr


def export(exp_dir, out=None):
    plt = _style()
    rows, runs = _rows(exp_dir)
    if not rows:
        return []
    out = out or os.path.join(exp_dir, "figures")
    made, fams = [], {r["family"] for r in rows}
    n = len(rows)
    h = max(3.2, 0.2 * n + 1.0)
    # bootstrap intervals from findings (if per-query data exists)
    ci = {}
    try:
        from sci_jepa.evaluation.findings import bootstrap, load_runs
        rng = np.random.default_rng(0)
        for tag, ms in load_runs(exp_dir).items():
            c0 = ms[0].get("config") or {}
            b = bootstrap(ms, rng, c0.get("primary_average", "micro"), c0.get("min_domain_queries", 20))
            if b is not None:
                x = b["index"][~np.isnan(b["index"])]
                if len(x):
                    ci[tag] = (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    except Exception:                                    # noqa: BLE001 -- figures without intervals
        pass

    # fig01 reasoning index
    fig, ax = plt.subplots(figsize=(6.4, h))
    macro = any(r.get("index") != r.get("index_micro") for r in rows if r.get("index") is not None)
    _hbar(ax, rows, "index", "reasoning index (× chance; 1 = chance)" + (", macro over research domains" if macro else ""),
          chance=1.0, err=ci)
    ax.set_title("Reasoning index per system" + (" (95% bootstrap intervals)" if ci else ""), loc="left")
    _legend(ax, fams)
    _footnote(fig, rows)
    made.append(_save(fig, out, "fig01_reasoning_index"))

    # fig02 tasks: small multiples
    tk = [("forward", "Forward prediction (× chance)", 1, 1.0), ("inverse", "Experiment design (× chance)", 1, 1.0),
          ("abduction", "Abduction (× chance)", 1, 1.0), ("consistency", "Consistency (balanced acc.)", 1, 0.5),
          ("toward", "Intervention (toward new result)", 1, 0.5)]
    fs = 7 if n <= 24 else 5.5
    fig, axs = plt.subplots(1, 5, figsize=(17, h), sharey=False)
    for ax, (k, lab, sc, ch) in zip(axs, tk):
        _hbar(ax, sorted(rows, key=lambda r: -(r.get(k) or -1)), k, lab, chance=ch, fs=fs,
              fmt="{:.0%}" if k in ("consistency", "toward") else "{:.1f}")
        if k in ("consistency", "toward"):
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.set_xlim(0, 1.15)
        else:
            ax.set_xlim(0, max(r.get(k) or 0 for r in rows) * 1.18)
    fig.suptitle("The five reasoning tasks (test split; each panel sorted)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(w_pad=1.2)
    _fig_legend(fig, fams)
    _footnote(fig, rows, y=-0.07)
    made.append(_save(fig, out, "fig02_tasks"))

    # fig03 counterfactuals
    ck = [("fwd_cf", "Forward: beats its counterfactuals"), ("inv_cf", "Experiment design: beats its counterfactuals"),
          ("inv_sub", "Experiment design: beats substitutions"), ("abd_cf", "Abduction: beats its counterfactuals")]
    ck = [c for c in ck if any(r.get(c[0]) is not None for r in rows)]
    if ck:
        fig, axs = plt.subplots(1, len(ck), figsize=(3.4 * len(ck), h), squeeze=False)
        for ax, (k, lab) in zip(axs[0], ck):
            _hbar(ax, sorted(rows, key=lambda r: -(r.get(k) or -1)), k, "share of edits ranked below the truth",
                  chance=0.5, fmt="{:.0%}")
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.set_xlim(0, 1.08)
            ax.set_title(lab, loc="left", fontsize=8.5)
        fig.suptitle("Counterfactual robustness (chance 50%)", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(w_pad=1.2)
        _fig_legend(fig, fams)
        _footnote(fig, rows, y=-0.07)
        made.append(_save(fig, out, "fig03_counterfactuals"))

    # fig04 heatmap systems x tasks
    hk = [("index", 1, "reasoning\nindex"), ("forward", 1, "forward"), ("inverse", 1, "exp.\ndesign"),
          ("abduction", 1, "abduction"), ("consistency", 0.5, "consistency\n(÷0.5)"), ("toward", 0.5, "intervention\n(÷0.5)"),
          ("fwd_cf", 1, "forward\n> edits"), ("inv_sub", 1, "design\n> subst."), ("abd_cf", 1, "abduction\n> edits"),
          ("retrieval", 1, "retrieval")]
    hk = [x for x in hk if any(r.get(x[0]) is not None for r in rows)]
    M = np.array([[(r[k] / d) if r.get(k) is not None else np.nan for k, d, _ in hk] for r in rows])
    R = np.full_like(M, np.nan)
    for j in range(M.shape[1]):                          # colour = rank within the column (0 worst .. 1 best)
        col = M[:, j]; ok = ~np.isnan(col)
        if ok.sum() > 1:                                 # average ranks: equal values get the same colour
            from scipy.stats import rankdata
            rk = rankdata(np.round(col[ok], 6), method="average") - 1
            R[ok, j] = rk / max(ok.sum() - 1, 1) if np.ptp(col[ok]) > 1e-9 else 0.5
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(1.0 * len(hk) + 3.2, max(3.0, 0.24 * n + 1.2)))
    cm = ListedColormap(SEQ); cm.set_bad("#f0efec")
    ax.imshow(np.ma.masked_invalid(R), aspect="auto", cmap=cm, vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                txt = f"{M[i, j]:.0%}" if hk[j][0] in ("fwd_cf", "inv_sub", "abd_cf") else f"{M[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.3,
                        color="white" if (R[i, j] if not np.isnan(R[i, j]) else 0) > 0.55 else INK)
    ax.set_xticks(range(len(hk))); ax.set_xticklabels([l for *_, l in hk], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n)); ax.set_yticklabels([_label(r) for r in rows], fontsize=7 if n <= 24 else 5.5)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(FAMCOL[r["family"]])
    ax.grid(False)
    ax.set_title("Which system works for which task (colour = rank in the column, darker = better)", loc="left",
                 fontsize=9, pad=28)
    fig.subplots_adjust(bottom=0.08)
    _fig_legend(fig, fams, y=0.045)
    _footnote(fig, rows, y=0.012)
    made.append(_save(fig, out, "fig04_methods_x_tasks"))

    # fig05 / fig06 learning curves per family
    from sci_jepa.evaluation.tasks import reasoning_index
    for name, getter, lab in (("fig05_learning_reasoning", lambda e: reasoning_index(e.get("tasks") or {}),
                               "validation reasoning index"),
                              ("fig06_learning_retrieval", lambda e: e["bal_MRR"] / max(e.get("bal_chance") or 1e-9, 1e-9),
                               "validation retrieval (× chance)")):
        trained = [r for r in rows if any(m.get("epoch_results") for m in runs[r["tag"]])]
        if not trained:
            continue
        tf = [f for f in ("jepa", "gnn") if any(r["family"] == f for r in trained)]
        def repel(ax, labels):                           # line-end labels: keep >= 4% of the axis apart
            if not labels:
                return
            lo, hi = ax.get_ylim(); gap = 0.045 * (hi - lo)
            labels.sort(key=lambda t: t[1])
            ys = [labels[0][1]]
            for _, y, _ in labels[1:]:
                ys.append(max(y, ys[-1] + gap))
            for (x, _, txt), y in zip(labels, ys):
                ax.text(x, y, " " + txt, fontsize=6.5, color=INK2, va="center")
        fig, axs = plt.subplots(1, len(tf), figsize=(5.2 * len(tf), 3.4), squeeze=False)
        for ax, f in zip(axs[0], tf):
            fr = [r for r in trained if r["family"] == f]
            fr = sorted(fr, key=lambda r: -(r["index"] or -1))
            ends = []
            for k, r in enumerate(fr):
                curves = [[(e["epoch"], getter(e)) for e in m.get("epoch_results", []) if getter(e) is not None]
                          for m in runs[r["tag"]]]
                ep = sorted({x for c in curves for x, _ in c})
                if not ep:
                    continue
                mean = [np.mean([y for c in curves for x, y in c if x == e]) for e in ep]
                strong = k < 4                            # top 4 per family in the family colour, the rest muted
                ax.plot(ep, mean, color=FAMCOL[f] if strong else AXIS, lw=2 if strong else 1,
                        alpha=1.0 if strong else 0.8, zorder=3 if strong else 2)
                if strong:
                    ends.append((ep[-1], mean[-1], r["name"]))
            repel(ax, ends)
            ax.set_title(FAMILY[f] + " (top 4 labelled)", loc="left")
            ax.set_xlabel("epoch"); ax.set_ylabel(lab)
        fig.tight_layout()
        made.append(_save(fig, out, name))

    # fig07 contamination dumbbells
    from sci_jepa.evaluation.validity import contamination
    cont = contamination(exp_dir)
    tags = [r["tag"] for r in rows if any(k.endswith("until_cutoff") for k in cont.get(r["tag"], {}))]
    if tags:
        fig, axs = plt.subplots(1, 3, figsize=(12, max(3.0, 0.22 * len(tags) + 1.2)), squeeze=False)
        for ax, k in zip(axs[0], ("forward", "inverse", "abduction")):
            ys = np.arange(len(tags))[::-1]
            for yi, t in zip(ys, tags):
                a, b = cont[t].get(f"{k}:until_cutoff"), cont[t].get(f"{k}:after_cutoff")
                if a is None or b is None:
                    continue
                ax.plot([a, b], [yi, yi], color=AXIS, lw=1.5, zorder=1)
                ax.scatter([a], [yi], color="#2a78d6", s=28, zorder=2, edgecolor="white", linewidth=1)
                ax.scatter([b], [yi], color="#eb6834", s=28, zorder=2, edgecolor="white", linewidth=1)
            ax.set_yticks(ys); ax.set_yticklabels([t.split("__")[-1] for t in tags], fontsize=6.8)
            ax.set_title({"forward": "Forward", "inverse": "Experiment design", "abduction": "Abduction"}[k] + " (× chance)",
                         loc="left")
            ax.grid(axis="y", visible=False)
        from matplotlib.lines import Line2D
        axs[0][-1].legend(handles=[Line2D([], [], marker="o", ls="", color="#2a78d6", label="test papers up to the cutoff"),
                                   Line2D([], [], marker="o", ls="", color="#eb6834", label="test papers after the cutoff")],
                          loc="lower right", fontsize=7)
        fig.suptitle("Contamination check: scores before vs after the text encoders' pretraining cutoff", x=0.01,
                     ha="left", fontsize=10)
        fig.tight_layout()
        made.append(_save(fig, out, "fig07_contamination"))

    # fig08 hypotheses forest plot
    try:
        from sci_jepa.evaluation.findings import analyse
        F = analyse(exp_dir)
        H = [x for x in F.get("hypotheses", []) if x.get("difference") is not None]
    except Exception:                                    # noqa: BLE001
        H = []
    if H:                                                # one panel per metric: each keeps its own scale
        groups = [("index", "reasoning index difference"), ("shift", "intervention-shift difference"),
                  ("cf", "counterfactual-share difference")]
        groups = [(g, lab) for g, lab in groups if any(x.get("metric", "index") == g for x in H)]
        fig, axs = plt.subplots(len(groups), 1, figsize=(7.6, 0.42 * len(H) + 0.85 * len(groups)), squeeze=False,
                                gridspec_kw={"height_ratios": [sum(x.get("metric", "index") == g for x in H)
                                                               for g, _ in groups]})
        for ax, (g, lab) in zip(axs[:, 0], groups):
            hs = [x for x in H if x.get("metric", "index") == g]
            ys = np.arange(len(hs))[::-1]
            for yi, x in zip(ys, hs):
                d, c = x["difference"], x.get("ci95")
                if c and c[0] is not None:
                    ax.plot(c, [yi, yi], color=INK, lw=1.4)
                ax.scatter([d], [yi], color=INK, s=30, zorder=3)
                ax.text(1.02, yi, x["outcome"], transform=ax.get_yaxis_transform(), va="center", fontsize=7.2,
                        color=INK2, weight="bold")
            ax.axvline(0, color=MUTED, ls="--", lw=1)
            ax.set_yticks(ys); ax.set_yticklabels([f"{x['id']}: {x['a']} vs {x['b']}" for x in hs], fontsize=7.2)
            ax.set_ylim(-0.6, len(hs) - 0.4)
            ax.set_xlabel(lab + " (A − B)")
            ax.grid(axis="y", visible=False)
        axs[0, 0].set_title("Pre-registered hypotheses: difference with 95% bootstrap interval", loc="left")
        fig.tight_layout(h_pad=1.2)
        made.append(_save(fig, out, "fig08_hypotheses"))

    # fig09 world model: two panels, one scale each
    wm = [r for r in rows if r.get("shift") is not None]
    if wm:
        fig, axs = plt.subplots(1, 2, figsize=(10, max(3.0, 0.2 * len(wm) + 1.0)))
        _hbar(axs[0], sorted(wm, key=lambda r: -(r["shift"] or -1)), "shift",
              "intervention response (relative shift)", chance=None)
        axs[0].set_title("How much a swapped condition moves the prediction", loc="left", fontsize=8.5)
        sw = [r for r in rows if r.get("swap") is not None]
        if sw:
            _hbar(axs[1], sorted(sw, key=lambda r: -(r["swap"] or -1)), "swap", "action direction (cosine)", chance=0.0)
            axs[1].set_title("Does the predicted change point the right way? (world-model actions)", loc="left",
                             fontsize=8.5)
        else:
            axs[1].axis("off")
        fig.tight_layout(w_pad=1.5)
        _fig_legend(fig, fams)
        made.append(_save(fig, out, "fig09_world_model"))

    # fig10 families
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, f in enumerate([f for f in ("comparison", "gnn", "jepa") if f in fams]):
        fr = [r for r in rows if r["family"] == f and r["index"] is not None]
        v = [r["index"] for r in fr]
        x = np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        ax.scatter(x, v, color=FAMCOL[f], s=36, edgecolor="white", linewidth=1, zorder=3)
        if v:
            ax.hlines(np.mean(v), i - 0.3, i + 0.3, color=INK, lw=1.5, zorder=4)
            b = max(fr, key=lambda r: r["index"])
            ax.text(i + 0.18, b["index"], b["name"], fontsize=6.8, color=INK2, va="center")
    lab = [FAMILY[f] for f in ("comparison", "gnn", "jepa") if f in fams]
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=7.5)
    ax.axhline(1, color=MUTED, ls="--", lw=1); ax.set_ylabel("reasoning index (× chance)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Every system by family (bar = family mean; best labelled)", loc="left")
    made.append(_save(fig, out, "fig10_families"))

    # fig11 KU types heatmap
    P = []
    for r in rows:
        ms = runs[r["tag"]]
        line = []
        for t in TYPES:
            v = [m.get("per_type", {}).get(t) for m in ms]
            v = [x["MRR_bal"] / max(x.get("chance_bal") or 1e-9, 1e-9) for x in v if x and x.get("MRR_bal") is not None]
            line.append(np.mean(v) if v else np.nan)
        P.append(line)
    P = np.array(P)
    if np.isfinite(P).any():
        fig, ax = plt.subplots(figsize=(8.6, max(3.0, 0.24 * n + 1.2)))
        cm = ListedColormap(SEQ); cm.set_bad("#f0efec")
        vmax = np.nanpercentile(P, 95) if np.isfinite(P).any() else 1
        ax.imshow(np.ma.masked_invalid(P), aspect="auto", cmap=cm, vmin=1, vmax=max(vmax, 1.5))
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if np.isfinite(P[i, j]):
                    ax.text(j, i, f"{P[i, j]:.1f}", ha="center", va="center", fontsize=6.2,
                            color="white" if P[i, j] > (1 + max(vmax, 1.5)) / 2 else INK)
        ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels([r["name"] for r in rows], fontsize=7 if n <= 24 else 5.5)
        for lab_, r in zip(ax.get_yticklabels(), rows):
            lab_.set_color(FAMCOL[r["family"]])
        ax.grid(False)
        ax.set_title("Masked-KU retrieval per KU type (test, × chance; darker = better)", loc="left")
        made.append(_save(fig, out, "fig11_ku_types"))

    # fig12 canonical research domains: reasoning index per domain, and test queries per domain
    D = {}
    for r in rows:
        for m in runs[r["tag"]]:
            T = m.get("tasks") or {}
            for k, dv, key in (("forward", 1, "ratio"), ("inverse", 1, "ratio"), ("abduction", 1, "ratio")):
                for d, v in ((T.get(k) or {}).get("by_domain") or {}).items():
                    D.setdefault(r["tag"], {}).setdefault(d, {}).setdefault(k, []).append((v[key] / dv, v["n"]))
            for d, v in ((T.get("consistency") or {}).get("by_domain") or {}).items():
                D.setdefault(r["tag"], {}).setdefault(d, {}).setdefault("consistency", []).append((v["bal_acc"] / 0.5, v["n"]))
            iv = ((T.get("intervention") or {}).get("condition") or {}).get("by_domain") or {}
            for d, v in iv.items():
                D.setdefault(r["tag"], {}).setdefault(d, {}).setdefault("toward", []).append((v["toward_donor"] / 0.5, v["n"]))
    if D:
        from sci_jepa.data.ku_schema import DOMAINS
        nq = {}
        for tg in D.values():
            for d, ks in tg.items():
                nq[d] = max(nq.get(d, 0), max((n for x in ks.get("forward", []) for _, n in [x]), default=0))
        doms = [d for d in DOMAINS if d in nq] + [d for d in nq if d not in DOMAINS]
        Mx = np.array([[np.mean([np.mean([v for v, _ in D.get(r["tag"], {}).get(d, {}).get(k, [])])
                                 for k in ("forward", "inverse", "abduction", "consistency", "toward")
                                 if D.get(r["tag"], {}).get(d, {}).get(k)] or [np.nan]) for d in doms] for r in rows])
        fig, (ax0, ax) = plt.subplots(2, 1, figsize=(1.0 * len(doms) + 3.4, max(3.4, 0.24 * n + 2.2)),
                                      gridspec_kw={"height_ratios": [1, max(4, n * 0.3)]}, sharex=True)
        ax0.bar(range(len(doms)), [nq[d] for d in doms], color=AXIS, width=0.7)
        for j, d in enumerate(doms):
            ax0.text(j, nq[d], str(nq[d]), ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax0.set_ylabel("test queries\n(forward)", fontsize=7); ax0.grid(axis="x", visible=False)
        ax0.set_title("Per research domain: test queries (top) and reasoning index (bottom; darker = better)", loc="left",
                      fontsize=9)
        cm = ListedColormap(SEQ); cm.set_bad("#f0efec")
        fin = Mx[np.isfinite(Mx)]
        lo_, hi_ = (np.nanpercentile(fin, 5), np.nanpercentile(fin, 95)) if len(fin) else (0, 1)
        ax.imshow(np.ma.masked_invalid(Mx), aspect="auto", cmap=cm, vmin=lo_, vmax=max(hi_, lo_ + 1e-6))
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, f"{Mx[i, j]:.1f}", ha="center", va="center", fontsize=6.2,
                            color="white" if Mx[i, j] > (lo_ + hi_) / 2 else INK)
        ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels([_label(r) for r in rows], fontsize=7 if n <= 24 else 5.5)
        for lab_, r in zip(ax.get_yticklabels(), rows):
            lab_.set_color(FAMCOL[r["family"]])
        ax.grid(False)
        fig.tight_layout(h_pad=0.4)
        made.append(_save(fig, out, "fig12_domains"))

    # per-run figures
    rdir = os.path.join(out, "runs")
    for tag, ms in runs.items():
        for m in ms:
            base = f"{tag.replace('/', '_')}_seed{m.get('seed')}"
            hist = m.get("history") or []
            if hist:
                fig, ax = plt.subplots(figsize=(5.2, 3.0))
                st = [x["step"] for x in hist]
                for k, col in (("jepa", "#2a78d6"), ("inverse_probe", "#eb6834"), ("inverse", "#eb6834")):
                    if any(k in x for x in hist):
                        ax.plot(st, [x.get(k) for x in hist], color=col, label=k)
                ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("training loss")
                ax.legend(fontsize=7); ax.set_title(f"{tag} (seed {m.get('seed')}): training losses", loc="left", fontsize=8.5)
                made.append(_save(fig, rdir, base + "_losses"))
            E = m.get("epoch_results") or []
            if E:
                fig, ax = plt.subplots(figsize=(6.0, 3.4))
                ep = [e["epoch"] for e in E]
                for t in TYPES:
                    v = [(e.get("per_type") or {}).get(t) for e in E]
                    if any(v):
                        ax.plot(ep, [x["MRR"] / max(x["chance"], 1e-9) if x else np.nan for x in v], lw=1.2,
                                color=AXIS)
                ax.plot(ep, [e["bal_MRR"] / max(e.get("bal_chance") or 1e-9, 1e-9) for e in E], color="#2a78d6",
                        lw=2.2, label="all types")
                rl = m.get("run_length") or {}
                if rl.get("best_epoch"):
                    ax.axvline(rl["best_epoch"], color=MUTED, ls="--", lw=1)
                    ax.text(rl["best_epoch"], ax.get_ylim()[1], " best epoch", fontsize=6.5, color=MUTED, va="top")
                ax.axhline(1, color=MUTED, lw=0.8)
                ax.set_xlabel("epoch"); ax.set_ylabel("validation retrieval (× chance)")
                ax.legend(fontsize=7, loc="upper left")
                ax.set_title(f"{tag} (seed {m.get('seed')}): per KU type (grey) and overall", loc="left", fontsize=8.5)
                made.append(_save(fig, rdir, base + "_per_type"))
                if any(e.get("tasks") for e in E):
                    fig, ax = plt.subplots(figsize=(6.0, 3.4))
                    for k, col in (("forward", "#2a78d6"), ("inverse", "#eb6834"), ("abduction", "#1baf7a")):
                        ax.plot(ep, [((e.get("tasks") or {}).get(k) or {}).get("ratio") for e in E], color=col, label=k)
                    ax.axhline(1, color=MUTED, lw=0.8)
                    ax.set_xlabel("epoch"); ax.set_ylabel("validation × chance")
                    ax.legend(fontsize=7, loc="upper left")
                    ax.set_title(f"{tag} (seed {m.get('seed')}): retrieval tasks per epoch", loc="left", fontsize=8.5)
                    made.append(_save(fig, rdir, base + "_tasks"))

    # leaderboard tables
    cols = [("name", "system"), ("family", "family"), ("index", "reasoning index"), ("index_micro", "index (micro)"),
            ("worst_domain", "worst domain (retrieval)"), ("forward", "forward"),
            ("inverse", "exp. design"), ("abduction", "abduction"), ("consistency", "consistency"),
            ("toward", "intervention"), ("fwd_cf", "fwd > edits"), ("inv_sub", "design > subst."),
            ("retrieval", "retrieval"), ("acc_sim", "acc (sim>=0.8)"), ("acc_top1", "acc (top-1 and sim>=0.8)"),
            ("false_accept", "false accept (best wrong sim>=0.8)"), ("best_epoch", "best epoch"), ("n", "seeds")]
    with open(os.path.join(out, "leaderboard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for _, c in cols] + ["source", "how it works"])
        for r in rows:
            w.writerow([FAMILY[r["family"]] if k == "family" else ("" if r.get(k) is None else
                        (round(r[k], 4) if isinstance(r[k], float) else r[k])) for k, _ in cols] + [r["source"], r["about"]])
    best = {k: max([r[k] for r in rows if r.get(k) is not None and not _old(r)] or [None]) for k, _ in cols[2:-2]}
    def cell(r, k):
        v = r.get(k)
        if v is None:
            return "--"
        s = f"{v:.0%}".replace("%", r"\%") if k in ("consistency", "toward", "fwd_cf", "inv_sub", "acc_sim") else f"{v:.2f}"
        return r"\textbf{" + s + "}" if best.get(k) is not None and not _old(r) and abs(v - best[k]) < 1e-12 else s
    tex = [r"\begin{table}[t]\centering\small", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{l" + "r" * 10 + "}", r"\toprule",
           r"System & Index & Fwd & Design & Abd & Cons & Interv & Fwd$>$cf & Des$>$sub & Retr & Acc$_{\geq 0.8}$ \\", r"\midrule"]
    for fam in ("comparison", "gnn", "jepa"):
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        tex.append(r"\multicolumn{11}{l}{\textit{" + FAMILY[fam] + r"}} \\")
        for r in fr:
            tex.append(r["name"].replace("_", r"\_").replace("&", r"\&") + (r"$^\dagger$" if _old(r) else "") + " & " +
                       " & ".join(cell(r, k) for k in ("index", "forward", "inverse", "abduction", "consistency",
                                                        "toward", "fwd_cf", "inv_sub", "retrieval", "acc_sim")) + r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}", r"\caption{Reasoning benchmark (test split, mean over seeds). Retrieval tasks: MRR $\div$ "
            r"chance; consistency: balanced accuracy; intervention: share moving toward the donor's result; "
            r"$>$cf / $>$sub: share of counterfactual edits ranked below the true answer; Acc$_{\geq 0.8}$: share of JEPA "
            r"predictions with cosine $\geq 0.8$ to the true unit's target representation (mean of the three retrieval "
            r"tasks; not defined for comparison systems). Best per column in bold. "
            r"$^\dagger$Evaluated with the earlier, easier protocol (ablation; not ranked).}",
            r"\label{tab:benchmark}", r"\end{table}"]
    open(os.path.join(out, "leaderboard.tex"), "w").write("\n".join(tex) + "\n")

    # fig13: several LLMs (LLM-Reasoning@<model>); infra_*: training infrastructure (+ infrastructure.json/md/tex)
    from sci_jepa.infra import report as infra_report
    try:
        name = infra_report.llm_figure(rows, runs, out)
        if name:
            made.append(name)
        summ = infra_report.summarize(exp_dir, runs)
        infra_report.write_tables(summ, exp_dir)
        made += infra_report.figures(exp_dir, runs, out)
    except Exception as e:                               # noqa: BLE001  (the paper figures above stay written)
        print(f"[figures] infrastructure / LLM figures skipped: {e}", flush=True)
    return made


if __name__ == "__main__":
    made = export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"{len(made)} figures (PDF + PNG) + leaderboard.csv / .tex in "
          f"{sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], 'figures')}")
