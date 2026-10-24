"""
PDF report of one experiment's results (e.g. BENCH or ALL): leaderboards, charts and the method
catalogue, from the per-run JSON files in <out_dir>/<experiment>/.

    python -m sci_jepa.reporting.report_pdf paper_a_results/ALL            # -> paper_a_results/ALL/ALL_report.pdf
    python -m sci_jepa.reporting.report_pdf paper_a_results/BENCH out.pdf

Pages: overview (data, settings, best method per family), the reasoning leaderboard (every method,
mean over seeds), reasoning index per method, a methods x tasks heatmap, family comparison,
retrieval and model health, and the catalogue: what every method changes.
"""
import glob
import json
import os
import sys
import textwrap
import time

import numpy as np

FAMILY = {"comparison": "comparison system", "gnn": "JEPA + GNN", "jepa": "JEPA (graph transformer)"}
FAMCOL = {"comparison": "#8a8f98", "gnn": "#7c3aed", "jepa": "#2563eb"}
TASKS = [("forward", "Forward"), ("inverse", "Exp. design"), ("abduction", "Abduction"),
         ("consistency", "Consistency"), ("intervention", "Intervention")]


def _g(m, *path):
    for k in path:
        if not isinstance(m, dict) or m.get(k) is None:
            return None
        m = m[k]
    return m


def _metrics(m):
    from sci_jepa.evaluation.tasks import reasoning_index
    T = m.get("tasks") or {}
    rb = m.get("retrieval_balanced") or {}
    rl = m.get("run_length") or {}
    avg = (m.get("config") or {}).get("primary_average") or "micro"
    return {"index": reasoning_index(T, avg), "index_micro": reasoning_index(T, "micro"),
            "worst_domain": min([(T.get(k) or {}).get("worst_domain", {}).get("ratio") for k in ("forward", "inverse", "abduction")
                                 if (T.get(k) or {}).get("worst_domain")] or [None]) if any((T.get(k) or {}).get("worst_domain")
                                 for k in ("forward", "inverse", "abduction")) else None,
            "forward": _g(T, "forward", "ratio"), "fwd_cf": _g(T, "forward", "mp_pair"),
            "inverse": _g(T, "inverse", "ratio"), "inv_cf": _g(T, "inverse", "mp_pair"),
            "inv_sub": _g(T, "inverse", "mp_by_kind", "substitution"),
            "abduction": _g(T, "abduction", "ratio"), "abd_cf": _g(T, "abduction", "mp_pair"),
            "consistency": _g(T, "consistency", "bal_acc"),
            "toward": _g(T, "intervention", "condition", "toward_donor"),
            "shift": _g(T, "intervention", "condition", "rel_shift"),
            "retrieval": rb["MRR"] / rb["chance_MRR"] if rb.get("MRR") is not None and rb.get("chance_MRR") else None,
            # similarity rule (cfg.sim_threshold): share of predictions >= 0.80 cosine to the true KU's target vector,
            # mean of the three retrieval tasks; JEPA models only (comparison systems predict no vector)
            "acc_sim": (lambda v: float(np.mean(v)) if v else None)(
                [x for x in (_g(T, t, "acc_sim") for t in ("forward", "inverse", "abduction")) if x is not None]),
            "acc_top1": (lambda v: float(np.mean(v)) if v else None)(
                [x for x in (_g(T, t, "acc_sim_top1") for t in ("forward", "inverse", "abduction")) if x is not None]),
            "false_accept": (lambda v: float(np.mean(v)) if v else None)(
                [x for x in (_g(T, t, "false_accept") for t in ("forward", "inverse", "abduction")) if x is not None]),
            "structure": (_g(m, "structure", "with") - _g(m, "structure", "without"))
            if _g(m, "structure", "with") is not None else None,
            "swap": _g(m, "actions", "swap", "direction_cos"),
            "best_epoch": rl.get("best_epoch"), "epochs": rl.get("epochs_trained"),
            "minutes": (m.get("train_sec") or 0) / 60,
            "cf": (lambda v: float(np.mean(v)) if v else None)(      # counterfactual robustness, mean of the tasks
                [x for x in (_g(T, t, "mp_pair") for t in ("forward", "inverse", "abduction")) if x is not None])}


def _display_name(arm):
    """LLM-Reasoning@Mistral-7B-Instruct-v0.3 -> LLM-Reasoning (Mistral-7B-Instruct-v0.3)."""
    return f"LLM-Reasoning ({arm.split('@', 1)[1]})" if arm.startswith("LLM-Reasoning@") else arm


def load(exp_dir):
    """-> list of method rows (mean over seeds), best reasoning index first."""
    from sci_jepa.config import EXPERIMENTS, describe, method_family
    lookup = {t: o for arms in EXPERIMENTS.values() for t, o in arms}
    runs = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, "*_seed*.json"))):
        try:
            m = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        runs.setdefault(m.get("tag", os.path.basename(f)), []).append(m)
    rows = []
    for tag, ms in runs.items():
        over = lookup.get(tag, {})
        mets = [_metrics(m) for m in ms]
        row = {"tag": tag, "n": len(ms), "about": ms[0].get("about") or describe(over),
               "family": ms[0].get("family") or method_family(over),
               "source": tag.split("__")[0] if "__" in tag else os.path.basename(exp_dir.rstrip("/")),
               "name": _display_name(tag.split("__", 1)[1] if "__" in tag else tag)}
        for k in mets[0]:
            v = [x[k] for x in mets if x[k] is not None and not (isinstance(x[k], float) and np.isnan(x[k]))]
            row[k] = float(np.mean(v)) if v else None
            row[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        rows.append(row)
    rows.sort(key=lambda r: -(r["index"] if r["index"] is not None else -1))
    return rows, runs


def _f(v, kind="x"):
    if v is None:
        return "–"
    return {"x": f"{v:.2f}", "%": f"{v:.0%}", "n": f"{v:.0f}", "c": f"{v:.2f}", "m": f"{v:.1f}"}[kind]


def build(exp_dir, out_pdf=None, png_dir=None):
    """png_dir: also save every page as a PNG there (previews)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    rows, runs = load(exp_dir)
    if not rows:
        raise SystemExit(f"no results in {exp_dir}")
    name = os.path.basename(exp_dir.rstrip("/"))
    out_pdf = out_pdf or os.path.join(exp_dir, f"{name}_report.pdf")
    any_run = next(iter(runs.values()))[0]
    cfg = any_run.get("config", {})
    summ = {}
    sp = os.path.join(exp_dir, "summary.json")
    if os.path.exists(sp):
        summ = (json.load(open(sp)).get("index") or {})
    A4 = (11.69, 8.27)
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

    def page_title(fig, title, sub=""):
        fig.text(0.04, 0.955, title, fontsize=15, weight="bold")
        if sub:
            fig.text(0.04, 0.925, sub, fontsize=8.5, color="#555")

    with PdfPages(out_pdf) as _pdf:
        class _Save:
            n = 0

            def savefig(self, fig):
                _pdf.savefig(fig)
                if png_dir:
                    os.makedirs(png_dir, exist_ok=True)
                    _Save.n += 1
                    fig.savefig(os.path.join(png_dir, f"page_{_Save.n:02d}.png"), dpi=80)
        pdf = _Save()
        # ---- 1. overview
        fig = plt.figure(figsize=A4)
        page_title(fig, f"SCI-JEPA benchmark: {name}", time.strftime("generated %d %b %Y %H:%M") +
                   f"  ·  {len(rows)} methods  ·  {sum(len(v) for v in runs.values())} runs")
        n_pap = summ.get("papers") or cfg.get("max_papers")
        lines = [f"Data: {n_pap} papers" + (f", splits {summ.get('splits')}" if summ.get("splits") else "") +
                 f", time split T = {cfg.get('time_T')}" + (" (synthetic smoke data)" if cfg.get("synthetic") else ""),
                 f"Model size: d = {cfg.get('d_model')}, {cfg.get('enc_layers')} + {cfg.get('pred_layers')} layers; "
                 f"up to {cfg.get('epochs'):.0f} epochs, early stopping patience {cfg.get('patience')}, "
                 f"best epoch tested: {cfg.get('select_best')}" if cfg.get("epochs") else
                 f"Model size: d = {cfg.get('d_model')}; fixed {cfg.get('steps')} steps",
                 f"Seeds per method: {max(r['n'] for r in rows)}",
                 "Reasoning index = mean of the five task scores on one × chance scale (consistency and intervention ÷ 0.5).",
                 "Counterfactual columns: how often the true answer outranks each edited version of itself (chance 50%)."]
        y = 0.86
        for ln in lines:
            fig.text(0.05, y, ln, fontsize=10); y -= 0.04
        y -= 0.02
        fig.text(0.05, y, "Best method per family (reasoning index)", fontsize=12, weight="bold"); y -= 0.045
        for fam, lab in FAMILY.items():
            r = next((x for x in rows if x["family"] == fam and x["index"] is not None), None)
            if r:
                fig.text(0.07, y, f"{lab}: {r['name']} ({r['source']})  —  {_f(r['index'])}×", fontsize=10,
                         color=FAMCOL[fam]); y -= 0.035
        y -= 0.02
        fig.text(0.05, y, "Best method per task", fontsize=12, weight="bold"); y -= 0.045
        for k, lab in (("forward", "Forward prediction"), ("inverse", "Experiment design"), ("abduction", "Abduction"),
                       ("consistency", "Consistency"), ("shift", "Intervention (shift)"), ("fwd_cf", "Beats forward counterfactuals"),
                       ("inv_sub", "Beats design substitutions"), ("retrieval", "Retrieval")):
            c = [r for r in rows if r[k] is not None]
            if c:
                b = max(c, key=lambda r: r[k])
                fig.text(0.07, y, f"{lab}: {b['name']} ({FAMILY[b['family']]}) — "
                         f"{_f(b[k], '%' if k in ('consistency', 'fwd_cf', 'inv_sub') else 'x')}", fontsize=10); y -= 0.033
        if cfg.get("synthetic") or (n_pap and int(n_pap) < 5000):
            fig.text(0.05, 0.05, "Note: small or synthetic data. Scores saturate and differences of a few tenths are noise; "
                     "this run checks that every method works. Rank methods with 3–5 seeds at 60k+ papers.",
                     fontsize=9, color="#b4322a")
        pdf.savefig(fig); plt.close(fig)

        # ---- 1b. main findings: pre-registered hypotheses and conclusion
        try:
            from sci_jepa.evaluation.findings import analyse
            F = analyse(exp_dir)
        except Exception as e:                            # noqa: BLE001
            F = {"hypotheses": [], "conclusion": f"findings not computed: {e!r}"}
        fig = plt.figure(figsize=A4)
        page_title(fig, "Main findings · pre-registered hypotheses",
                   "decision rule: SUPPORTED if the 95% bootstrap interval of the difference is above 0, CONTRADICTED "
                   "if below, INCONCLUSIVE otherwise (2,000 resamples of the test queries, pooled over seeds)")
        ax = fig.add_axes([0.02, 0.42, 0.96, 0.48]); ax.axis("off")
        cell = []
        for h in F.get("hypotheses", []):
            ci = h.get("ci95") or [None, None]
            cell.append([h["id"], textwrap.shorten(h["claim"], 78), f"{h.get('a', '–')} vs {h.get('b', '–')}",
                         _f(h.get("difference"), "x") if h.get("difference") is not None else "–",
                         f"{ci[0]:+.2f} .. {ci[1]:+.2f}" if ci[0] is not None else "–",
                         _f(h.get("p_seeds"), "c") if h.get("p_seeds") is not None else "–", h.get("outcome", "")])
        c = F.get("contamination", {})
        if c:
            ld = c.get("largest_drop") or {}
            cell.append(["C", textwrap.shorten(c.get("claim", ""), 78), ld.get("system") or "–",
                         _f(ld.get("drop_x_chance"), "x") if ld else "–", "–", "–", c.get("outcome", "")])
        if cell:
            t = ax.table(cellText=cell, colLabels=["", "hypothesis", "compared", "difference", "95% CI", "p (seeds)", "outcome"],
                         loc="upper center", cellLoc="left", colWidths=[0.04, 0.44, 0.2, 0.07, 0.1, 0.06, 0.11])
            t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.6)
            col = {"SUP": "#1d7a4a", "CON": "#b4322a", "INC": "#9a6200", "PAS": "#1d7a4a", "FAI": "#b4322a"}
            for (ri, ci_), cc in t.get_celld().items():
                if ri == 0:
                    cc.set_facecolor("#e7ecf7"); cc.set_text_props(weight="bold")
                elif ci_ == 6:
                    cc.set_text_props(weight="bold", color=col.get(cell[ri - 1][6][:3], "#333"))
        fig.text(0.04, 0.38, "Conclusion", fontsize=12, weight="bold")
        y = 0.345
        for ln in textwrap.wrap(F.get("conclusion", ""), 175):
            fig.text(0.04, y, ln, fontsize=9); y -= 0.028
        pdf.savefig(fig); plt.close(fig)

        # ---- 2. reasoning leaderboard (paged)
        cols = [("#", None), ("method", None), ("family", None), ("index", "x"), ("forward", "x"), ("fwd>cf", "%"),
                ("design", "x"), ("design>cf", "%"), ("design>subst", "%"), ("abduction", "x"), ("abd>cf", "%"),
                ("consist.", "%"), ("toward", "%"), ("shift", "c"), ("retrieval", "x"), ("acc≥0.8", "%"), ("false acc.", "%"), ("best ep", "n")]
        keys = [None, None, None, "index", "forward", "fwd_cf", "inverse", "inv_cf", "inv_sub", "abduction", "abd_cf",
                "consistency", "toward", "shift", "retrieval", "acc_sim", "false_accept", "best_epoch"]
        per = 30
        for p0 in range(0, len(rows), per):
            fig = plt.figure(figsize=A4)
            page_title(fig, "Leaderboard 1 · reasoning benchmark (test split, mean over seeds)",
                       f"methods {p0 + 1}–{min(p0 + per, len(rows))} of {len(rows)}, ranked by reasoning index")
            ax = fig.add_axes([0.02, 0.03, 0.96, 0.87]); ax.axis("off")
            cell = []
            for i, r in enumerate(rows[p0:p0 + per]):
                line = [str(p0 + i + 1), textwrap.shorten(f"{r['name']} ({r['source']})", 46), FAMILY[r["family"]]]
                line += [_f(r[k], kind) for (_, kind), k in zip(cols[3:], keys[3:])]
                cell.append(line)
            t = ax.table(cellText=cell, colLabels=[c for c, _ in cols], loc="upper center", cellLoc="right",
                         colLoc="center", colWidths=[0.025, 0.17, 0.09] + [0.045] * 15)
            t.auto_set_font_size(False); t.set_fontsize(6.6); t.scale(1, 1.25)
            for (ri, ci), c in t.get_celld().items():
                if ri == 0:
                    c.set_facecolor("#e7ecf7"); c.set_text_props(weight="bold")
                elif ci in (1, 2):
                    c.set_text_props(ha="left", color=FAMCOL[rows[p0 + ri - 1]["family"]])
            pdf.savefig(fig); plt.close(fig)

        # ---- 3. reasoning index per method
        fig = plt.figure(figsize=(A4[0], max(A4[1], 0.16 * len(rows) + 1.5)))
        page_title(fig, "Reasoning index per method", "bars coloured by family; dashed line = chance (1×)")
        ax = fig.add_axes([0.32, 0.04, 0.64, 0.86])
        rr = [r for r in rows if r["index"] is not None][::-1]
        ax.barh(range(len(rr)), [r["index"] for r in rr], color=[FAMCOL[r["family"]] for r in rr],
                xerr=[r["index_sd"] or 0 for r in rr], error_kw={"lw": 0.6})
        ax.set_yticks(range(len(rr))); ax.set_yticklabels([f"{r['name']} ({r['source']})" for r in rr], fontsize=6.5)
        ax.axvline(1, ls="--", color="#999", lw=1); ax.set_xlabel("reasoning index (× chance)")
        ax.legend(handles=[Patch(color=FAMCOL[f], label=l) for f, l in FAMILY.items()], loc="lower right")
        pdf.savefig(fig); plt.close(fig)

        # ---- 4. heatmap methods x tasks
        fig = plt.figure(figsize=(A4[0], max(A4[1], 0.16 * len(rows) + 1.5)))
        page_title(fig, "Which method works for which task", "each task on the × chance scale (consistency and "
                   "intervention ÷ 0.5); counterfactual columns: share of edits ranked below the true answer")
        ax = fig.add_axes([0.32, 0.04, 0.6, 0.84])
        hk = [("forward", 1, "forward"), ("inverse", 1, "exp. design"), ("abduction", 1, "abduction"),
              ("consistency", 0.5, "consistency"), ("toward", 0.5, "intervention"), ("fwd_cf", 0.5, "fwd > cf"),
              ("inv_sub", 0.5, "design > subst"), ("abd_cf", 0.5, "abd > cf")]
        M = np.array([[(r[k] / dv) if r[k] is not None else np.nan for k, dv, _ in hk] for r in rows])
        Mn = (M - np.nanmin(M, 0)) / np.maximum(np.nanmax(M, 0) - np.nanmin(M, 0), 1e-9)
        ax.imshow(Mn, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
        ax.set_xticks(range(len(hk))); ax.set_xticklabels([l for *_, l in hk], rotation=30, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r['name']} ({r['source']})" for r in rows], fontsize=6.5)
        for lab, r in zip(ax.get_yticklabels(), rows):
            lab.set_color(FAMCOL[r["family"]])
        fig.text(0.93, 0.5, "colour: rank within each column\n(green = best)", rotation=90, va="center", fontsize=7)
        pdf.savefig(fig); plt.close(fig)

        # ---- 5. families
        fig, axs = plt.subplots(1, 2, figsize=A4)
        page_title(fig, "Families compared", "left: reasoning index of every method; right: best score per task in each family")
        fig.subplots_adjust(top=0.85, bottom=0.12, wspace=0.3)
        for i, (fam, lab) in enumerate(FAMILY.items()):
            v = [r["index"] for r in rows if r["family"] == fam and r["index"] is not None]
            axs[0].scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v,
                           color=FAMCOL[fam], s=18)
            if v:
                axs[0].hlines(np.mean(v), i - 0.3, i + 0.3, color=FAMCOL[fam], lw=2)
        axs[0].set_xticks(range(3)); axs[0].set_xticklabels(list(FAMILY.values()), rotation=12)
        axs[0].set_ylabel("reasoning index (× chance)"); axs[0].axhline(1, ls="--", color="#999", lw=1)
        tk = [("forward", 1), ("inverse", 1), ("abduction", 1), ("consistency", 0.5), ("toward", 0.5), ("shift", 0.1)]
        w = 0.26
        for i, (fam, lab) in enumerate(FAMILY.items()):
            b = [max([r[k] for r in rows if r["family"] == fam and r[k] is not None] or [np.nan]) / dv for k, dv in tk]
            axs[1].bar(np.arange(len(tk)) + (i - 1) * w, b, w, color=FAMCOL[fam], label=lab)
        axs[1].set_xticks(range(len(tk)))
        axs[1].set_xticklabels(["forward", "exp. design", "abduction", "consistency\n÷0.5", "toward\n÷0.5", "shift\n÷0.1"])
        axs[1].legend(fontsize=7); axs[1].set_ylabel("best in family")
        pdf.savefig(fig); plt.close(fig)

        # ---- 6. retrieval and model health
        hc = [("#", None), ("method", None), ("family", None), ("retrieval", "x"), ("from structure", "x"),
              ("action swap", "c"), ("best epoch", "n"), ("epochs", "m"), ("minutes", "m"), ("seeds", "n")]
        hk2 = ["retrieval", "structure", "swap", "best_epoch", "epochs", "minutes", "n"]
        hrows = sorted(rows, key=lambda r: -(r["retrieval"] or -1))
        for p0 in range(0, len(hrows), per):
            fig = plt.figure(figsize=A4)
            page_title(fig, "Leaderboard 2 · retrieval and model health (supporting)",
                       "retrieval = masked-KU MRR ÷ chance; from structure = accuracy lost without typed edges; "
                       "action swap = world-model direction cosine")
            ax = fig.add_axes([0.02, 0.03, 0.96, 0.87]); ax.axis("off")
            cell = [[str(p0 + i + 1), textwrap.shorten(f"{r['name']} ({r['source']})", 56), FAMILY[r["family"]]] +
                    [_f(r[k], kind) for (_, kind), k in zip(hc[3:], hk2)] for i, r in enumerate(hrows[p0:p0 + per])]
            t = ax.table(cellText=cell, colLabels=[c for c, _ in hc], loc="upper center", cellLoc="right",
                         colWidths=[0.03, 0.3, 0.15] + [0.07] * 7)
            t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.25)
            for (ri, ci), c in t.get_celld().items():
                if ri == 0:
                    c.set_facecolor("#e7ecf7"); c.set_text_props(weight="bold")
                elif ci in (1, 2):
                    c.set_text_props(ha="left", color=FAMCOL[hrows[p0 + ri - 1]["family"]])
            pdf.savefig(fig); plt.close(fig)

        # ---- 7. method catalogue
        per_c = 22
        for p0 in range(0, len(rows), per_c):
            fig = plt.figure(figsize=A4)
            page_title(fig, "Method catalogue · how each method works",
                       "what every method changes relative to the reference JEPA model (graph transformer, task-aware "
                       "masking, plain JEPA objective)")
            y = 0.89
            for i, r in enumerate(rows[p0:p0 + per_c]):
                fig.text(0.03, y, f"{p0 + i + 1}. {r['name']}", fontsize=8.5, weight="bold", color=FAMCOL[r["family"]])
                fig.text(0.03, y - 0.017, f"{FAMILY[r['family']]} · from {r['source']} · index {_f(r['index'])}×",
                         fontsize=7, color="#666")
                wrapped = textwrap.wrap(r["about"], 150)[:2]
                for j, ln in enumerate(wrapped):
                    fig.text(0.28, y - 0.004 - j * 0.016, ln, fontsize=7.5)
                y -= 0.038
            pdf.savefig(fig); plt.close(fig)

        # ---- 8. LLM comparison (several LLM-Reasoning arms) and 9. training infrastructure
        from sci_jepa.infra.report import llm_pages, infra_pages
        llm_pages(pdf, plt, rows, A4, page_title)
        infra_pages(pdf, plt, exp_dir, runs, A4, page_title)
    return out_pdf


if __name__ == "__main__":
    print(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, os.environ.get("REPORT_PNG_DIR")))
