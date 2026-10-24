"""
Run every distinct method (experiment ALL, 66 methods) or the benchmark (BENCH), stream every run to
the live dashboard and write a PDF report.

    # synthetic smoke data, tiny model, every method in a few minutes (checks that everything works)
    python -m sci_jepa.training.bench_all --data smoke
    # a small slice of the real arXiv summaries (functional check on real text; not for ranking)
    python -m sci_jepa.training.bench_all --data real --papers 300 --epochs 12 --seeds 0
    # view: python -m sci_jepa.dashboard.serve paper_a_results/dashboard   (Benchmark tab, pick ALL_real)

Results: <out>/<name>/ (one JSON per method x seed, summary.json, <name>_report.pdf); finished
runs are skipped when rerun with the same settings.
"""
import argparse
import os

from sci_jepa.config import FAST, SIZES, PaperAConfig
from sci_jepa.data.graph_index import get_index
from sci_jepa.reporting.report_pdf import build
from sci_jepa.dashboard.serve import setup_dashboard
from sci_jepa.training.train import run_experiment


def main():
    ap = argparse.ArgumentParser("bench_all")
    ap.add_argument("--data", choices=["smoke", "real"], default="smoke")
    ap.add_argument("--exp", default="ALL", help="ALL (every method) or BENCH (the 14-system benchmark)")
    ap.add_argument("--papers", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=12.0, help="maximum epochs (early stopping on validation)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--size", default="local", choices=list(SIZES))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=os.environ.get("PAPER_A_OUT", "paper_a_results"))
    ap.add_argument("--llm-model", default=None, help="LLM-Reasoning model (default: config.llm_model)")
    ap.add_argument("--llm-queries", type=int, default=None, help="queries per task for the LLM")
    ap.add_argument("--llm-on-mac", action="store_true", help="run LLM-Reasoning without a CUDA GPU")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    name = f"{a.exp}_{a.data}"
    if a.data == "smoke":
        from sci_jepa.smoke_test import FAST as SMOKE
        cfg = PaperAConfig().override(**SIZES["tiny"], **{**SMOKE, "seeds": seeds, "eval_tasks": True,
                                                          "eval_worldview": True, "out_dir": a.out})
    else:
        cfg = PaperAConfig().override(**SIZES[a.size], **FAST, max_papers=a.papers, rows_per_shard=100,
                                      epochs=a.epochs, patience=a.patience, seeds=seeds, out_dir=a.out,
                                      batch_size=64 if a.size in ("tiny", "local") else 128,
                                      neuron_report=False, epoch_worldview=False, save_model=False,
                                      **{k: v for k, v in (("llm_model", a.llm_model), ("llm_eval_instances", a.llm_queries),
                                                           ("llm_allow_cpu", a.llm_on_mac or None)) if v is not None})
    dash = setup_dashboard(os.path.join(a.out, "dashboard"))
    gi = get_index(cfg)
    print(f"[bench_all] {name}: {gi.n_papers} papers, {gi.n_comps} KUs; dashboard {dash}", flush=True)
    run_experiment(cfg, gi, a.exp, log=lambda *x: print(*x, flush=True), live_root=dash, name=name)
    pdf = build(os.path.join(a.out, name))
    print(f"[bench_all] PDF report: {pdf}", flush=True)
    print(f"[bench_all] dashboard: python -m sci_jepa.dashboard.serve {dash}  (Benchmark tab -> {name})", flush=True)


if __name__ == "__main__":
    main()
