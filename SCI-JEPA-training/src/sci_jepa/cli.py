"""
`sci-jepa <command> [options]`: one entry point for every tool in the package.

Each command runs the module's own command line unchanged (`sci-jepa train --exp E10` is
`python -m sci_jepa.training.train --exp E10`); `sci-jepa <command> --help` shows its options.
"""
import os
import runpy
import sys

COMMANDS = {
    # command        module                              what it does
    "run":          ("sci_jepa.pipeline",                "full pipeline: KU graph, dashboard, reference run, experiments, report"),
    "evaluate":     ("sci_jepa.evaluate",                "evaluate a saved model, a comparison system, or a finished experiment"),
    "train":        ("sci_jepa.training.train",          "train one configuration or one experiment (arms x seeds)"),
    "bench":        ("sci_jepa.training.bench_all",      "run BENCH or ALL end to end and write the PDF report"),
    "split":        ("sci_jepa.data.split",              "make the stored, domain-balanced train / val / test split (once)"),
    "build-index":  ("sci_jepa.data.ku_graph",           "build and cache the KU graph for a data setting"),
    "smoke-test":   ("sci_jepa.smoke_test",              "synthetic-data check of every component"),
    "validity":     ("sci_jepa.evaluation.validity",     "counterfactual audit, faithfulness, annotation export / agreement, contamination"),
    "findings":     ("sci_jepa.evaluation.findings",     "pre-registered hypotheses H1-H5 for a finished experiment"),
    "safety":       ("sci_jepa.evaluation.safety",       "safety and reliability checks: robustness, shortcuts, domains, calibration, paraphrases"),
    "compliance":   ("sci_jepa.evaluation.compliance",    "regulation and ethics assessment: EU AI Act, HLEG / ALTAI, GDPR, copyright / TDM and other AI frameworks"),
    "release":      ("sci_jepa.release",                  "release package: export a model with model card, NOTICE, licence and redacted KU texts; check a release folder"),
    "paired-eval":  ("sci_jepa.evaluation.paired_eval",   "paired benchmark v2: every system re-scored on one shared question set (evaluation audit fixes)"),
    "science-safety": ("sci_jepa.evaluation.science_safety", "science-specific safety: KU faithfulness, fabrication, novelty, popularity, overclaiming, dual-use, retractions"),
    "figures":      ("sci_jepa.reporting.figures",       "paper figures and leaderboard tables for a finished experiment"),
    "report":       ("sci_jepa.reporting.report_pdf",    "PDF report for a finished experiment"),
    "serve":        ("sci_jepa.dashboard.serve",         "serve a results folder's dashboard: sci-jepa serve <dashboard dir> [port]"),
}


def fetch_assets():
    """Download the dashboard's two JS libraries into the package once (for machines without internet)."""
    from sci_jepa.dashboard.serve import VENDOR, ensure_vendor
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    ensure_vendor(root)
    print("dashboard assets in", os.path.join(root, "vendor"), "->", sorted(VENDOR))


def usage():
    width = max(map(len, COMMANDS)) + 2
    lines = [__doc__.strip(), "", "commands:"]
    lines += [f"  {c:<{width}}{d}" for c, (_, d) in COMMANDS.items()]
    lines.append(f"  {'fetch-assets':<{width}}download the dashboard's JS libraries into the package (once, online)")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage()); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "fetch-assets":
        fetch_assets(); return 0
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n\n{usage()}", file=sys.stderr); return 2
    module = COMMANDS[cmd][0]
    sys.argv = [f"sci-jepa {cmd}"] + rest
    runpy.run_module(module, run_name="__main__", alter_sys=False)   # keeps argv[0] = "sci-jepa <command>" for --help
    return 0


if __name__ == "__main__":
    sys.exit(main())
