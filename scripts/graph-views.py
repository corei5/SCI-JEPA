#!/usr/bin/env python3
"""Build the human-readable Graphify views for this repository.

graphify names communities ("Lightning Training Engine", ...) but its call-flow
page groups them under generic headings written for graphify's own codebase.
This script assigns every community to a project area by the source files its
nodes come from, writes graphify-out/sections.json, and regenerates:

    graphify-out/ARCHITECTURE.html   one Mermaid diagram per project area
    graphify-out/GRAPH_TREE.html     collapsible tree (area -> community -> symbol)

Grouping is by path, not by community id, so it survives re-clustering.

    python3 scripts/graph-views.py            # views only
    bash scripts/graph-refresh.sh             # rebuild graph, label, then views
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "graphify-out"

# (area id, area title, path prefixes). First match wins, so order matters.
AREAS = [
    ("gj-engine", "Graph-JEPA · training engine", [
        "Analysis/core/engine.py", "Analysis/core/trainer.py", "Analysis/core/lit.py",
        "Analysis/core/objectives.py", "Analysis/core/asam.py", "Analysis/core/callbacks.py",
        "Analysis/core/experiment.py", "Analysis/core/cli.py", "Analysis/core/config.py",
        "Analysis/core/log.py", "Analysis/core/paths.py", "Analysis/core/seeds.py",
        "Analysis/core/probe.py", "Analysis/core/get_data.py", "Analysis/core/get_model.py",
        "Analysis/core/__init__.py", "Analysis/core/dashboard.py"]),
    ("gj-models", "Graph-JEPA · model and layers", [
        "Analysis/core/model.py", "Analysis/core/model_utils"]),
    ("gj-data", "Graph-JEPA · datasets, paper graph, positional encodings", [
        "Analysis/core/data_utils", "Analysis/core/transform", "Analysis/train/build_rwse.py"]),
    ("gj-reason", "Graph-JEPA · reasoning experiments (paper_reason*)", [
        "Analysis/train/paper_reason"]),
    ("gj-bench", "Graph-JEPA · benchmark entry points (train/*.py)", [
        "Analysis/train"]),
    ("gj-diag", "Graph-JEPA · rebuttal diagnostics (diag/)", [
        "Analysis/diag"]),
    ("pa-data", "Paper A · data and knowledge-unit graph", [
        "SCI-JEPA-training/src/sci_jepa/data", "SCI-JEPA-training/scripts/data"]),
    ("pa-model", "Paper A · model, losses, training", [
        "SCI-JEPA-training/src/sci_jepa/models", "SCI-JEPA-training/src/sci_jepa/training",
        "SCI-JEPA-training/src/sci_jepa/pipeline.py", "SCI-JEPA-training/src/sci_jepa/config.py",
        "SCI-JEPA-training/src/sci_jepa/inference.py"]),
    ("pa-eval", "Paper A · evaluation and baselines", [
        "SCI-JEPA-training/src/sci_jepa/evaluation", "SCI-JEPA-training/src/sci_jepa/evaluate.py"]),
    ("pa-report", "Paper A · reporting, dashboard, infra monitoring", [
        "SCI-JEPA-training/src/sci_jepa/reporting", "SCI-JEPA-training/src/sci_jepa/dashboard",
        "SCI-JEPA-training/src/sci_jepa/infra", "SCI-JEPA-training/src/sci_jepa"]),
    ("tests", "Tests", [
        "tests/", "SCI-JEPA-training/tests"]),
    ("ops", "Cluster, SLURM and run scripts", [
        "Analysis/JEPA-rea", "Analysis/run_", "SCI-JEPA-training/cluster",
        "SCI-JEPA-training/scaling", "SCI-JEPA-training/scripts", "SCI-JEPA-training/run_60k.sh"]),
    ("docs", "Documentation", [
        "README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md", "docs/",
        "Analysis/README.md", "SCI-JEPA-training/README.md", "SCI-JEPA-training/MIGRATION.md",
        "SCI-JEPA-training/docs"]),
    ("tooling", "Repository tooling (uv, graphify, editors)", [
        "pyproject.toml", "SCI-JEPA-training/pyproject.toml", "scripts/", ".cursor", ".mcp.json"]),
]


def area_of(path: str) -> str | None:
    for aid, _title, prefixes in AREAS:
        if any(path == p or path.startswith(p) for p in prefixes):
            return aid
    return None


def build_sections() -> list[dict]:
    graph = json.loads((OUT / "graph.json").read_text())
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for node in graph["nodes"]:
        aid = area_of(node.get("source_file") or "")
        if aid is not None and node.get("community") is not None:
            votes[str(node["community"])][aid] += 1

    members: dict[str, list[str]] = collections.defaultdict(list)
    for cid, counter in votes.items():
        members[counter.most_common(1)[0][0]].append(cid)

    return [
        {"id": aid, "name": title, "communities": sorted(members[aid], key=int)}
        for aid, title, _ in AREAS
        if members[aid]
    ]


def main() -> int:
    if not (OUT / "graph.json").exists():
        print("graphify-out/graph.json is missing; run: bash scripts/graph-refresh.sh", file=sys.stderr)
        return 1
    sections = build_sections()
    sections_path = OUT / "sections.json"
    sections_path.write_text(json.dumps(sections, indent=2) + "\n")
    for sec in sections:
        print(f"  {sec['name']}: {len(sec['communities'])} communities")

    subprocess.run(
        ["graphify", "export", "callflow-html", "--graph", str(OUT / "graph.json"),
         "--sections", str(sections_path), "--output", str(OUT / "ARCHITECTURE.html")],
        check=True, cwd=ROOT,
    )
    subprocess.run(
        ["graphify", "tree", "--graph", str(OUT / "graph.json"), "--label", "SCI-JEPA",
         "--output", str(OUT / "GRAPH_TREE.html")],
        check=True, cwd=ROOT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
