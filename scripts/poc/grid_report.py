#!/usr/bin/env python
"""Merge a grid run's per-worker TSVs into one table, best R@1 first.

    python scripts/poc/grid_report.py scripts/poc/grid/<job-id>

Reads every results_w*.tsv written by scripts/poc/grid.sbatch. Safe to run while
the sweep is still going: it reports whatever has finished so far.
"""

import csv
import sys
from pathlib import Path

COLUMNS = ("embed", "batch", "hidden", "dropout", "lr", "best_epoch", "epochs", "R@1", "R@10", "MRR", "val_loss", "base_R@1")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for tsv in sorted(out.glob("results_w*.tsv")):
        with open(tsv) as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit(f"no results_w*.tsv under {out}")

    rows.sort(key=lambda r: float(r["R@1"]), reverse=True)

    failures = out / "failures.txt"
    n_failed = len(failures.read_text().split()) if failures.exists() else 0
    print(f"{len(rows)} runs, best R@1 first" + (f"  ({n_failed} failed, see failures.txt)" if n_failed else ""))
    print()
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        # .get, not [] : TSVs written before a column existed are still readable.
        print("| " + " | ".join(r.get(c, "–") for c in COLUMNS) + " |")

    # The baseline is a property of the embedding model, not of any config, so a run
    # only counts as a win if it beats the baseline for its own embedder.
    print()
    for embed in sorted({r["embed"] for r in rows}):
        same = [r for r in rows if r["embed"] == embed]
        beat = [r for r in same if float(r["R@1"]) > float(r["base_R@1"])]
        print(f"{embed}: {len(beat)}/{len(same)} configs beat the evidence-only baseline ({same[0]['base_R@1']})")


if __name__ == "__main__":
    main()
