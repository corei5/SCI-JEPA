#!/usr/bin/env python
"""Mean +/- std across seeds for one configuration.

    python scripts/poc/seed_report.py scripts/poc/seeds/<name>/results.tsv

Reports each metric and, more importantly, its margin over the evidence-only
baseline. The seed also picks the paper split, so the baseline is not a constant
across seeds: a raw mean of R@1 alone says nothing about whether the model learned
anything. The margin is the quantity to read.
"""

import csv
import statistics
import sys
from pathlib import Path

METRICS = ("R@1", "R@10", "MRR")


def spread(vals):
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), sd


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.tsv")
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit(f"no rows in {path}")

    cfg = rows[0]
    print(f"{cfg['embed']}  batch {cfg['batch']}  hidden {cfg['hidden']}  dropout {cfg['dropout']}  lr {cfg['lr']}")
    print(f"{len(rows)} seeds: {', '.join(r.get('seed', '?') for r in rows)}")
    print()

    print(f"{'':6s}  {'model':>17s}  {'baseline':>17s}  {'margin':>18s}")
    for m in METRICS:
        mu, sd = spread([float(r[m]) for r in rows])
        bmu, bsd = spread([float(r[f"base_{m}"]) for r in rows])
        dmu, dsd = spread([float(r[m]) - float(r[f"base_{m}"]) for r in rows])
        print(f"{m:6s}  {mu:8.4f} +/- {sd:.4f}  {bmu:8.4f} +/- {bsd:.4f}  {dmu:+9.4f} +/- {dsd:.4f}")

    epochs = [int(r["best_epoch"]) for r in rows]
    print()
    print(f"best epoch: {', '.join(map(str, epochs))}  (mean {statistics.fmean(epochs):.0f})")

    wins = sum(float(r["R@1"]) > float(r["base_R@1"]) for r in rows)
    print(f"seeds beating the baseline on R@1: {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
