"""
SCI-JEPA: joint-embedding predictive learning over scientific knowledge units (Paper A).

A paper's structured summary is split into typed knowledge units (KUs: context, question, method,
procedure, result, claim, evidence, ...) linked by typed edges. A JEPA model reads a paper plus related
papers with one KU hidden and predicts the hidden KU's representation; it is evaluated by ranking the
true KU against on-topic distractors and counterfactual edits, on five reasoning tasks.

Subpackages, in the order data flows through them:

    data/        parquet -> KU graph -> index -> training / evaluation instances
    models/      the JEPA model (encoders, predictor, heads), losses, SHACL shape bank
    training/    training loop, experiments (arms x seeds), benchmark runner
    evaluation/  retrieval metrics, reasoning tasks, comparison systems, validity checks, findings
    reporting/   paper figures, PDF report, animated graph pages
    dashboard/   live training dashboard (static page + local server + run telemetry)

Top-level modules: config (every setting and experiment preset), inference (use a trained model),
pipeline (the full notebook pipeline as a script), evaluate (standalone evaluation), smoke_test, cli.
"""
__version__ = "0.1.0"
