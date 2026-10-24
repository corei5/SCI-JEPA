"""Seed lists used by the split and k-fold drivers.

The lists are the ones the legacy loops already ran, so Lightning jobs sample
the same initializations.
"""

from __future__ import annotations

SPLIT_SEEDS = [21, 42, 41, 95, 12, 35, 66, 85, 3, 1234]
FOLD_SEEDS = [42, 21, 95, 12, 35]


def seeds_for(cfg, mode: str) -> list[int]:
    default = FOLD_SEEDS if mode == "kfold" else SPLIT_SEEDS
    if getattr(cfg, "seed", None) is not None:
        cfg.train.runs = 1
        return [int(cfg.seed)]
    runs = int(cfg.train.runs)
    if runs > len(default):
        names = "k-fold" if mode == "kfold" else "split"
        raise ValueError(
            f"train.runs={runs} exceeds the {len(default)} built-in {names} seeds. "
            "Set cfg.seed for a single extra run, or extend core.seeds."
        )
    if runs < 1:
        raise ValueError("train.runs must be at least 1.")
    return list(default[:runs])
