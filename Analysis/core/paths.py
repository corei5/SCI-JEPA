"""Repository paths and the spawn entrypoint for multi-GPU seed runs.

``spawn_fit`` is imported by a fresh interpreter. It sets ``CUDA_VISIBLE_DEVICES``
before importing PyTorch so each process sees only the GPU it was assigned.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Git or uv project root. Falls back to the directory that contains ``core``."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return here.parents[1]


def analysis_dir() -> Path:
    """Directory that contains ``core``, ``train``, and ``dataset``."""
    return Path(__file__).resolve().parents[1]


def runs_root(cfg=None) -> Path:
    configured = ""
    logging_cfg = getattr(cfg, "logging", None) if cfg is not None else None
    if logging_cfg is not None:
        configured = getattr(logging_cfg, "root", "") or ""
    if not configured:
        configured = os.environ.get("SCIJEPA_RUNS_ROOT", "runs")
    path = Path(configured)
    if not path.is_absolute():
        path = project_root() / path
    return path


def data_root(cfg, folder: str) -> str:
    """Dataset directory for ``folder`` (papers, ZINC, TUD, EXP).

    ``cfg.data.root``, when set, is the full root for this run's dataset.
    ``SCIJEPA_DATA_ROOT`` is the parent directory that contains those folders.
    Otherwise data lives in ``Analysis/dataset/<folder>``.
    """
    configured = ""
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is not None:
        configured = getattr(data_cfg, "root", "") or ""
    if configured:
        return str(configured)
    env = os.environ.get("SCIJEPA_DATA_ROOT", "").strip()
    if env:
        return str(Path(env) / folder)
    return str(analysis_dir() / "dataset" / folder)


def graphify_dir() -> Path:
    for base in (project_root(), analysis_dir()):
        candidate = base / "graphify-out"
        if (candidate / "graph.json").exists() or (candidate / "graph.html").exists():
            return candidate
    return project_root() / "graphify-out"


def run_directory(cfg, group: str, seed: int) -> Path:
    return runs_root(cfg) / group / f"seed-{int(seed)}"


def papers_layout() -> dict[str, str]:
    """Paths for the papers corpus, caches, and reasoning checkpoints.

    Honors ``SCIJEPA_DATA_ROOT`` (the parent of ``papers/``) and
    ``cfg.data.root`` is intentionally not required: these scripts run before
    the YACS config is loaded. ``SCIJEPA_CHECKPOINT_ROOT`` overrides the
    checkpoint directory.
    """
    root = Path(data_root(None, "papers"))
    checkpoints = os.environ.get("SCIJEPA_CHECKPOINT_ROOT", "").strip()
    if not checkpoints:
        checkpoints = str(project_root() / "checkpoints" / "reasoning")
    return {
        "raw": str(root / "raw"),
        "hetero": str(root / "processed" / "hetero_graphA.pt"),
        "rwse": str(root / "processed" / "rwse_pe.pt"),
        "ckpt": checkpoints,
    }


def spawn_fit(cuda_visible: str, cfg_yaml: str, seed: int, mode: str, group: str, run_index: int) -> None:
    """Process entrypoint. Must stay free of PyTorch imports at module scope."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    from core.engine import fit_one_seed

    fit_one_seed(cfg_yaml, int(seed), mode, group, int(run_index), remap_device=True)
