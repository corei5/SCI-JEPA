"""Run records for the SCI-JEPA dashboard already in ``SCI-JEPA-training``.

That dashboard is a static page. It lists ``runs.json`` and reads each
``runs/<name>/live.json``. Training here writes those files, so a Graph-JEPA
run shows up next to a Paper A run. Open it with
``bash SCI-JEPA-training/scripts/present_dashboard.sh``.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import graphify_dir, project_root


def json_default(value: Any):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dashboard_root(cfg=None) -> Path:
    """Folder the existing dashboard serves (index.html, runs.json, runs/)."""
    configured = ""
    logging_cfg = getattr(cfg, "logging", None) if cfg is not None else None
    if logging_cfg is not None:
        configured = getattr(logging_cfg, "dashboard_root", "") or ""
    chosen = configured or os.environ.get("SCIJEPA_DASHBOARD_ROOT", "")
    if chosen:
        return Path(chosen)
    return project_root() / "SCI-JEPA-training" / "runs" / "paper_a_results" / "dashboard"


def _package_dashboard() -> Path:
    return project_root() / "SCI-JEPA-training" / "src" / "sci_jepa" / "dashboard"


def ensure_dashboard(root: Path) -> None:
    """Create the folder the present script expects, using the package page."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "graphs").mkdir(exist_ok=True)
    package = _package_dashboard()
    page = package / "dashboard.html"
    infra = package / "infra.html"
    if page.is_file():
        shutil.copyfile(page, root / "index.html")
    if infra.is_file():
        shutil.copyfile(infra, root / "infra.html")
    vendor = package / "vendor"
    if vendor.is_dir():
        dest_vendor = root / "vendor"
        dest_vendor.mkdir(exist_ok=True)
        for item in vendor.iterdir():
            target = dest_vendor / item.name
            if item.is_file() and not target.exists():
                shutil.copyfile(item, target)
    for name in ("runs.json", "graphs.json"):
        path = root / name
        if not path.exists():
            path.write_text("[]\n", encoding="utf-8")
    _register_code_graph(root)


def _atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# Graphify outputs shown in the dashboard's Graphs tab, most readable first.
_CODE_GRAPH_VIEWS = (
    ("ARCHITECTURE.html", "codebase-architecture.html", "Codebase architecture (Graphify)"),
    ("GRAPH_TREE.html", "codebase-tree.html", "Codebase tree (Graphify)"),
    ("graph.html", "codebase.html", "Codebase graph (Graphify)"),
)


def _register_code_graph(root: Path) -> None:
    views = [(graphify_dir() / src, dst, title) for src, dst, title in _CODE_GRAPH_VIEWS]
    views = [view for view in views if view[0].is_file()]
    if not views:
        return
    catalog_path = root / "graphs.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8") or "[]")
    except (OSError, json.JSONDecodeError):
        catalog = []
    for source, dst, title in views:
        dest = root / "graphs" / dst
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        rel = f"graphs/{dst}"
        catalog = [item for item in catalog if item.get("path") != rel]
        catalog.append({"title": title, "path": rel})
    _atomic(catalog_path, catalog)


def _register_run(root: Path, name: str) -> None:
    catalog_path = root / "runs.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8") or "[]")
    except (OSError, json.JSONDecodeError):
        catalog = []
    rel = f"runs/{name}/live.json"
    catalog = [item for item in catalog if item.get("name") != name]
    catalog.insert(0, {"name": name, "started": _now(), "path": rel})
    _atomic(catalog_path, catalog)


class Experiment:
    """Local JSONL plus one ``live.json`` the existing dashboard already polls."""

    def __init__(self, directory: Path, meta: dict, cfg=None, dashboard: Path | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.meta = dict(meta)
        self.fold = None
        self.cfg = cfg
        self.root = Path(dashboard) if dashboard is not None else dashboard_root(cfg)
        seed = self.meta.get("seed")
        dataset = self.meta.get("dataset", "run")
        group = self.meta.get("group", "group")
        self.name = str(self.meta.get("name") or f"{dataset}__{group}__seed{seed}")
        self.live_path = self.root / "runs" / self.name / "live.json"
        self.state = {
            "name": self.name,
            "status": "running",
            "started": _now(),
            "updated": _now(),
            "steps_total": 0,
            "epochs_total": 0,
            "config": {
                "dataset": dataset,
                "seed": seed,
                "group": group,
                "ku_mode": "graph-jepa",
            },
            "schedule": {"schedule": str(dataset), "phases": []},
            "history": [],
            "evals": [],
            "epoch_results": [],
            "final": None,
            "index": {},
            "rel_names": [],
            "comp_types": [],
        }
        ensure_dashboard(self.root)
        _register_run(self.root, self.name)
        self._flush()

    def write_config(self, cfg) -> None:
        text = cfg.dump() if hasattr(cfg, "dump") else str(cfg)
        (self.directory / "config.yaml").write_text(text, encoding="utf-8")
        self.state["config"].update(_public_config(cfg))
        if hasattr(cfg, "items"):
            from core.config import flatten_cfg

            self.state["hparams"] = flatten_cfg(cfg)
        self._flush()

    def event(self, kind: str, **payload) -> dict:
        record = {
            "ts": _now(),
            "kind": kind,
            "group": self.meta.get("group"),
            "run_id": self.meta.get("run_id"),
            "seed": self.meta.get("seed"),
            "dataset": self.meta.get("dataset"),
            "fold": self.fold,
        }
        record.update(payload)
        line = json.dumps(record, default=json_default)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self._apply(kind, payload)
        return record

    def write_summary(self, summary: dict) -> None:
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
        self.event("summary", metrics=summary)

    def _apply(self, kind: str, payload: dict) -> None:
        self.state["updated"] = _now()
        if kind == "start":
            epochs = int(payload.get("epochs") or 0)
            self.state["steps_total"] = epochs
            self.state["epochs_total"] = epochs
            self.state["config"].update({
                "objective": payload.get("objective"),
                "ema_scope": payload.get("ema_scope"),
                "batch_size": payload.get("batch_size"),
            })
        elif kind == "epoch":
            metrics = payload.get("metrics") or {}
            epoch = int(payload.get("epoch", len(self.state["history"])))
            seconds = payload.get("seconds")
            row = {
                "step": epoch,
                "epoch": epoch,
                "jepa": metrics.get("train_loss"),
                "total": metrics.get("train_total_loss"),
                "lr": metrics.get("lr"),
                "ema": metrics.get("momentum"),
                "sec": seconds,
                "val_loss": metrics.get("val_loss"),
                "test_loss": metrics.get("test_loss"),
                "phase": "mixed",
            }
            self.state["history"].append(row)
        elif kind == "summary":
            metrics = payload.get("metrics") or {}
            self.state["status"] = "done"
            self.state["final"] = metrics
        self._flush()

    def _flush(self) -> None:
        _atomic(self.live_path, self.state)


def _public_config(cfg) -> dict:
    train = getattr(cfg, "train", None)
    out = {"dataset": str(getattr(cfg, "dataset", ""))}
    if train is None:
        return out
    for key in ("lr", "epochs", "batch_size", "objective", "ema_scope", "backend"):
        if hasattr(train, key):
            out[key] = getattr(train, key)
    return out
