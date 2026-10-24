"""Infrastructure that does not need a GPU or a dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

from yacs.config import CfgNode as CN

from core.config import resolve_config, set_cfg
from core.experiment import Experiment
from core.paths import data_root, papers_layout
from core.seeds import seeds_for


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_defaults_select_lightning():
    cfg = set_cfg(CN())
    assert cfg.train.backend == "lightning"
    assert cfg.train.objective == "auto"
    assert cfg.train.ema_scope == "full"
    assert cfg.jepa.ema_momentum_end == 0.999
    assert cfg.logging.root == "runs"


def test_resolve_config_finds_papers_yaml():
    path = Path(resolve_config("papers.yaml"))
    assert path.name == "papers.yaml"
    assert path.is_file()


def test_data_root_is_inside_the_repo(monkeypatch):
    monkeypatch.delenv("SCIJEPA_DATA_ROOT", raising=False)
    root = Path(data_root(None, "papers"))
    assert root.name == "papers"
    assert root.parent.name == "dataset"
    layout = papers_layout()
    assert layout["raw"].endswith(f"papers{os.sep}raw")
    assert layout["hetero"].endswith("hetero_graphA.pt")


def test_data_root_honors_explicit_override():
    cfg = _Cfg(data=_Cfg(root="/tmp/sci-jepa-papers"))
    assert data_root(cfg, "papers") == "/tmp/sci-jepa-papers"


def test_seeds_match_the_historical_lists():
    cfg = _Cfg(seed=None, train=_Cfg(runs=2))
    assert seeds_for(cfg, "split") == [21, 42]
    cfg = _Cfg(seed=7, train=_Cfg(runs=4))
    assert seeds_for(cfg, "kfold") == [7]
    assert cfg.train.runs == 1


def test_experiment_writes_existing_dashboard(tmp_path):
    dash = tmp_path / "dashboard"
    experiment = Experiment(
        tmp_path / "runs" / "toy-group" / "seed-3",
        {"group": "toy-group", "run_id": "seed-3", "seed": 3, "dataset": "toy"},
        dashboard=dash,
    )
    experiment.event("epoch", epoch=0, metrics={"train_loss": 1.25, "val_loss": 1.1})
    experiment.write_summary({"seed": 3, "train_loss": 1.25, "mae": 0.4})

    catalog = json.loads((dash / "runs.json").read_text(encoding="utf-8"))
    assert catalog[0]["name"] == "toy__toy-group__seed3"
    assert catalog[0]["path"] == "runs/toy__toy-group__seed3/live.json"
    live = json.loads((dash / "runs" / "toy__toy-group__seed3" / "live.json").read_text())
    assert live["status"] == "done"
    assert live["history"][0]["jepa"] == 1.25
    assert live["history"][0]["val_loss"] == 1.1
    assert live["epoch_results"] == []
    assert live["final"]["mae"] == 0.4
    assert (dash / "index.html").is_file()


def test_objective_math():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from core.objectives import interpolate_momentum, regression_loss, resolve_plan

    assert interpolate_momentum(0, 10, 0.99, 0.999) == 0.99
    assert abs(interpolate_momentum(10, 10, 0.99, 0.999) - 0.999) < 1e-9
    zeros = torch.zeros(2, 3, 4)
    assert float(regression_loss(zeros, zeros, 0, 0.5)) == 0.0

    cfg = set_cfg(CN())
    cfg.dataset = "papers"

    def train():
        return None

    train.__module__ = "train.papers"
    assert resolve_plan(cfg, train) == ("vicreg", "full")
    cfg.dataset = "ZINC"
    train.__module__ = "train.zinc"
    assert resolve_plan(cfg, train)[0] == "distance"


def test_update_cfg_accepts_flags_pairs_and_equals():
    from core.config import update_cfg

    cfg = update_cfg(
        set_cfg(CN()),
        "--epochs 3 --lr 1e-4 train.batch_size 64 metis.n_patches=16 "
        "--loggers tensorboard,csv trainer.limit_train_batches 10 --fast-dev-run",
    )
    assert cfg.train.epochs == 3
    assert cfg.train.lr == 1e-4
    assert cfg.train.batch_size == 64
    assert cfg.metis.n_patches == 16
    assert cfg.logging.loggers == ["tensorboard", "csv"]
    assert cfg.trainer.limit_train_batches == 10
    assert cfg.trainer.fast_dev_run == 1


def test_update_cfg_keeps_the_slurm_syntax():
    from core.config import update_cfg

    cfg = update_cfg(set_cfg(CN()), "--config papers.yaml device 0")
    assert cfg.dataset == "papers"
    assert cfg.device == 0


def test_update_cfg_suggests_a_key_for_typos():
    import pytest

    from core.config import update_cfg

    with pytest.raises(SystemExit, match="train.batch_size"):
        update_cfg(set_cfg(CN()), "batch_size 4")


def test_every_trainer_key_has_help():
    from core.config import flatten_cfg, option_docs

    docs = option_docs()
    keys = [key for key in flatten_cfg(set_cfg(CN())) if key.startswith(("trainer.", "logging."))]
    assert keys
    assert all(docs.get(key) for key in keys), [key for key in keys if not docs.get(key)]


def test_trainer_flags_skip_unset_keys():
    from core.engine import _trainer_flags

    cfg = set_cfg(CN())
    cfg.trainer.max_time = "00:01:00:00"
    flags = _trainer_flags(cfg)
    assert flags == {"max_time": "00:01:00:00", "log_every_n_steps": 50}


def test_loggers_write_one_version_per_fold(tmp_path):
    from core.engine import _checkpoint_dir, _loggers

    cfg = set_cfg(CN())
    cfg.logging.loggers = ["tensorboard", "csv"]
    experiment = Experiment(
        tmp_path / "runs" / "g" / "seed-1",
        {"group": "g", "run_id": "seed-1", "seed": 1, "dataset": "toy"},
        dashboard=tmp_path / "dashboard",
    )
    experiment.fold = 2
    names = [(type(logger).__name__, logger.version) for logger in _loggers(cfg, experiment)]
    assert names == [("TensorBoardLogger", "fold2"), ("CSVLogger", "fold2")]
    assert _checkpoint_dir(experiment).name == "fold2"
    cfg.logging.tensorboard = False
    assert [type(logger).__name__ for logger in _loggers(cfg, experiment)] == ["CSVLogger"]
