"""Lightning entrypoints used by ``core.trainer.run`` and ``run_k_fold``.

``train.backend: legacy`` keeps the original loops. The default ``lightning``
backend trains with DDP when more than one GPU is requested and can place one
seed on each GPU when ``train.parallel_runs`` is true.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DDPStrategy

from core.callbacks import DashboardCallback
from core.config import flatten_cfg
from core.experiment import Experiment
from core.lit import GraphDataModule, GraphJepaLitModule, configure_sharing
from core.objectives import resolve_plan
from core.paths import run_directory, spawn_fit
from core.probe import probe_classification, probe_regression
from core.seeds import seeds_for
from core.trainer import count_parameters, k_fold, save_model, set_seed, summarize_collapse


def run_lightning(cfg, create_dataset, create_model, train, test, evaluator=None, mode="split"):
    del test, evaluator
    try:
        import lightning  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch Lightning is not installed. From the repository root run `uv sync`."
        ) from exc

    cfg = cfg.clone()
    objective, ema_scope = resolve_plan(cfg, train)
    cfg.train.objective = objective
    cfg.train.ema_scope = ema_scope
    entry = getattr(train, "__module__", "") if train is not None else ""
    if entry and entry != "__main__":
        cfg.train.entry = entry

    seeds = seeds_for(cfg, "kfold" if mode == "kfold" else "split")
    group = _group_name(cfg)
    _check_resume(cfg, seeds, mode)
    print(
        f"Lightning trainer | dataset={cfg.dataset} | objective={objective} | "
        f"ema={ema_scope} | seeds={seeds} | group={group}"
    )
    if _use_parallel(cfg, seeds):
        if not str(getattr(cfg.train, "entry", "")):
            print("parallel_runs needs `python -m train.<dataset>`. Running seeds sequentially.")
        else:
            _run_parallel(cfg, seeds, mode, group)
            summaries = _read_summaries(cfg, group, seeds)
            _report(summaries, mode, cfg)
            return summaries

    summaries = []
    for index, seed in enumerate(seeds):
        child = cfg.clone()
        child.seed = int(seed)
        child.train.runs = 1
        child.train.parallel_runs = False
        summary = fit_one_seed_cfg(
            child, int(seed), mode, group, index, create_dataset, create_model
        )
        if summary is not None:
            summaries.append(summary)
    _report(summaries, mode, cfg)
    return summaries


def fit_one_seed(cfg_yaml: str, seed: int, mode: str, group: str, run_index: int, remap_device: bool = False):
    import importlib

    from yacs.config import CfgNode as CN

    cfg = CN.load_cfg(cfg_yaml)
    if remap_device:
        cfg.device = 0
        cfg.train.devices = 0
        cfg.train.strategy = "auto"
    cfg.seed = int(seed)
    cfg.train.runs = 1
    cfg.train.parallel_runs = False
    module = importlib.import_module(str(cfg.train.entry))
    return fit_one_seed_cfg(
        cfg, int(seed), mode, group, int(run_index), module.create_dataset, module.create_model
    )


def fit_one_seed_cfg(cfg, seed: int, mode: str, group: str, run_index: int, create_dataset, create_model):
    configure_sharing()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    seed_everything(int(seed), workers=True, verbose=False)
    set_seed(int(seed))

    directory = run_directory(cfg, group, seed)
    experiment = Experiment(
        directory,
        meta={
            "group": group,
            "run_id": f"seed-{int(seed)}",
            "seed": int(seed),
            "dataset": str(cfg.dataset),
        },
        cfg=cfg,
    )
    experiment.write_config(cfg)
    experiment.event(
        "start",
        objective=str(cfg.train.objective),
        ema_scope=str(cfg.train.ema_scope),
        run_index=int(run_index),
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.train.batch_size),
    )
    print(f"\nRun {run_index} | seed {seed} | logs {directory}")
    print(f"Dashboard run {experiment.name} -> {experiment.live_path}")

    if mode == "kfold":
        summary = _fit_kfold(cfg, seed, run_index, group, experiment, create_dataset, create_model)
    else:
        summary = _fit_split(cfg, seed, run_index, group, experiment, create_dataset, create_model)
    if summary is not None:
        experiment.write_summary(summary)
    return summary


def _fit_split(cfg, seed, run_index, group, experiment, create_dataset, create_model):
    train_set, val_set, test_set = create_dataset(cfg)
    model = create_model(cfg)
    print(f"\nNumber of parameters: {count_parameters(model)}")
    lit, callback, trainer = _make_trainer(
        cfg, model, experiment,
        monitor="val_loss",
        val_names=["val_loss", "test_loss"],
        track_best=True,
    )
    data = GraphDataModule(train_set, val_set, test_set, cfg.train.batch_size, cfg.num_workers,
                          int(getattr(cfg.train, "prefetch_factor", 2)))
    trainer.fit(lit, datamodule=data, ckpt_path=_resume_path(cfg, experiment))
    _barrier(trainer)
    summary = None
    if trainer.is_global_zero:
        _save_weights(
            cfg, lit.model, lit.best_state, f"best_run{run_index}_seed{seed}",
            {"run": run_index, "seed": seed, "val_loss": lit.best_score, "group": group},
        )
        save_model(
            cfg, lit.model, f"last_run{run_index}_seed{seed}",
            {"run": run_index, "seed": seed, "group": group},
        )
        device = next(lit.model.parameters()).device
        probe = probe_regression(
            lit.model, train_set, test_set, cfg.train.batch_size, device, tag=f"run{run_index}-test"
        )
        summary = _timing_summary(callback, seed, run_index, extra=probe)
        _log_final(trainer, cfg, summary)
    _barrier(trainer)
    return summary


def _fit_kfold(cfg, seed, run_index, group, experiment, create_dataset, create_model):
    dataset, transform, transform_eval = create_dataset(cfg)
    if hasattr(dataset, "train_indices"):
        folds = dataset.train_indices, dataset.test_indices
    else:
        folds = k_fold(dataset, cfg.k)

    fold_rows = []
    best_balanced = -1.0
    diag_records = []
    model = None
    for fold, (train_idx, test_idx) in enumerate(zip(*folds)):
        experiment.fold = fold
        train_set = dataset[train_idx]
        test_set = dataset[test_idx]
        train_set.transform = transform
        test_set.transform = transform_eval
        test_set = [row for row in test_set]
        if not cfg.metis.online:
            train_set = [row for row in train_set]

        model = create_model(cfg)
        if fold == 0:
            print(f"\nNumber of parameters: {count_parameters(model)}")
        lit, callback, trainer = _make_trainer(
            cfg, model, experiment,
            monitor="test_loss",
            val_names=["test_loss"],
            track_best=False,
        )
        data = GraphDataModule(train_set, None, test_set, cfg.train.batch_size, cfg.num_workers,
                          int(getattr(cfg.train, "prefetch_factor", 2)))
        trainer.fit(lit, datamodule=data, ckpt_path=_resume_path(cfg, experiment))
        _barrier(trainer)
        if trainer.is_global_zero:
            save_model(
                cfg, lit.model, f"run{run_index}_fold{fold}_seed{seed}",
                {"run": run_index, "fold": fold, "seed": seed, "group": group},
            )
            device = next(lit.model.parameters()).device
            probe = probe_classification(
                lit.model, train_set, test_set, cfg.train.batch_size, device,
                tag=f"run{run_index}-fold{fold}",
            )
            diag_records.append(probe.pop("collapse"))
            if probe["balanced_acc"] > best_balanced:
                best_balanced = probe["balanced_acc"]
                save_model(
                    cfg, lit.model, "best",
                    {"run": run_index, "fold": fold, "seed": seed, "group": group, **probe},
                )
            row = _timing_summary(callback, seed, run_index, extra=probe)
            row["fold"] = fold
            fold_rows.append(row)
            _log_final(trainer, cfg, row)
            print(
                f"Fold {fold}, Seconds/epoch: {row['seconds_per_epoch']:.4f} | "
                f"balanced_acc={probe['balanced_acc']:.4f}"
            )
        _barrier(trainer)
        del trainer, lit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fold_rows:
        return None
    balanced = np.array([row["balanced_acc"] for row in fold_rows])
    macro = np.array([row["macro_f1"] for row in fold_rows])
    raw = np.array([row["raw_acc"] for row in fold_rows])
    print(
        f"Balanced Acc: {balanced.mean():.4f} ± {balanced.std():.4f} "
        f"| Macro-F1: {macro.mean():.4f} ± {macro.std():.4f} "
        f"| Raw Acc: {raw.mean():.4f} ± {raw.std():.4f}"
    )
    summarize_collapse(diag_records)
    print(
        f"\n>>> Best model (balanced_acc={best_balanced:.4f}) saved as "
        f"checkpoints/{cfg.dataset}/graphjepa_best.pt"
    )
    return {
        "seed": int(seed),
        "run_index": int(run_index),
        "train_loss": float(np.mean([row["train_loss"] for row in fold_rows])),
        "seconds_per_epoch": float(np.mean([row["seconds_per_epoch"] for row in fold_rows])),
        "hours": float(np.sum([row["hours"] for row in fold_rows])),
        "balanced_acc": float(balanced.mean()),
        "balanced_acc_std": float(balanced.std()),
        "macro_f1": float(macro.mean()),
        "raw_acc": float(raw.mean()),
        "best_balanced_acc": float(best_balanced),
        "folds": fold_rows,
    }


def _make_trainer(cfg, model, experiment, monitor, val_names, track_best):
    import lightning as L

    lit = GraphJepaLitModule(
        model, cfg, str(cfg.train.objective), str(cfg.train.ema_scope), monitor, val_names
    )
    lit.track_best = track_best
    callback = DashboardCallback(experiment)
    callbacks = [callback]
    if bool(getattr(cfg.train, "save_last", True)):
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks.append(ModelCheckpoint(
            dirpath=str(_checkpoint_dir(experiment)), save_top_k=0, save_last=True,
        ))
    accelerator, devices, strategy = _devices(cfg)
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=int(cfg.train.epochs),
        precision=str(getattr(cfg.train, "precision", "32-true")),
        logger=_loggers(cfg, experiment) or False,
        callbacks=callbacks,
        enable_checkpointing=len(callbacks) > 1,
        num_sanity_val_steps=0,
        inference_mode=False,
        default_root_dir=str(experiment.directory),
        **_trainer_flags(cfg),
    )
    return lit, callback, trainer


def _trainer_flags(cfg) -> dict:
    """``cfg.trainer`` keys that are set, passed straight to ``L.Trainer``."""
    section = getattr(cfg, "trainer", None)
    if section is None:
        return {"log_every_n_steps": 1}
    return {key: value for key, value in section.items() if value is not None}


def _loggers(cfg, experiment) -> list:
    """Lightning loggers from ``logging.loggers``. All write under the run directory."""
    names = [str(name).lower() for name in getattr(cfg.logging, "loggers", ["tensorboard"])]
    if not bool(getattr(cfg.logging, "tensorboard", True)):
        names = [name for name in names if name != "tensorboard"]
    version = "latest" if experiment.fold is None else f"fold{experiment.fold}"
    directory = str(experiment.directory)
    project = str(getattr(cfg.logging, "project", "graph-jepa"))
    loggers = []
    for name in names:
        if name == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            loggers.append(TensorBoardLogger(
                save_dir=directory, name="tb", version=version, default_hp_metric=False,
            ))
        elif name == "csv":
            from lightning.pytorch.loggers import CSVLogger

            loggers.append(CSVLogger(save_dir=directory, name="csv", version=version))
        elif name == "wandb":
            from lightning.pytorch.loggers import WandbLogger

            run_name = experiment.name if experiment.fold is None else f"{experiment.name}__{version}"
            loggers.append(WandbLogger(
                project=project, name=run_name, group=str(experiment.meta.get("group")),
                save_dir=directory,
            ))
        elif name == "mlflow":
            from lightning.pytorch.loggers import MLFlowLogger

            uri = str(getattr(cfg.logging, "mlflow_uri", "") or "")
            if not uri:
                uri = (Path(directory).parents[1] / "mlruns").resolve().as_uri()
            run_name = experiment.name if experiment.fold is None else f"{experiment.name}__{version}"
            loggers.append(MLFlowLogger(experiment_name=project, run_name=run_name, tracking_uri=uri))
        else:
            raise SystemExit(
                f"Unknown logger `{name}` in logging.loggers. Use tensorboard, csv, wandb or mlflow."
            )
    return loggers


def _log_final(trainer, cfg, summary) -> None:
    """Probe results as metrics, so TensorBoard's HPARAMS tab can sort runs by them."""
    metrics = {
        f"final/{key}": float(value) for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and key not in {"seed", "run_index", "fold"}
    }
    for logger in trainer.loggers:
        if type(logger).__name__ == "TensorBoardLogger":
            logger.log_hyperparams(flatten_cfg(cfg), metrics)
        else:
            logger.log_metrics(metrics, step=int(trainer.global_step))
        logger.save()
        if type(logger).__name__ == "WandbLogger":
            # one W&B run per fold or seed
            logger.experiment.finish()


def _checkpoint_dir(experiment) -> Path:
    directory = experiment.directory / "checkpoints"
    if experiment.fold is not None:
        directory = directory / f"fold{experiment.fold}"
    return directory


def _check_resume(cfg, seeds, mode) -> None:
    value = str(getattr(cfg.train, "resume", "") or "")
    if value == "auto" and not getattr(cfg, "logfile", None):
        print("train.resume auto needs a fixed run name (--name / logfile). Starting fresh.")
    elif value and value != "auto" and (len(seeds) > 1 or mode == "kfold"):
        raise SystemExit(
            "A checkpoint path resumes one split run with one seed. "
            "Use train.resume auto (with --name) for several seeds or k-fold."
        )


def _resume_path(cfg, experiment):
    value = str(getattr(cfg.train, "resume", "") or "")
    if not value:
        return None
    if value == "auto":
        path = _checkpoint_dir(experiment) / "last.ckpt"
        if not path.is_file():
            return None
    else:
        path = Path(value)
        if not path.is_file():
            raise SystemExit(f"train.resume: checkpoint not found: {value}")
    print(f"Resuming from {path}")
    return str(path)


def _devices(cfg):
    requested = int(getattr(cfg.train, "devices", 0) or 0)
    strategy_name = str(getattr(cfg.train, "strategy", "auto"))
    if torch.cuda.is_available():
        if strategy_name == "ddp" or requested > 1:
            count = requested if requested > 1 else torch.cuda.device_count()
            return "gpu", count, DDPStrategy(find_unused_parameters=True)
        return "gpu", [int(cfg.device)], "auto"
    if torch.backends.mps.is_available():
        return "mps", 1, "auto"
    return "cpu", 1, "auto"


def _save_weights(cfg, model, state, tag, extra):
    if not state:
        return
    live = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    try:
        save_model(cfg, model, tag, extra)
    finally:
        model.load_state_dict(live)


def _timing_summary(callback, seed, run_index, extra):
    seconds = callback.epoch_seconds or [0.0]
    train_loss = float(callback.last_metrics.get("train_loss", float("nan")))
    summary = {
        "seed": int(seed),
        "run_index": int(run_index),
        "train_loss": train_loss,
        "seconds_per_epoch": float(np.mean(seconds)),
        "hours": float(np.sum(seconds) / 3600.0),
        "epochs_ran": len(callback.epoch_seconds),
    }
    for key, value in extra.items():
        if key == "collapse":
            summary["collapse"] = value
        else:
            summary[key] = value
    print(
        f"AVG TIME PER EPOCH: {summary['seconds_per_epoch']:.4f} s | "
        f"TOTAL TIME TAKEN: {summary['hours']:.4f} h"
    )
    return summary


def _barrier(trainer) -> None:
    strategy = getattr(trainer, "strategy", None)
    barrier = getattr(strategy, "barrier", None)
    if callable(barrier) and int(getattr(trainer, "world_size", 1) or 1) > 1:
        barrier()


def _group_name(cfg) -> str:
    if getattr(cfg, "logfile", None):
        raw = str(cfg.logfile)
    else:
        raw = f"{cfg.dataset}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in raw)
    return cleaned or "run"


def _use_parallel(cfg, seeds) -> bool:
    if not bool(getattr(cfg.train, "parallel_runs", False)) or len(seeds) < 2:
        return False
    strategy = str(getattr(cfg.train, "strategy", "auto"))
    requested = int(getattr(cfg.train, "devices", 0) or 0)
    if strategy == "ddp" or requested > 1:
        print("DDP uses the requested GPUs for one seed at a time, so seeds stay sequential.")
        return False
    if torch.cuda.device_count() < 2:
        print("parallel_runs needs at least 2 visible GPUs. Seeds will run one after another.")
        return False
    return True


def _run_parallel(cfg, seeds, mode, group) -> None:
    import torch.multiprocessing as mp

    devices = list(range(torch.cuda.device_count()))
    width = min(len(seeds), len(devices))
    free = devices[:width]
    pending = list(enumerate(seeds))
    active = []
    ctx = mp.get_context("spawn")
    base = cfg.clone()
    base.train.parallel_runs = False
    base.num_workers = max(1, int(cfg.num_workers) // width)
    failed = []
    while pending or active:
        while pending and free and not failed:
            index, seed = pending.pop(0)
            device = free.pop(0)
            child = base.clone()
            child.seed = int(seed)
            child.train.runs = 1
            process = ctx.Process(
                target=spawn_fit,
                args=(str(device), child.dump(), int(seed), mode, group, int(index)),
            )
            process.start()
            print(f"Started seed {seed} on cuda:{device} (pid {process.pid})")
            active.append((process, seed, device))
        if not active:
            break
        process, seed, device = active.pop(0)
        process.join()
        if process.exitcode != 0:
            failed.append(seed)
            print(f"Seed {seed} on cuda:{device} exited with status {process.exitcode}")
        else:
            free.append(device)
    for process, seed, device in active:
        process.terminate()
        process.join()
    if failed:
        raise RuntimeError(f"Parallel seeds failed: {failed}")


def _read_summaries(cfg, group, seeds):
    rows = []
    for seed in seeds:
        path = run_directory(cfg, group, seed) / "summary.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _report(summaries, mode, cfg) -> None:
    if len(summaries) < 2:
        return
    train = np.array([row["train_loss"] for row in summaries], dtype=float)
    seconds = np.array([row["seconds_per_epoch"] for row in summaries], dtype=float)
    hours = np.array([row["hours"] for row in summaries], dtype=float)
    print(
        f"\nFinal Train Loss: {np.nanmean(train):.4f} ± {np.nanstd(train):.4f}"
        f"\nSeconds/epoch: {np.nanmean(seconds):.4f}"
        f"\nHours/total: {np.nanmean(hours):.4f}"
    )
    if mode == "split" and "mae" in summaries[0]:
        maes = np.array([row["mae"] for row in summaries], dtype=float)
        print(f"MAE avg: {maes.mean()}, std: {maes.std()}")
    if mode == "kfold" and "balanced_acc" in summaries[0]:
        stacked = np.array(
            [[row["balanced_acc"], row.get("balanced_acc_std", 0.0), row["macro_f1"], row["raw_acc"]]
             for row in summaries],
            dtype=float,
        )
        print("Averages over runs (balanced_acc, bal_std, macro_f1, raw_acc):")
        print(stacked.mean(axis=0))
        print(f"checkpoints/{cfg.dataset}/graphjepa_best.pt holds the best probe seen in each process.")
