"""Epoch metrics mirrored into the run directory and the terminal."""

from __future__ import annotations

import time

from lightning.pytorch.callbacks import Callback


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class DashboardCallback(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.epoch_seconds: list[float] = []
        self.last_metrics: dict = {}
        self._started = None
        self._logged_epoch = -1

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self._started = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        epoch = int(trainer.current_epoch)
        if epoch == self._logged_epoch:
            return
        self._logged_epoch = epoch
        elapsed = 0.0 if self._started is None else time.perf_counter() - self._started
        self.epoch_seconds.append(elapsed)
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            try:
                metrics[key] = _scalar(value)
            except (TypeError, ValueError):
                continue
        self.last_metrics = metrics
        self.experiment.event(
            "epoch",
            epoch=int(trainer.current_epoch),
            seconds=round(elapsed, 4),
            metrics=metrics,
        )
        self._print(int(trainer.current_epoch), metrics, elapsed)

    def _print(self, epoch: int, metrics: dict, elapsed: float) -> None:
        if self.experiment.fold is None:
            bits = [f"Epoch: {epoch:03d}"]
        else:
            bits = [f"Epoch/Fold: {epoch:03d}/{self.experiment.fold}"]
        if "train_loss" in metrics:
            bits.append(f"Train Loss: {metrics['train_loss']:.4f}")
        if "val_loss" in metrics:
            bits.append(f"Val: {metrics['val_loss']:.4f}")
        if "test_loss" in metrics:
            bits.append(f"Test: {metrics['test_loss']:.4f}")
        bits.append(f"Seconds: {elapsed:.4f}")
        print(", ".join(bits))
