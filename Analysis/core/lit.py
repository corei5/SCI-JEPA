"""PyTorch Lightning wrappers around Graph-JEPA.

Optimization is manual so the EMA target update stays in the same place as the
old loops: after ``optimizer.step()``, once per batch. Momentum now moves on
every step from ``ema_momentum`` to ``ema_momentum_end``. The legacy backend
still advances it once per epoch.
"""

from __future__ import annotations

import torch
import lightning as L
from torch_geometric.loader import DataLoader

from core.config import flatten_cfg
from core.objectives import OBJECTIVES, ema_bounds, ema_update, interpolate_momentum


def configure_sharing() -> None:
    """Prefer file-system tensor sharing so DataLoader workers do not exhaust fds."""
    import torch.multiprocessing as mp

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


class GraphDataModule(L.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size: int, num_workers: int,
                 prefetch_factor: int = 2):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)

    def _loader(self, dataset, shuffle: bool):
        workers = self.num_workers
        extra = {}
        if workers > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            **extra,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        loaders = []
        if self.val_set is not None:
            loaders.append(self._loader(self.val_set, shuffle=False))
        if self.test_set is not None:
            loaders.append(self._loader(self.test_set, shuffle=False))
        if not loaders:
            raise RuntimeError("A Lightning run needs a validation or test set.")
        if len(loaders) == 1:
            return loaders[0]
        return loaders


class GraphJepaLitModule(L.LightningModule):
    def __init__(self, model, cfg, objective: str, ema_scope: str, monitor: str, val_names: list[str]):
        super().__init__()
        self.automatic_optimization = False
        # Every config key, flattened, so loggers record the run's hyperparameters.
        self.save_hyperparameters(flatten_cfg(cfg))
        self.model = model
        self.cfg = cfg
        self.objective_name = objective
        self.step_fn = OBJECTIVES[objective]
        self.ema_scope = ema_scope
        self.monitor = monitor
        self.val_names = list(val_names)
        self.best_score = float("inf")
        self.best_state = None
        self.track_best = True
        self._stopped = False
        self._sched_epoch = -1
        self.use_asam = str(getattr(cfg.train, "optimizer", "Adam")) == "ASAM"
        self.grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
        self.min_lr = float(cfg.train.min_lr)
        self.minimizer = None

    def configure_optimizers(self):
        params = self.model.parameters()
        lr = float(self.cfg.train.lr)
        weight_decay = float(self.cfg.train.wd)
        # fused kernels need every parameter on CUDA; Lightning moves the model before this call
        fused = {"fused": True} if (
            bool(getattr(self.cfg.train, "fused_optim", False)) and torch.cuda.is_available()
            and all(p.is_cuda for p in self.model.parameters())) else {}
        if self.use_asam or str(self.cfg.train.optimizer) == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, **fused)
        else:
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **fused)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.cfg.train.lr_decay),
            patience=int(self.cfg.train.lr_patience),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.monitor,
                "strict": False,
            },
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["graph_jepa"] = {
            "best_score": self.best_score,
            "best_state": self.best_state,
            "stopped": self._stopped,
            "sched_epoch": self._sched_epoch,
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        state = checkpoint.get("graph_jepa") or {}
        self.best_score = float(state.get("best_score", self.best_score))
        self.best_state = state.get("best_state", self.best_state)
        self._stopped = bool(state.get("stopped", False))
        self._sched_epoch = int(state.get("sched_epoch", -1))

    def on_train_start(self) -> None:
        # a resumed run that had already hit min_lr stays stopped
        if self._stopped:
            self.trainer.should_stop = True

    def on_fit_start(self) -> None:
        if not self.use_asam:
            return
        from core.asam import ASAM

        self.minimizer = ASAM(self.optimizers(), self.model, rho=0.5)

    def _momentum(self) -> float:
        start, end = ema_bounds(self.cfg)
        total = int(self.trainer.estimated_stepping_batches or 0)
        return interpolate_momentum(int(self.global_step), total, start, end)

    def _sync(self) -> bool:
        return int(getattr(self.trainer, "world_size", 1) or 1) > 1

    def _log_step(self, out: dict, prefix: str) -> None:
        sync = self._sync()
        batch_size = int(out["batch_size"])
        if prefix == "train":
            # written every trainer.log_every_n_steps steps
            self.log("train_loss_step", out["pred_loss"], on_step=True, on_epoch=False,
                     batch_size=batch_size, add_dataloader_idx=False)
        self.log(
            f"{prefix}_loss",
            out["pred_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync,
            add_dataloader_idx=False,
        )
        extra = {
            "total_loss": out["loss"].detach(),
            "var_loss": out.get("var_loss"),
            "cov_loss": out.get("cov_loss"),
            "repr_std": out.get("repr_std"),
        }
        for name, value in extra.items():
            if value is None:
                continue
            self.log(
                f"{prefix}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=sync,
                add_dataloader_idx=False,
            )

    def training_step(self, batch, batch_idx):
        momentum = self._momentum()
        optimizer = self.optimizers()
        if self.use_asam:
            optimizer.zero_grad(set_to_none=True)
            first = self.step_fn(self.model, batch, self.cfg, training=True)
            self.manual_backward(first["loss"])
            self.minimizer.ascent_step()
            out = self.step_fn(self.model, batch, self.cfg, training=True)
            self.manual_backward(out["loss"])
            self.minimizer.descent_step()
        else:
            optimizer.zero_grad(set_to_none=True)
            out = self.step_fn(self.model, batch, self.cfg, training=True)
            self.manual_backward(out["loss"])
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
        ema_update(self.model, momentum, self.ema_scope)
        self._log_step(out, "train")
        self.log(
            "momentum",
            float(momentum),
            on_step=False,
            on_epoch=True,
            batch_size=int(out["batch_size"]),
            add_dataloader_idx=False,
        )
        self.log(
            "lr",
            float(optimizer.param_groups[0]["lr"]),
            on_step=False,
            on_epoch=True,
            batch_size=1,
            add_dataloader_idx=False,
        )
        if batch_idx == 0 and self.objective_name == "vicreg" and self.trainer.is_global_zero:
            self._debug_representations(batch, out, momentum)

    def _debug_representations(self, batch, out, momentum: float) -> None:
        with torch.no_grad():
            encoded = self.model.encode(batch)
            encode_std = float(encoded.std(0).mean())
            dead = int((encoded.std(0) < 1e-3).sum())
        repr_std = float(out["repr_std"]) if "repr_std" in out else float("nan")
        var_loss = float(out["var_loss"]) if "var_loss" in out else float("nan")
        cov_loss = float(out["cov_loss"]) if "cov_loss" in out else float("nan")
        pred = float(out["pred_loss"])
        width = int(encoded.shape[-1])
        print(
            f"[DEBUG] repr_z.std(mean)={repr_std:.4f} | var_loss={var_loss:.4f} | "
            f"cov_loss={cov_loss:.4f} | inv_loss={pred:.4f} || "
            f"encode().std(mean)={encode_std:.4f} dead_dims={dead}/{width} | "
            f"momentum={momentum:.4f}"
        )

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        out = self.step_fn(self.model, batch, self.cfg, training=False)
        name = self.val_names[dataloader_idx]
        self.log(
            name,
            out["pred_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(out["batch_size"]),
            sync_dist=self._sync(),
            add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        epoch = int(self.trainer.current_epoch)
        if epoch == self._sched_epoch:
            return
        self._sched_epoch = epoch
        metrics = self.trainer.callback_metrics
        if self.monitor in metrics:
            score = float(metrics[self.monitor])
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, (list, tuple)):
                scheduler = scheduler[0] if scheduler else None
            if scheduler is not None:
                scheduler.step(score)
            if self.track_best and score < self.best_score:
                self.best_score = score
                if self.trainer.is_global_zero:
                    self.best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
        self._maybe_stop()

    def _maybe_stop(self) -> None:
        if self._stopped:
            return
        learning_rate = float(self.optimizers().param_groups[0]["lr"])
        if learning_rate < self.min_lr:
            if self.trainer.is_global_zero:
                print("!! LR EQUAL TO MIN LR SET.")
            self.trainer.should_stop = True
            self._stopped = True
