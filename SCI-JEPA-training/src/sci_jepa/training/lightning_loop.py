"""
The training loop of train_one on PyTorch Lightning (cfg.trainer = "lightning").

train_one keeps everything around the loop: model, optimizer, LR schedule, batch plan, epoch-end validation and
the final test evaluation. Only the loop over training steps runs here, as a LightningModule with manual
optimization, so every step does the same arithmetic as the native loop: the run's own autocast around the
forward pass, losses in fp32, fp16 loss scaling, gradient clipping, the per-step LR schedule and the EMA target
update.

cfg.devices > 1 trains ONE run on several GPUs (Lightning DDP). The global batch stays cfg.batch_size: every
rank builds batch_size / devices instances of the same step, and DDP averages the gradients. The model has no
batch statistics (LayerNorm only) and the loss is a mean over the batch, so the averaged gradient is the
gradient of the full batch: same optimization, the step's work split across GPUs. Validation and the final
evaluation run on rank 0; the other ranks return None.

Not supported here (train_one falls back to the native loop): resuming from a checkpoint, the E5 contrast loss.
"""
import os
import time

import numpy as np
import torch

from sci_jepa.models import losses as L
from sci_jepa.models.model import to_device


def supported(cfg, ckpt_path):
    """(ok, reason): whether this run can use the Lightning loop."""
    if ckpt_path:
        return False, "resuming from checkpoints is not supported by the Lightning loop"
    if cfg.lambda_contrast > 0:
        return False, "the E5 contrast loss draws from the sampler inside the step"
    if cfg.devices > 1 and cfg.batch_size % cfg.devices:
        return False, f"batch_size {cfg.batch_size} is not divisible by devices {cfg.devices}"
    return True, ""


def _rank_plan(plan_fn, rank, world):
    """The step plan of one rank: a per-type count dict is split across the ranks (None = one shared pool)."""
    def f(step):
        plan, phase = plan_fn(step)
        if isinstance(plan, dict):                      # c instances of type t: rank r builds c//W, +1 if r < c%W
            plan = {t: c // world + (1 if rank < c % world else 0) for t, c in plan.items()}
        return plan, phase
    return f


def fit_lightning(*, model, cfg, gi, dev, seed, log, live, opt, sched, scaler, params, sampler, plan_fn, n_steps,
                  prefetch, workers, epoch_ends, epoch_of, track_best, shapes, gen, t0, history, epoch_results,
                  report, prog):
    import lightning as Lg
    from lightning.pytorch.strategies import DDPStrategy

    from sci_jepa.training import train as T

    world = max(int(cfg.devices), 1)
    state = {"ep_sum": 0.0, "ep_n": 0, "best": None, "stopped": None, "step": -1, "phase": None}

    class PaperALit(Lg.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model
            self.automatic_optimization = False

        def configure_optimizers(self):
            return opt

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            step, phase, insts, cpu_batch, sampler_state = batch
            return step, phase, insts, (to_device(cpu_batch, device) if cpu_batch is not None else None), sampler_state

        def training_step(self, item, batch_idx):
            step, phase_now, insts, batch, _ = item
            state["step"] = step
            if phase_now != state["phase"] and cfg.mask_schedule == "sequential" and self.global_rank == 0:
                log(f"  [mask] step {step}: phase -> masking '{phase_now}'")
            state["phase"] = phase_now
            if batch is None:
                return None
            with T._autocast(cfg, self.device):
                out = self.model(batch)
            out = T._fp32(out)
            if "pred_extra" in out:
                parts = {"jepa": L.jepa_loss(torch.cat([out["pred"], out["pred_extra"]]),
                                             torch.cat([out["target"], out["target_extra"]]), cfg.jepa_loss)}
            else:
                parts = {"jepa": L.jepa_loss(out["pred"], out["target"], cfg.jepa_loss)}
            if "inv_logits" in out:
                if cfg.lambda_inv > 0:
                    parts["inverse"] = cfg.lambda_inv * L.inverse_loss(out["inv_logits"], out["inv_labels"])
                else:
                    b, i, j, _ = batch["probe"]
                    h = out["h"].detach()
                    parts["inverse_probe"] = L.inverse_loss(self.model.inverse(h[b, i], h[b, j]), out["inv_labels"])
            if cfg.lambda_reg > 0:
                c = batch["ctx"]
                parts["reg"] = cfg.lambda_reg * L.grouping_regulariser(
                    out["h"], c["slot"], c["type"], c["pad"], cfg.reg_grouping, cfg.reg_projections, gen)
            if cfg.lambda_align > 0:
                parts["align"] = cfg.lambda_align * L.align_loss(out["pred"], out["target"], batch["target_type"],
                                                                 cfg.align_tau)
            if cfg.lambda_shacl > 0 and shapes is not None:
                parts["shacl"] = cfg.lambda_shacl * shapes.loss(out["pred"], batch["target_comp"], batch["target_type"])
            loss = sum(parts.values())
            state["ep_sum"], state["ep_n"] = state["ep_sum"] + parts["jepa"].detach(), state["ep_n"] + 1
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                self.manual_backward(scaler.scale(loss))
                scaler.unscale_(opt)
            else:
                self.manual_backward(loss)
            logging_now = step % cfg.log_every == 0 or step == n_steps - 1
            gnorm = T._grad_norms(self.model) if (logging_now and live is not None) else None
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * step / max(n_steps - 1, 1)
            self.model.ema(m)
            if time.time() - prog["t"] >= 5 and self.global_rank == 0:
                report("training", step)
            if logging_now and self.global_rank == 0:
                rec = {"step": step, "epoch": epoch_of(step), "phase": phase_now,
                       **{k: float(v.detach()) for k, v in parts.items()},
                       "flagged": float(np.mean([a["flagged"] for a in batch["audit"]])),
                       "sec": round(time.time() - t0, 1)}
                history.append(rec)
                log("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
                if live is not None:
                    live.relbias(self.model)
                    live.step({**rec, "total": float(loss.detach()), "lr": float(sched.get_last_lr()[0]), "ema": m,
                               "grad_norm": gnorm, **T._step_stats(out, batch, cfg)})
            return None

        def on_train_start(self):
            if int(self.trainer.world_size) != world:   # e.g. a SLURM job with 1 task: Lightning would run 1 process
                raise RuntimeError(f"Lightning runs {self.trainer.world_size} process(es) for devices={world}")

        def on_train_end(self):
            if world > 1:                               # replicas must hold identical weights after every step
                with torch.no_grad():
                    ck = torch.stack([sum(p.detach().double().sum() for p in self.model.context.parameters()),
                                      sum(p.detach().double().sum() for p in self.model.predictor.parameters())])
                allck = self.trainer.strategy.all_gather(ck)
                if self.global_rank == 0:
                    spread = float((allck - allck[0]).abs().max())
                    log(f"  [lightning] weight checksums of the {allck.shape[0]} replicas: max difference {spread:.3e}")

        def on_train_batch_end(self, outputs, item, batch_idx):
            step = item[0]
            if step not in epoch_ends:
                return
            stop = False
            if self.global_rank == 0:
                rec = T.epoch_checkpoint(self.model, gi, cfg, self.device, epoch_ends[step], step,
                                         float(state["ep_sum"]) / max(state["ep_n"], 1), time.time() - t0, live)
                epoch_results.append(rec)
                rec["select_score"] = T._select_score(rec, cfg)
                best = state["best"]
                if track_best and (best is None or rec["select_score"] > best["score"]):
                    state["best"] = best = {"epoch": rec["epoch"], "score": rec["select_score"],
                                            "state": {k: v.detach().cpu().clone()
                                                      for k, v in self.model.state_dict().items()}}
                log(f"  [epoch {rec['epoch']}] val balanced MRR {rec['bal_MRR']:.4f} (chance {rec['bal_chance']:.4f}) "
                    f"train jepa {rec['train_jepa']:.4g} {cfg.select_metric} {rec['select_score']:.3f}")
                if (cfg.patience > 0 and best is not None and rec["epoch"] >= cfg.min_epochs
                        and rec["epoch"] - best["epoch"] >= cfg.patience):
                    state["stopped"] = rec["epoch"]
                    log(f"  [early stop] no better validation {cfg.select_metric} for {cfg.patience} epochs: "
                        f"stopped after epoch {rec['epoch']}, best epoch {best['epoch']}")
                    stop = True
            state["ep_sum"], state["ep_n"] = 0.0, 0
            if world > 1:
                stop = self.trainer.strategy.broadcast(stop, src=0)
            if stop:
                self.trainer.should_stop = True

    rank = 0
    if world > 1:                                       # each rank: its own share of every step, own random stream
        import os
        rank = int(os.environ.get("LOCAL_RANK", 0))
        from sci_jepa.data.instances import Sampler
        offset = 0 if os.environ.get("SCIJEPA_DDP_SAME_DATA") else 7_919 * rank   # debug: every rank the same data
        sampler = Sampler(gi, cfg, "train", seed + offset)
        plan_fn = _rank_plan(plan_fn, rank, world)
    else:
        offset = 0
    batches = T._batches(sampler, plan_fn, n_steps, cfg.batch_size // world, prefetch, 0, None, workers,
                         seed + offset)
    # LightningEnvironment: Lightning starts the other ranks itself. Inside a SLURM job it would otherwise read the
    # world size from SLURM (ntasks) and, with one task, silently train on ONE process.
    from lightning.pytorch.plugins.environments import LightningEnvironment
    strategy = DDPStrategy(find_unused_parameters=False, cluster_environment=LightningEnvironment()) \
        if world > 1 else "auto"
    trainer = Lg.Trainer(accelerator="gpu" if dev.type == "cuda" else "cpu", devices=world, strategy=strategy,
                         precision="32-true", max_epochs=1, max_steps=-1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, use_distributed_sampler=False, inference_mode=False)
    log(f"  [lightning] {world} device(s), {cfg.batch_size // world} instances per device and step, "
        f"precision handled by the run's own autocast (amp={cfg.amp})")
    trainer.fit(PaperALit(), train_dataloaders=batches)
    model.to(dev)                                       # Lightning's teardown moves the model to the CPU
    if trainer.global_rank != 0:
        return None
    return state["best"], state["stopped"], state["step"], state["phase"]
