"""Losses used by the Lightning trainer.

``distance`` is the original SmoothL1 / MSE / hyperbolic prediction loss.
``vicreg`` is the papers objective: a cosine prediction term plus variance and
covariance penalties. The value logged as the epoch loss is the prediction
term, matching the previous training scripts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from core.model_utils.hyperbolic_dist import hyperbolic_dist


def interpolate_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return float(start)
    progress = min(max(step, 0) / float(total_steps), 1.0)
    return float(start) + progress * (float(end) - float(start))


def ema_bounds(cfg) -> tuple[float, float]:
    start = float(getattr(cfg.jepa, "ema_momentum", 0.996))
    end = float(getattr(cfg.jepa, "ema_momentum_end", 0.999))
    return start, end


def resolve_plan(cfg, train_fn) -> tuple[str, str]:
    objective = str(getattr(cfg.train, "objective", "auto"))
    if objective == "auto":
        module = getattr(train_fn, "__module__", "") if train_fn is not None else ""
        dataset = str(getattr(cfg, "dataset", ""))
        objective = "vicreg" if module.endswith("papers") or dataset == "papers" else "distance"
    if objective not in {"vicreg", "distance"}:
        raise ValueError(
            f"Unknown train.objective {objective!r}. Use auto, distance, or vicreg."
        )

    scope = str(getattr(cfg.train, "ema_scope", "full"))
    if scope == "auto":
        scope = "full"
    if scope not in {"full", "encoder"}:
        raise ValueError(f"Unknown train.ema_scope {scope!r}. Use full or encoder.")
    return objective, scope


def regression_loss(target_x, target_y, criterion_type: int, beta: float):
    if int(criterion_type) == 0:
        return F.smooth_l1_loss(target_x, target_y, beta=float(beta))
    if int(criterion_type) == 1:
        return F.mse_loss(target_x, target_y)
    if int(criterion_type) == 2:
        return hyperbolic_dist(target_x, target_y)
    raise ValueError(f"Loss function {criterion_type} is not supported.")


def prediction_loss(pred, target):
    cosine = (pred * target).sum(dim=-1)
    return (1.0 - cosine).mean()


def variance_loss(z, gamma: float = 1.0, eps: float = 1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    n, d = z.shape
    if n <= 1:
        return z.sum() * 0.0
    centered = z - z.mean(dim=0)
    cov = (centered.T @ centered) / (n - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / d


def prepare_batch(model, batch, training: bool):
    if training and getattr(model, "use_lap", False):
        encoding = getattr(batch, "lap_pos_enc", None)
        if encoding is not None:
            sign = torch.rand(encoding.size(1), device=encoding.device)
            sign = torch.where(sign >= 0.5, torch.ones_like(sign), -torch.ones_like(sign))
            batch.lap_pos_enc = encoding * sign.unsqueeze(0)
    return batch


def _batch_size(batch, target) -> int:
    graphs = getattr(batch, "num_graphs", None)
    if graphs:
        return int(graphs)
    return int(target.shape[0])


def distance_step(model, batch, cfg, training: bool) -> dict:
    batch = prepare_batch(model, batch, training)
    target_x, target_y = model(batch)
    beta = float(getattr(cfg.train, "smooth_l1_beta", 0.5))
    criterion = int(getattr(cfg.jepa, "dist", 0))
    loss = regression_loss(target_x, target_y, criterion, beta)
    return {
        "loss": loss,
        "pred_loss": loss.detach(),
        "batch_size": _batch_size(batch, target_y),
    }


def vicreg_step(model, batch, cfg, training: bool) -> dict:
    batch = prepare_batch(model, batch, training)
    target_x, target_y, repr_z = model(batch, return_repr=True)
    invariant = prediction_loss(target_y, target_x)
    variance = variance_loss(repr_z)
    covariance = covariance_loss(repr_z)
    sim_weight = float(getattr(cfg.train, "sim_weight", 1.0))
    var_weight = float(getattr(cfg.train, "var_weight", 1.0))
    cov_weight = float(getattr(cfg.train, "cov_weight", 0.04))
    loss = sim_weight * invariant + var_weight * variance + cov_weight * covariance
    return {
        "loss": loss,
        "pred_loss": invariant.detach(),
        "var_loss": variance.detach(),
        "cov_loss": covariance.detach(),
        "repr_std": repr_z.detach().std(dim=0).mean(),
        "batch_size": _batch_size(batch, target_y),
    }


OBJECTIVES = {
    "distance": distance_step,
    "vicreg": vicreg_step,
}


@torch.no_grad()
def ema_update(model, momentum: float, scope: str) -> None:
    """Advance the target network.

    ``full`` updates every online/target pair (the papers fix). ``encoder``
    updates only the context encoder, which is what the older TU scripts did.
    """
    if scope == "full" and hasattr(model, "update_target"):
        model.update_target(momentum)
        return
    pairs = zip(model.context_encoder.parameters(), model.target_encoder.parameters())
    for online, target in pairs:
        target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
