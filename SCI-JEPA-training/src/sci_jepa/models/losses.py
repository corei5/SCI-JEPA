"""
L = sum_y w_y ||Pred(E(x), z_tau) - sg(E_bar(y))||^2
    + lambda_s L_SHACL + lambda_a L_align + lambda_r L_reg
    + lambda_inv L_inverse + lambda_contrast L_contrast

L_inverse makes this a hybrid rather than a pure JEPA (plan, mechanism two);
L_align is contrastive. Both are off by default so the reference arm is the
plain objective.
"""
import math

import torch
import torch.nn.functional as F

from sci_jepa.data.graph_index import HUB


def jepa_loss(pred, target, kind="l2", weights=None):
    if kind == "l2":
        # mean over dimensions, not sum: keeps every lambda on the same scale
        # across model sizes (a sum is ~d at init and drowns the other terms)
        per = (pred - target).pow(2).mean(-1)
    elif kind == "cos":
        per = 1 - F.cosine_similarity(pred, target, dim=-1)
    else:
        raise ValueError(kind)
    if weights is not None:                          # E9 confidence weights, when they exist
        per = per * weights
    return per.mean()


def inverse_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def align_loss(pred, target, ttype, tau):
    """InfoNCE between predictions and targets, negatives restricted to the same
    component type so the aspect shortcut cannot solve it."""
    p, t = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    logits = p @ t.T / tau
    logits = logits.masked_fill(ttype[:, None] != ttype[None, :], float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(p), device=p.device))


def contrast_margin(pred, pred_pair, margin):
    """Relational contrast masking: minimal pairs (one context component swapped)
    should not produce the same prediction. Expected to act as a floor that keeps
    the predictor off the identity map, not as a selector (H-JEPA App. L.6)."""
    d = (F.normalize(pred, dim=-1) - F.normalize(pred_pair, dim=-1)).norm(dim=-1)
    return F.relu(margin - d).mean()


# ----------------------------------------------------------------------
# grouping regulariser (E12): an Epps-Pulley normality test on random
# 1-D projections, i.e. a SIGReg-style marginal regulariser, applied either
# within one subgraph or across subgraphs.
# ----------------------------------------------------------------------
_T = torch.linspace(-3, 3, 17)


def epps_pulley(z):
    """z: (n, K) projected samples -> mean EP statistic vs N(0, 1)."""
    t = _T.to(z.device)
    arg = z.unsqueeze(-1) * t                        # (n, K, |t|)
    ecf_re, ecf_im = arg.cos().mean(0), arg.sin().mean(0)
    target = torch.exp(-t ** 2 / 2)
    w = torch.exp(-t ** 2 / 2)
    stat = ((ecf_re - target) ** 2 + ecf_im ** 2) * w
    return torch.trapz(stat, t, dim=-1).mean()


def grouping_regulariser(h, slot, typ, pad, grouping, n_proj, gen=None):
    """h: (B, T, d) context token states. Groups:
       within: components of ONE subgraph form a normality test
       across: ONE component from each subgraph of an instance forms a test
    Plan prediction (H-JEPA Prop. 2 transposed): within carries a barrier
    against subgraph-identity collapse, across carries none."""
    d = h.size(-1)
    U = F.normalize(torch.randn(d, n_proj, device=h.device, generator=gen), dim=0)
    comp = (~pad) & (typ != HUB)
    stats = []
    for b in range(h.size(0)):
        s = slot[b][comp[b]]
        z = h[b][comp[b]] @ U                         # identical for both groupings
        if grouping == "within":
            for k in s.unique():
                g = z[s == k]
                if len(g) >= 3:
                    stats.append(epps_pulley(g))
        else:
            ids = [torch.nonzero(s == k, as_tuple=True)[0] for k in s.unique()]
            if len(ids) < 3:
                continue
            for r in range(min(len(i) for i in ids)):
                stats.append(epps_pulley(z[torch.stack([i[r] for i in ids])]))
    return torch.stack(stats).mean() if stats else h.sum() * 0.0
