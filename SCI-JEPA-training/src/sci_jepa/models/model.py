"""
Context encoder E_theta, EMA target encoder E_bar, conditioned predictor
Pred_phi, inverse-relation head I and (optionally) the shape bank.

The encoder is a graph transformer over instance tokens: full self-attention
plus a learned per-head bias for the typed relation between every token pair
(TokenGT-style in spirit, without node identifiers). Conditioning available
on this graph: the target type z_tau. z_do / z_pub from the plan need
action and publication metadata the graph does not carry yet.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sci_jepa.data.graph_index import N_RELS, N_TOKEN_TYPES, REL_NONE, REL_SAME_PAPER  # noqa: F401
from sci_jepa.data.instances import N_PROBE_CLASSES


class _RelBias(torch.autograd.Function):
    """w[rel]: the learned per-head bias of every token pair's relation, (B, T, T) ids -> (B, T, T, heads).
    Forward = the embedding lookup (identical values). Backward: the weight gradient as one product of a one-hot
    matrix with the incoming gradient (fp32, no TF32), instead of the embedding's sort-and-reduce of millions of
    pair positions into ~N_RELS rows, which took ~70% of the GPU time of a training step (scaling/README.md)."""

    @staticmethod
    def forward(ctx, rel, w):
        ctx.save_for_backward(rel)
        ctx.n_rows = w.shape[0]
        return F.embedding(rel, w)

    @staticmethod
    def backward(ctx, g):
        (rel,) = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1]).float()
        oh = torch.zeros(g2.shape[0], ctx.n_rows, dtype=torch.float32, device=g.device)
        oh.scatter_(1, rel.reshape(-1, 1), 1.0)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            gw = oh.t() @ g2
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return None, gw.to(g.dtype)


def rel_bias(emb, rel):
    """emb(rel) for an nn.Embedding of relation biases, with the fast backward above on CUDA."""
    if rel.is_cuda and emb.weight.requires_grad and torch.is_grad_enabled():
        return _RelBias.apply(rel, emb.weight)
    return emb(rel)


class _RelBiasHeads(torch.autograd.Function):
    """bias[b, h, i, j] = w[rel[b, i, j], h], built straight in the attention layout (B, heads, T, T) and dtype:
    one gather of 2-byte values instead of an fp32 (B, T, T, heads) lookup, a permute, and a cast copy (the
    lookup alone was ~30% of a training step on an H100). Same values. Backward as _RelBias (one-hot product,
    fp32, no TF32), with the one-hot shared by every layer that reads the same rel tensor."""

    @staticmethod
    def forward(ctx, rel, w, dtype, onehot):
        ctx.save_for_backward(onehot)
        ctx.w_dtype = w.dtype
        return w.t().to(dtype)[:, rel].transpose(0, 1)

    @staticmethod
    def backward(ctx, g):
        (onehot,) = ctx.saved_tensors
        gh = g.transpose(0, 1).reshape(g.shape[1], -1).float()        # (heads, B*T*T)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            gw = gh @ onehot                                           # (heads, N_RELS)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return None, gw.t().to(ctx.w_dtype), None, None


def rel_bias_heads(emb, rel, dtype):
    """Relation biases of every token pair as a (B, heads, T, T) attention mask in `dtype`."""
    w = emb.weight
    if rel.is_cuda and w.requires_grad and torch.is_grad_enabled():
        onehot = getattr(rel, "_relbias_onehot", None)                 # built once per batch, freed with it
        if onehot is None or onehot.shape[1] != w.shape[0]:
            onehot = torch.zeros(rel.numel(), w.shape[0], dtype=torch.float32, device=rel.device)
            onehot.scatter_(1, rel.reshape(-1, 1), 1.0)
            rel._relbias_onehot = onehot
        return _RelBiasHeads.apply(rel, w, dtype, onehot)
    return w.t().to(dtype)[:, rel].transpose(0, 1)


class RelBiasAttention(nn.Module):
    def __init__(self, d, heads, dropout, fused=False, edges_only=False):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.rel_bias = nn.Embedding(N_RELS, heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.drop = nn.Dropout(dropout)
        self.fused = fused                             # cfg.attn_impl == "sdpa": same maths, one fused kernel
        self.edges_only = edges_only                   # cfg.encoder == "gnn": attend only along graph edges
        self.capture, self.last_att = False, None      # worldview.py reads attention when capture is on

    def _mask(self, rel, pad):
        """(B, 1|T, T) True = do not attend: padding, and for the GNN every token pair without an edge
        (relation 'none'), except a token to itself."""
        m = pad[:, None, :]
        if self.edges_only:                             # every token keeps itself (padding too: an all-masked
            eye = torch.eye(rel.shape[-1], dtype=torch.bool, device=rel.device)   # row would give NaN that
            m = (m | (rel == REL_NONE)) & ~eye          # leaks through the values into real tokens)
        return m[:, None]

    def forward(self, x, rel, pad):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        if self.fused and not self.capture:
            block = torch.zeros(B, 1, T, T, dtype=q.dtype, device=x.device).masked_fill(self._mask(rel, pad),
                                                                                        float("-inf"))
            bias = rel_bias_heads(self.rel_bias, rel, q.dtype) + block
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                               dropout_p=self.drop.p if self.training else 0.0)
            return self.out(o.transpose(1, 2).reshape(B, T, -1))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att + rel_bias(self.rel_bias, rel).permute(0, 3, 1, 2)
        att = att.masked_fill(self._mask(rel, pad), float("-inf"))
        att = self.drop(att.softmax(-1))
        if self.capture:
            self.last_att = att.detach()
        return self.out((att @ v).transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d, heads, dropout, fused=False, edges_only=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RelBiasAttention(d, heads, dropout, fused, edges_only)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, rel, pad):
        x = x + self.att(self.n1(x), rel, pad)
        return x + self.ff(self.n2(x))


def _edges(rel, pad):
    """(B, T, T) 1.0 where token j is a graph neighbour of token i: any relation but 'none', no self
    loop, no padding."""
    eye = torch.eye(rel.shape[-1], dtype=torch.bool, device=rel.device)
    return ((rel != REL_NONE) & ~eye & ~pad[:, None, :] & ~pad[:, :, None]).float()


class GINEMessage(nn.Module):
    """GINE message passing (Hu et al., 2020; the encoder family of Graph-JEPA): every token receives
    the mean over its graph neighbours of (neighbour state + learned embedding of the edge's relation)
    and is updated by an MLP: h_i <- MLP((1 + eps) h_i + mean_j (h_j + e_r(ij))). Mean instead of sum
    aggregation: a paper's KUs form a ~25-token clique, and sums of that size destabilise training."""

    def __init__(self, d, dropout):
        super().__init__()
        self.edge = nn.Embedding(N_RELS, d)
        nn.init.normal_(self.edge.weight, std=0.02)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.Dropout(dropout))

    def forward(self, h, rel, pad):
        A = _edges(rel, pad).to(h.dtype)
        deg = A.sum(-1, keepdim=True).clamp_min(1.0)
        cnt = torch.stack([(A * (rel == r)).sum(-1) for r in range(N_RELS)], -1)   # (B, T, R) edges per relation
        agg = (A @ h + cnt @ self.edge.weight.to(h.dtype)) / deg
        return self.mlp((1 + self.eps) * h + agg)


class GINEBlock(nn.Module):
    """One GINE layer (pre-norm, residual). Attends to nothing: information moves one hop per layer."""

    def __init__(self, d, dropout):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.msg = GINEMessage(d, dropout)
        self.att = None                                 # no attention to capture or relation biases to show

    def forward(self, x, rel, pad):
        return x + self.msg(self.n(x), rel, pad)


class GPSBlock(nn.Module):
    """One GraphGPS layer (Rampasek et al., 2022): a local message-passing branch (GINE) and a global
    relation-biased attention branch run in parallel on the same input and are summed, then an MLP."""

    def __init__(self, d, heads, dropout, fused=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.local = GINEMessage(d, dropout)
        self.att = RelBiasAttention(d, heads, dropout, fused)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, rel, pad):
        h = self.n1(x)
        x = x + self.local(h, rel, pad) + self.att(h, rel, pad)
        return x + self.ff(self.n2(x))


ENCODERS = {"transformer": "relation-biased attention over every token pair (graph transformer)",
            "rgat": "relational graph attention: attention only along graph edges (R-GAT)",
            "gnn": "alias of rgat",
            "gine": "GINE message passing with relation embeddings (Graph-JEPA's encoder family)",
            "gps": "GraphGPS: GINE message passing + global relation-biased attention in every layer"}


def make_blocks(cfg, n):
    """n layers of the encoder family cfg.encoder (shared by the context / target encoders and the predictor)."""
    enc = getattr(cfg, "encoder", "transformer")
    fused = getattr(cfg, "attn_impl", "manual") == "sdpa"
    d = cfg.d_model
    if enc not in ENCODERS:
        raise KeyError(f"unknown encoder {enc!r}; choose from {sorted(ENCODERS)}")
    if enc == "gine":
        return nn.ModuleList([GINEBlock(d, cfg.dropout) for _ in range(n)])
    if enc == "gps":
        return nn.ModuleList([GPSBlock(d, cfg.n_heads, cfg.dropout, fused) for _ in range(n)])
    return nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout, fused, edges_only=enc in ("rgat", "gnn"))
                          for _ in range(n)])


def rwse(rel, pad, k):
    """Random-walk structural encoding (Graph-JEPA, GraphGPS): for every token the
    probability that a k-step random walk returns to it, k = 1..K, on the graph of
    typed links (role, proxy, cites, field, membership). The 'same paper' clique and
    'none' are not edges here, so the encoding describes a token's structural role
    (e.g. a claim with three pieces of evidence). -> (B, T, K)"""
    A = ((rel != REL_NONE) & (rel != REL_SAME_PAPER)).float()
    valid = (~pad).float()
    A = A * valid[:, :, None] * valid[:, None, :]
    M = A / A.sum(-1, keepdim=True).clamp_min(1.0)
    out, P = [], M
    for _ in range(k):
        out.append(torch.diagonal(P, dim1=-2, dim2=-1))
        P = P @ M
    return torch.stack(out, -1)


class GraphEncoder(nn.Module):
    def __init__(self, cfg, n_layers):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(cfg.feat_dim, d)
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.blocks = make_blocks(cfg, n_layers)
        self.norm = nn.LayerNorm(d)
        self.deg_emb = None                              # E4 census, own seed: init of the rest unchanged
        if cfg.use_degree_features:
            with torch.random.fork_rng():
                torch.manual_seed(7_919)
                self.deg_emb = nn.Embedding(65, d)
        self.k_rwse, self.rwse_proj = cfg.rwse_steps, None   # E20 structural encoding, own seed
        if self.k_rwse:
            with torch.random.fork_rng():
                torch.manual_seed(7_927)
                self.rwse_proj = nn.Linear(self.k_rwse, d)

    def forward(self, v, return_layers=False):
        h = self.inp(v["x"]) + self.type_emb(v["type"])
        h = h + v["is_mask"].unsqueeze(-1) * self.mask_emb
        if self.deg_emb is not None:                     # E4: the node census, exposed
            h = h + self.deg_emb(v["deg"].clamp(max=64))
        if self.rwse_proj is not None:                   # E20: structural role of every token
            h = h + self.rwse_proj(rwse(v["rel"], v["pad"], self.k_rwse))
        layers = []
        for blk in self.blocks:
            h = blk(h, v["rel"], v["pad"])
            if return_layers:
                layers.append(h)
        out = self.norm(h)
        return (out, layers) if return_layers else out


class Predictor(nn.Module):
    """Pred_phi(E(x), z_tau): refines the context at the masked token, conditioned
    on the target type, and reads out a d_model prediction."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.type_cond = nn.Embedding(N_TOKEN_TYPES, d)
        self.blocks = make_blocks(cfg, cfg.pred_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.pos_proj = None                             # E20: WHERE the hidden node sits (Graph-JEPA Eq. 5)
        if cfg.rwse_steps:
            with torch.random.fork_rng():
                torch.manual_seed(7_933)
                self.pos_proj = nn.Linear(cfg.rwse_steps, d)

    def forward(self, h, v, mask_pos, target_type, extra=None, target_pos_enc=None):
        """Every hidden token is conditioned on its own KU type (masked tokens keep their
        type; for the anchor this is target_type). extra = (batch idx, ctx pos) of further
        hidden motif members -> returns (anchor prediction, extra predictions)."""
        h = h + v["is_mask"].unsqueeze(-1) * self.type_cond(v["type"])
        if self.pos_proj is not None and target_pos_enc is not None:
            h = h + v["is_mask"].unsqueeze(-1) * self.pos_proj(target_pos_enc)
        for blk in self.blocks:
            h = blk(h, v["rel"], v["pad"])
        anchor = self.head(h[torch.arange(h.size(0), device=h.device), mask_pos])
        if extra is None:
            return anchor
        return anchor, self.head(h[extra[0], extra[1]])


class InverseRelationHead(nn.Module):
    """The IDM analogue: from two component representations, recover the typed
    role edge between them (or 'no edge'). The edge was hidden from the encoder,
    so the label is free supervision the encoder must preserve."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, N_PROBE_CLASSES))

    def forward(self, hi, hj):
        return self.mlp(torch.cat([hi + hj, (hi - hj).abs(), hi * hj], -1))


class PaperAModel(nn.Module):
    def __init__(self, cfg, shape_bank=None):
        super().__init__()
        self.cfg = cfg
        self.context = GraphEncoder(cfg, cfg.enc_layers)
        self.target = copy.deepcopy(self.context).requires_grad_(False)
        self.predictor = Predictor(cfg)
        self.inverse = InverseRelationHead(cfg.d_model)
        self.shapes = shape_bank

    @torch.no_grad()
    def ema(self, m):
        tp, cp = list(self.target.parameters()), [p.detach() for p in self.context.parameters()]
        torch._foreach_mul_(tp, m)                      # one fused update instead of one kernel per tensor
        torch._foreach_add_(tp, cp, alpha=1 - m)

    @torch.no_grad()
    def encode_target(self, v, pos, extra=None):
        """v: target views, rows 0..B-1 = each instance's own, then multi-target views.
        extra = (batch idx, ctx pos, target pos, target-view row)."""
        z = self.target(v)
        za = F.layer_norm(z[torch.arange(len(pos), device=z.device), pos], z.shape[-1:])
        if extra is None:
            return za
        return za, F.layer_norm(z[extra[3], extra[2]], z.shape[-1:])

    def _target_pos_enc(self, batch):
        """E20: the hidden nodes' TRUE structural positions (RWSE in the unmasked target
        view), placed at their context positions -> (B, T_ctx, K)."""
        k = self.cfg.rwse_steps
        if not k:
            return None
        c, t = batch["ctx"], batch["tgt"]
        r = rwse(t["rel"], t["pad"], k)
        enc = torch.zeros(c["x"].shape[0], c["x"].shape[1], k, device=c["x"].device)
        b = torch.arange(enc.size(0), device=enc.device)
        enc[b, batch["mask_pos"]] = r[b, batch["target_pos"]]
        ex = batch.get("extra")
        if ex is not None and len(ex[0]):
            enc[ex[0], ex[1]] = r[ex[3], ex[2]]
        return enc

    def predict(self, batch, with_extra=False):
        h = self.context(batch["ctx"])
        pe = self._target_pos_enc(batch)
        ex = batch.get("extra")
        if with_extra and ex is not None and len(ex[0]):
            a, e = self.predictor(h, batch["ctx"], batch["mask_pos"], batch["target_type"], (ex[0], ex[1]), pe)
            return (a, e), h
        return self.predictor(h, batch["ctx"], batch["mask_pos"], batch["target_type"], target_pos_enc=pe), h

    def forward(self, batch):
        ex = batch.get("extra")
        has_extra = ex is not None and len(ex[0]) > 0
        pred, h = self.predict(batch, with_extra=has_extra)
        if has_extra:
            (pred, pred_x), (tgt, tgt_x) = pred, self.encode_target(batch["tgt"], batch["target_pos"], ex)
            out = {"pred": pred, "h": h, "target": tgt, "pred_extra": pred_x, "target_extra": tgt_x}
        else:
            out = {"pred": pred, "h": h, "target": self.encode_target(batch["tgt"], batch["target_pos"])}
        b, i, j, lab = batch["probe"]
        if len(b):
            out["inv_logits"] = self.inverse(h[b, i], h[b, j])
            out["inv_labels"] = lab
        return out


def to_device(batch, dev):
    if isinstance(batch, dict):
        out = {k: to_device(v, dev) for k, v in batch.items()}
        if torch.is_tensor(out.get("rel")) and out["rel"].dtype == torch.uint8:
            out["rel"] = out["rel"].long()              # collated as uint8, widened after the copy
        return out
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return type(batch)(t.to(dev, non_blocking=True) for t in batch)
    return batch.to(dev, non_blocking=True) if torch.is_tensor(batch) else batch   # async from pinned memory
