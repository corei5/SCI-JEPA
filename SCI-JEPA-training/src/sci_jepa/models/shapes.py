"""
Symbolic constraints as a differentiable training signal (E1-E3).

Spec format (JSON) -- the same format the real ORKG-derived vocabulary must be
exported to:

  {"shapes": [
     {"name": "ClaimShape", "target_type": "claim",
      "slots": [{"name": "challenged", "kind": "class", "n_classes": 2},
                {"name": "field", "kind": "class", "n_classes": "n_fields"}],
      "constraints": [
         {"kind": "class_range", "slot": "challenged", "allowed": "truth", "tight": true},
         {"kind": "class_range", "slot": "field", "allowed": "truth",
          "context_determined": true, "tight": false},
         {"kind": "datatype", "slot": "challenged", "tight": false}]}]}

`tight` is the A2 classification: a constraint is tight when its admissible
set given the context is small relative to the content it describes AND is
not computable from the context. `audit_tightness` estimates both from data;
the spec's flag is what E1 uses, the audit is what justifies it.

PLACEHOLDER. placeholder_shapes.json uses slots computable from the KUs
(real counter-evidence present, evidence cites a figure, quantitative text,
paper type, field). It exists to exercise the
machinery and to demonstrate the A2 corollary (context-determined field and
validity constraints should not separate the degenerate branch). It is not
the ORKG vocabulary and results on it are not Theorem A2 evidence.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sci_jepa.data.graph_index import COMP_TYPES, REL_ID
from sci_jepa.data.ku_schema import PAPER_TYPES

PLACEHOLDER_SPEC = os.path.join(os.path.dirname(__file__), "placeholder_shapes.json")


# ----------------------------------------------------------------------
# t-norms
# ----------------------------------------------------------------------
def tnorm(degrees, kind):
    """degrees: (B, K) truth degrees in [0, 1] -> (B,) conjunction."""
    if kind == "product":
        return degrees.clamp_min(1e-12).log().sum(-1).exp()
    if kind == "lukasiewicz":
        return (degrees.sum(-1) - (degrees.size(-1) - 1)).clamp_min(0.0)
    if kind == "godel":
        return degrees.min(-1).values
    raise ValueError(kind)


# ----------------------------------------------------------------------
# slot labels from the graph
# ----------------------------------------------------------------------
class SlotLabels:
    """Ground-truth slot values per KU, derived from the KU graph.

    Not used, because constant on this corpus (zero content entropy, so
    they can never be tight): a claim's count of supporting evidence and of
    implications (exactly 1 each, by the extraction template)."""

    def __init__(self, gi):
        n = gi.n_comps
        cnt = lambda rel: np.bincount(gi.role_src[gi.role_rel == REL_ID[rel]], minlength=n)
        ev_fig = np.zeros(n, dtype=np.int64)              # claim -> any child evidence cites a figure
        for rel in ("supported_by", "challenged_by"):
            m = gi.role_rel == REL_ID[rel]
            np.maximum.at(ev_fig, gi.role_src[m], gi.comp_cites_figure[gi.role_dst[m]].astype(np.int64))
        self.values = {
            "challenged": gi.comp_challenged.astype(np.int64),     # real counter-evidence exists
            "evidence_cites_figure": ev_fig,
            "quantitative": gi.comp_quantitative.astype(np.int64),
            "n_grounded_claims": cnt("grounds"),                   # result -> claims it grounds
            "paper_type": gi.paper_type[gi.comp_paper],
            "field": gi.paper_field[gi.comp_paper],
        }
        self.n_classes = {"n_fields": int(gi.field_x.shape[0]), "n_paper_types": len(PAPER_TYPES)}

    def get(self, slot, comps):
        return self.values[slot][comps]


# ----------------------------------------------------------------------
class ShapeBank(nn.Module):
    """Soft decoders + constraint evaluation for every shape in the spec."""

    def __init__(self, spec_path, labels, d_model, cfg):
        super().__init__()
        spec = json.load(open(spec_path or PLACEHOLDER_SPEC))
        self.labels, self.cfg = labels, cfg
        self.shapes = {}
        heads = {}
        for sh in spec["shapes"]:
            t = COMP_TYPES.index(sh["target_type"])
            cons = [c for c in sh["constraints"] if _keep(c, cfg.shape_subset)]
            if not cons:
                continue
            if cfg.shape_depth:                          # E2: manipulate conjunction depth
                cons = [cons[i % len(cons)] for i in range(cfg.shape_depth)]
            slots = [s for s in sh["slots"] if any(c["slot"] == s["name"] for c in cons)]
            n_slots = len(slots)
            d_slot = d_model // max(n_slots, 1) if cfg.shape_factored else d_model
            for k, s in enumerate(slots):
                nc = s.get("n_classes", 1)
                out = (labels.n_classes[nc] if isinstance(nc, str) else int(nc)) if s["kind"] == "class" else 1
                heads[f"{sh['name']}__{s['name']}"] = nn.Sequential(
                    nn.LayerNorm(d_slot), nn.Linear(d_slot, out))
                s["_sl"] = slice(k * d_slot, (k + 1) * d_slot) if cfg.shape_factored else slice(0, d_model)
            self.shapes[t] = {"name": sh["name"], "slots": {s["name"]: s for s in slots}, "cons": cons}
        self.heads = nn.ModuleDict(heads)

    # ------------------------------------------------------------------
    def degrees(self, pred, comps, ttype):
        """Per-sample conjunct truth degrees. Returns {type: (rows, (B_t, K))}."""
        out = {}
        for t, sh in self.shapes.items():
            rows = (ttype == t).nonzero(as_tuple=True)[0]
            if not len(rows):
                continue
            cid = comps[rows].cpu().numpy()
            decoded = {}
            for name, s in sh["slots"].items():
                h = self.heads[f"{sh['name']}__{name}"](pred[rows][:, s["_sl"]])
                decoded[name] = (h.softmax(-1) if s["kind"] == "class" else F.softplus(h.squeeze(-1)))
            degs = [self._degree(c, sh["slots"][c["slot"]], decoded[c["slot"]], cid, pred.device)
                    for c in sh["cons"]]
            out[t] = (rows, torch.stack(degs, -1))
        return out

    def _degree(self, c, slot, val, cid, dev, s=0.15):
        kind = c["kind"]
        if kind == "datatype":                          # validity only: always satisfiable
            return torch.ones(val.shape[0], device=dev)
        truth = torch.as_tensor(self.labels.get(c["slot"], cid), device=dev)
        if kind == "class_range":
            return val.gather(1, truth.view(-1, 1)).squeeze(1)
        if kind == "cardinality":
            return torch.sigmoid((val - (c.get("min", 0) - 0.5)) / s)
        if kind == "value_range":
            w = float(c.get("width", 0)) if c.get("range", "exact") != "exact" else 0.0
            lo, hi = truth.float() - w - 0.5, truth.float() + w + 0.5
            return torch.sigmoid((val - lo) / s) * torch.sigmoid((hi - val) / s)
        raise ValueError(kind)

    def loss(self, pred, comps, ttype):
        parts, n = 0.0, 0
        for t, (rows, d) in self.degrees(pred, comps, ttype).items():
            parts = parts + (1.0 - tnorm(d, self.cfg.tnorm)).sum(); n += len(rows)
        return parts / max(n, 1)

    @torch.no_grad()
    def satisfaction(self, pred, comps, ttype):
        vals = []
        for t, (rows, d) in self.degrees(pred, comps, ttype).items():
            vals.append(tnorm(d, self.cfg.tnorm))
        return torch.cat(vals) if vals else torch.zeros(0)


def _keep(c, subset):
    if subset == "all":
        return True
    return bool(c.get("tight")) == (subset == "tight")


# ----------------------------------------------------------------------
def audit_tightness(spec_path, labels, gi):
    """Theorem A2 audit, no training: per constraint, content entropy of the slot
    vs residual entropy of the admissible set, and whether the context fixes it."""
    spec = json.load(open(spec_path or PLACEHOLDER_SPEC))
    rows = []
    for sh in spec["shapes"]:
        t = COMP_TYPES.index(sh["target_type"])
        comps = np.nonzero(gi.comp_type == t)[0]
        for c in sh["constraints"]:
            v = labels.get(c["slot"], comps)
            _, counts = np.unique(v, return_counts=True)
            p = counts / counts.sum()
            h_content = max(0.0, float(-(p * np.log2(p)).sum()))
            if c["kind"] in ("datatype", "cardinality"):
                h_resid = h_content                      # admissible set = everything valid
            elif c["kind"] == "class_range" or c.get("range", "exact") == "exact":
                h_resid = 0.0                            # singleton admissible set
            else:
                h_resid = math.log2(2 * float(c.get("width", 0)) + 1)
            ctx = bool(c.get("context_determined", False))
            tight = (not ctx) and h_resid < h_content and c["kind"] not in ("datatype", "cardinality")
            rows.append({"shape": sh["name"], "kind": c["kind"], "slot": c["slot"],
                         "H_content_bits": round(h_content, 4), "H_residual_bits": round(h_resid, 4),
                         "context_determined": ctx, "tight_by_audit": tight,
                         "tight_in_spec": bool(c.get("tight")),
                         "agrees": tight == bool(c.get("tight"))})
    return rows
