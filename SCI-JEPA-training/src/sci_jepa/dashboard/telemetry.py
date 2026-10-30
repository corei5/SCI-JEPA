"""
Live telemetry for the Paper A dashboard.

Each run writes <root>/runs/<name>/live.json (atomically, so a reader never sees
half a file) and registers itself in <root>/runs.json. The dashboard
(dashboard.html, served by serve.py) polls both every few seconds, so curves
update while training runs. With SCIJEPA_PUSH_URL and SCIJEPA_PUSH_TOKEN set, the
folder is also pushed to the hosted dashboard while the run trains (push.py).
"""
import json
import math
import os
import time

import numpy as np

from sci_jepa.config import resolve_device
from sci_jepa.data.ku_schema import (COMP_TYPES, N_RELS, REL_CITES, REL_DO, REL_ID, REL_SAME_FIELD,
                                     REL_SAME_SUBFIELD, ROLE_RELS)

_REL_NAMES = {0: "none", 1: "same_paper", 2: "member"}
_REL_NAMES.update({v: k for k, v in REL_ID.items()})
_REL_NAMES.update({REL_CITES: "cites", REL_SAME_FIELD: "same_field", REL_SAME_SUBFIELD: "same_subfield",
                   REL_DO: "do (action)"})
REL_NAMES = [_REL_NAMES.get(i, str(i)) for i in range(N_RELS)]


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, default=str)
    os.replace(tmp, path)


class LiveRun:
    def __init__(self, root, name, cfg, gi, extra=None):
        self.root, self.name = root, name
        self.dir = os.path.join(root, "runs", name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "live.json")
        S = gi.summary()
        keep = ("n_subgraphs", "d_model", "enc_layers", "pred_layers", "batch_size", "steps", "lr",
                "lambda_inv", "lambda_reg", "reg_grouping", "lambda_contrast", "lambda_shacl", "tnorm",
                "shape_depth", "mask_policy", "split_mode", "time_T", "max_papers", "target_types", "device",
                "mask_schedule", "replay_frac", "epochs", "embedder", "embed_model", "embed_max_tokens",
                "feat_dim", "audit_tau", "proxy_tau", "ku_mode", "baseline", "motif_p", "rwse_steps",
                "wm_p_action", "task_mask_p", "pool_mode", "minimal_pairs", "encoder", "patience", "select_best",
                "select_metric", "amp")
        self.state = {"name": name, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "updated": None, "steps_total": cfg.steps,
                      "config": {**{k: getattr(cfg, k) for k in keep}, "device": resolve_device(cfg.device)},
                      "index": {k: S[k] for k in ("papers", "kus", "kus_per_paper", "per_type", "splits",
                                                    "paper_edges_cites", "paper_edges_proxy", "fields")},
                      "rel_names": REL_NAMES, "comp_types": COMP_TYPES, "role_rels": ROLE_RELS,
                      "history": [], "evals": [], "relbias": None, "final": None, **(extra or {})}
        self._register()
        self._write()
        from sci_jepa.dashboard.push import start_autopush
        start_autopush(root)                             # hosted dashboard, if SCIJEPA_PUSH_URL/TOKEN are set

    def _register(self):
        idx = os.path.join(self.root, "runs.json")
        runs = []
        if os.path.exists(idx):
            try:
                runs = json.load(open(idx))
            except (json.JSONDecodeError, OSError):
                runs = []
        runs = [r for r in runs if r.get("name") != self.name]
        runs.append({"name": self.name, "started": self.state["started"], "path": f"runs/{self.name}/live.json"})
        _atomic_dump(runs, idx)

    def _write(self):
        self.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_dump(self.state, self.path)

    # ------------------------------------------------------------------
    def step(self, rec):
        self.state["history"].append(rec)
        self._write()

    def relbias(self, model):
        """Learned attention bias per relation (mean over heads) for each encoder layer:
        which relations the encoder learned to attend along."""
        mats = [blk.att.rel_bias.weight.detach().float().mean(1).cpu().numpy().tolist()
                for blk in model.context.blocks if getattr(blk, "att", None) is not None]
        self.state["relbias"] = mats or None             # GINE has no attention: relations are edge embeddings

    def eval(self, step, m, epoch=None):
        keep = {"step": step, "epoch": epoch,
                "bal_MRR": m["retrieval_balanced"]["MRR"], "bal_chance": m["retrieval_balanced"].get("chance_MRR"),
                "bal_H1": m["retrieval_balanced"]["Hits@1"], "bal_H10": m["retrieval_balanced"]["Hits@10"],
                "acc_sim": (m.get("similarity") or {}).get("acc"),
                "acc_sim_top1": (m.get("similarity") or {}).get("acc_top1"),
                "false_accept": (m.get("similarity") or {}).get("false_accept"),
                "std_MRR": m["retrieval_standard"]["MRR"], "std_chance": m["chance_MRR_standard"],
                "chi": m["target_space"]["chi"], "chi_raw": m["target_space"]["chi_raw"],
                "eff_rank": m["target_space"]["eff_rank"], "ctx_eff_rank": m["context_space"]["eff_rank"],
                "probe_acc": m["partition_probe"].get("probe_acc"),
                "probe_chance": m["partition_probe"].get("probe_chance"),
                "between_var": m["partition_probe"].get("between_subgraph_var_share"),
                "inverse_acc": m["inverse_acc"], "rel_sens": m["relation_sensitivity"],
                "per_type": {k: v["MRR_bal"] for k, v in m.get("per_type", {}).items()},
                "per_type_chance": {k: v["chance_bal"] for k, v in m.get("per_type", {}).items()}}
        self.state["evals"].append(keep)
        self._write()

    def neurons(self, step, epoch, rep, final=False):
        """Neuron report: a compact per-layer summary on the timeline, and the full
        per-neuron table for the latest report."""
        if rep is None:
            return
        self.state.setdefault("neuron_timeline", []).append(
            {"step": step, "epoch": epoch,
             "layers": [{"name": l["name"], "counts": l["counts"], "share_paper_var": l["share_paper_var"],
                         "mean_o_type": l["mean_o_type"]} for l in rep["layers"]],
             "pred_learned": rep["predictor"]["learned"], "pred_n": rep["predictor"]["n"],
             "pred_mean_r": rep["predictor"]["mean_r_target"]})
        self.state["neurons_latest"] = {**rep, "step": step, "epoch": epoch, "final": final}
        self._write()

    def worldview(self, step, epoch, examples, final=False):
        """Explained example subgraphs (dashboard: World model tab)."""
        self.state["worldview"] = {"step": step, "epoch": epoch, "final": final, "examples": examples}
        self._write()

    def epoch_end(self, rec):
        """End-of-epoch validation row (dashboard: Epochs tab, All runs epoch matrix)."""
        self.state.setdefault("epoch_results", []).append(rec)
        self._write()

    def finish(self, m, status="done"):
        drop = ("history", "epoch_results")
        self.state["final"] = {k: v for k, v in m.items() if k not in drop and k != "config"}
        self.state["status"] = status
        self._write()

    def restore(self, step):
        """A resumed run (train_one with a checkpoint): keep the curves already on disk up to `step`, so the
        dashboard shows one continuous run instead of starting over."""
        try:
            old = json.load(open(self.path))
        except (OSError, json.JSONDecodeError):
            return
        for key in ("history", "evals", "epoch_results", "neuron_timeline"):
            rows = [r for r in old.get(key) or [] if (r.get("step", -1) if isinstance(r, dict) else -1) <= step]
            if rows:
                self.state[key] = rows
        for key in ("neurons_latest", "worldview", "relbias", "started"):
            if old.get(key) is not None:
                self.state[key] = old[key]
        self.state.setdefault("resumed", []).append({"step": step, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "job": os.environ.get("SLURM_JOB_ID")})
        self._write()

    def fail(self, err):
        self.state["status"] = f"failed: {err}"
        self._write()
