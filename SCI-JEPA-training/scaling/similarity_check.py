"""
Is "cosine >= 0.80 to the true KU's target vector" a meaningful correctness rule for trained JEPA models? Compares the
raw cosine with a centred cosine (the bank's mean vector removed) for the true KU and for the best distractor.

    python scaling/similarity_check.py <model dir> [<model dir> ...]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from sci_jepa.config import PaperAConfig
from sci_jepa.data.graph_index import get_index
from sci_jepa.data.instances import Sampler, collate
from sci_jepa.evaluation.diagnostics import encode_papers, hard_pool
from sci_jepa.models.model import PaperAModel, to_device
import json, os

torch.set_grad_enabled(False)
for d in sys.argv[1:]:
    raw = json.load(open(os.path.join(d, "config.json")))
    base = PaperAConfig()
    over = {k: (tuple(v) if isinstance(getattr(base, k), tuple) and isinstance(v, list) else v)
            for k, v in raw.items() if hasattr(base, k) and k not in ("embedder", "device")}
    cfg = base.override(**{**over, "device": "cpu", "amp": "off", "batch_workers": 0})
    gi = get_index(cfg)
    model = PaperAModel(cfg); model.load_state_dict(load_file(os.path.join(d, "best.safetensors")), strict=False); model.eval()
    papers = np.asarray(gi.papers("test"))
    ids, H = encode_papers(model, gi, papers, cfg, "cpu", "target")
    H = H.float(); mu = H.mean(0, keepdim=True)
    pos = {int(c): k for k, c in enumerate(ids.tolist())}
    ctype, bank_paper = gi.comp_type[ids.numpy()], gi.comp_paper[ids.numpy()]
    s = Sampler(gi, cfg, "test", 1234); rng = np.random.default_rng(0)
    rows = []
    for _ in range(8):
        insts = [i for i in s.sample(64) if i["target_comp"] in pos]
        pred, _ = model.predict(to_device(collate(insts), "cpu")); pred = pred.float()
        for b, inst in enumerate(insts):
            g = pos[int(inst["target_comp"])]
            same = np.nonzero(ctype == inst["target_type"])[0]
            others = hard_pool(gi, int(inst["papers"][0]), same, bank_paper, cfg.balanced_k, rng, mode=cfg.pool_mode)
            P = H[[g] + list(others)]
            raw_c = F.cosine_similarity(pred[b:b + 1], P)
            cen_c = F.cosine_similarity(pred[b:b + 1] - mu, P - mu)
            rows.append((raw_c[0].item(), raw_c[1:].max().item(), cen_c[0].item(), cen_c[1:].max().item(),
                         int(raw_c.argmax() == 0)))
    r = np.array(rows)
    print(f"== {os.path.basename(d)}: {len(r)} test queries, top-1 correct {r[:, 4].mean():.1%}")
    for name, a, b in (("raw cosine", 0, 1), ("centred cosine", 2, 3)):
        print(f"  {name:15s} true KU: median {np.median(r[:, a]):.3f}, >= 0.80 {np.mean(r[:, a] >= 0.8):.1%} | "
              f"best distractor: median {np.median(r[:, b]):.3f}, >= 0.80 {np.mean(r[:, b] >= 0.8):.1%}")
