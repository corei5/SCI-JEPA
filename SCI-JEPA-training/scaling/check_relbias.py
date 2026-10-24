"""
The fast relation-bias backward (models/model.py: _RelBias) gives the same values and gradients as nn.Embedding,
and how much faster it is. Real batch shape: 64 x 189 x 189 token pairs, 8 heads.

    python scaling/check_relbias.py
"""
import time

import torch

from sci_jepa.data.ku_schema import N_RELS
from sci_jepa.models.model import rel_bias

torch.manual_seed(0)
dev = "cuda"
emb = torch.nn.Embedding(N_RELS, 8).to(dev)
torch.nn.init.normal_(emb.weight)
rel = torch.randint(0, N_RELS, (64, 189, 189), device=dev)
g = torch.randn(64, 189, 189, 8, device=dev)


def grads(fn):
    emb.weight.grad = None
    out = fn()
    out.backward(g)
    return out.detach(), emb.weight.grad.clone()


ref_out, ref_g = grads(lambda: emb(rel))
new_out, new_g = grads(lambda: rel_bias(emb, rel))
print(f"forward identical: {torch.equal(ref_out, new_out)}")
print(f"gradient max abs diff {(ref_g - new_g).abs().max().item():.3e}, max rel diff "
      f"{((ref_g - new_g).abs() / ref_g.abs().clamp_min(1e-6)).max().item():.3e}")
for name, fn in (("nn.Embedding", lambda: emb(rel)), ("one-hot backward", lambda: rel_bias(emb, rel))):
    for _ in range(3):
        grads(fn)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        grads(fn)
    torch.cuda.synchronize()
    print(f"{name:18s} forward + backward {(time.time() - t) / 20 * 1e3:7.2f} ms per layer")
