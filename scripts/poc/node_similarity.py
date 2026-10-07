"""Node-type x node-type cosine similarity inside paper intragraphs vs across papers.

Samples N papers from the KG's cached embeddings and reduces each paper to one vector
per node type. A paper has one Paper, Method and Result but ~4 Evidence/Claim/
Implication; those are averaged into one vector per type (e.g. the paper's mean
Evidence). Cell [A, B] is then the cosine between the paper's A and B vectors, so the
intragraph diagonal is 1 by construction and is kept as a reference.

Baseline: each sampled paper is paired with a different sampled paper, and [A, B]
compares the A vector of one with the B vector of the other (averaged over both
directions, so the matrix is symmetric like the intragraph one). Here the diagonal is
informative: e.g. [paper, paper] is how alike two unrelated papers' Paper nodes are.
If nodes of a paper are more alike than nodes of unrelated papers, intra - cross is
positive.

Writes four separate heatmaps (intra mean, intra std, cross mean, intra - cross) and
the raw matrices. The Field node is left out: it is a shared hub, not per paper.

Run from the repo root, after the graph cache exists (see graph.sbatch):
    python scripts/poc/node_similarity.py
    python scripts/poc/node_similarity.py --n 100 --seed 1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm

from train_kg import CACHE_DIR, EMBED_MODEL, KG_DIR
from train_kg_graph import NODE_TYPES, PER_CLAIM, PER_PAPER

OUT_DIR = "scripts/poc/plots/node_similarity"


def paper_vectors(data: dict) -> torch.Tensor:
    """Cache -> [papers, T, D]: one unit vector per node type per paper.

    Per-claim types (Evidence/Claim/Implication) are averaged over the paper's claims
    first, then renormalised so a dot product is a cosine.
    """
    n_papers = data["paper"].shape[0]
    owner = data["sample_paper"]
    counts = torch.bincount(owner, minlength=n_papers).clamp(min=1).unsqueeze(1).float()
    per_type = []
    for k in NODE_TYPES:
        v = F.normalize(data[k].float(), dim=-1)
        if k in PER_CLAIM:
            v = torch.zeros(n_papers, v.shape[1]).index_add_(0, owner, v) / counts
        else:
            assert k in PER_PAPER
        per_type.append(F.normalize(v, dim=-1))
    return torch.stack(per_type, dim=1)


def intra_matrix(v: torch.Tensor) -> np.ndarray:
    """[T, D] vectors of one paper -> [T, T] cosines; the diagonal is 1."""
    return (v @ v.T).numpy()


def cross_matrix(v: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    """[T, T] cosines between the vectors of two different papers, symmetrised."""
    m = (v @ w.T).numpy()
    return (m + m.T) / 2


def heatmap(m: np.ndarray, title: str, path: Path, cmap: str, vmin=None, vmax=None, norm=None, fmt="{:.3f}"):
    labels = list(NODE_TYPES)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(m), cmap=cm, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # White gaps between cells.
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isnan(m[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#555555")
                continue
            r, g, b, _ = cm(im.norm(m[i, j]))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=9, color="white" if dark else "#1a1a1a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-dir", default=KG_DIR)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--cache", default=None, help="default scripts/poc/cache/<kg>_<model>_graph.pt")
    ap.add_argument("--n", type=int, default=100, help="number of sampled papers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    slug = args.embed_model.split("/")[-1].lower()
    cache = Path(args.cache or f"{CACHE_DIR}/{Path(args.kg_dir).name}_{slug}_graph.pt")
    data = torch.load(cache)
    papers = paper_vectors(data)

    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(papers), size=args.n, replace=False)
    # picked is already in random order, so shifting by one pairs each paper with a
    # random *other* sampled paper and never with itself.
    partner = np.roll(picked, 1)

    intra = np.stack([intra_matrix(papers[p]) for p in picked])
    cross = np.stack([cross_matrix(papers[p], papers[q]) for p, q in zip(picked, partner)])

    intra_mean, intra_std = intra.mean(0), intra.std(0, ddof=1)
    cross_mean = cross.mean(0)
    diff = intra_mean - cross_mean
    n_claims = torch.bincount(data["sample_paper"], minlength=len(papers))[picked]
    print(f"{cache.name}: {args.n} papers (seed {args.seed}), claims per paper "
          f"{n_claims.min()}-{n_claims.max()} (mean {n_claims.float().mean():.1f}), averaged per type")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.n} papers, seed {args.seed}, {slug}\nevidence/claim/implication averaged per paper"
    # The two means share one scale so their colours compare directly.
    lo = min(intra_mean.min(), cross_mean.min())
    hi = max(intra_mean.max(), cross_mean.max())
    heatmap(intra_mean, f"Intragraph cosine similarity, mean\n({tag})", out / "intra_mean.png", "Blues", lo, hi)
    heatmap(intra_std, f"Intragraph cosine similarity, std\n({tag})", out / "intra_std.png", "Purples", 0, intra_std.max())
    heatmap(cross_mean, f"Cross-paper cosine similarity, mean\n({tag})", out / "cross_mean.png", "Blues", lo, hi)
    lim = np.abs(diff).max()
    heatmap(
        diff, f"Intragraph minus cross-paper mean\n({tag})", out / "intra_minus_cross.png", "RdBu_r",
        norm=TwoSlopeNorm(0, -lim, lim), fmt="{:+.3f}",
    )

    np.savez(
        out / "matrices.npz", node_types=np.array(NODE_TYPES), papers=picked, partners=partner,
        intra=intra, cross=cross, intra_mean=intra_mean, intra_std=intra_std, cross_mean=cross_mean,
    )
    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "cache": str(cache), "n": args.n, "seed": args.seed, "node_types": list(NODE_TYPES),
                "papers": picked.tolist(), "partners": partner.tolist(),
                **{k: v.round(4).tolist() for k, v in
                   {"intra_mean": intra_mean, "intra_std": intra_std, "cross_mean": cross_mean, "intra_minus_cross": diff}.items()},
            },
            f, indent=1,
        )
    print(f"saved {out / 'matrices.npz'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
