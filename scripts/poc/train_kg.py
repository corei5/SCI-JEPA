"""JEPA prototype on the materials-science / chem-eng KG (next step after train.py).

One sample per claim: the model sees the claim's supporting Evidence and its
paper's Result, and predicts the *embedding* of the Claim.

    x      = (evidence_i, result of its paper)   [BS, 2, E]   frozen text-embedding vectors
    target = claim_i                             [BS, E]

    context encoder  f_theta : [BS, 2, E] -> [BS, 2, D]
    predictor        g_phi   : [BS, 2, D] -> [BS, D]
    target encoder   f_xi    : [BS, E]    -> [BS, D]      (EMA of f_theta, no grad)
    loss = MSE(unit(g_phi(f_theta(x))), unit(stopgrad(f_xi(claim))))    unit(v) = v / ||v||

Both embeddings are scaled to length 1 before the MSE, so the loss only sees
their direction; embeddings growing in size can no longer raise it.

Node texts are embedded by the default Qwen3-Embedding-0.6B (E=1024, 32k-token
context, so no Result is truncated). Any --embed-model must read at least
MIN_TOKENS=512 tokens; bge-small-en-v1.5 (E=384, 512 tokens) also qualifies.

Papers, not claims, are split into train/val, so a Result never appears on both
sides. Validation reports claim retrieval as a kNN search: for each held-out
sample, rank all held-out claims by closeness to the prediction, under both
cosine similarity and L2 distance (R@1, R@10, MRR). The "evidence only"
baseline does the same search with the raw evidence embedding, untrained.

Run from the repo root (the first run embeds ~60k texts and caches them):
    python scripts/poc/train_kg.py
    python scripts/poc/train_kg.py --drop-result      # ablation: is the Result used?
    python scripts/poc/train_kg.py --embed-model BAAI/bge-small-en-v1.5
    python scripts/poc/train_kg.py --patience 10      # stop once val retrieval stops improving
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KG_DIR = "scripts/dataset_creation/data/kg_materials_chemeng"
CACHE_DIR = "scripts/poc/cache"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # 32k-token context, 1024-d
MIN_TOKENS = 512  # embedding models must read at least this many tokens
METRICS = ("cosine", "l2")
# What early stopping may watch, and which direction counts as better. Retrieval is the
# default: in this setup the MSE often rises while ranking still improves, so the loss is
# the wrong thing to stop on.
MONITORS = {"cosine_MRR": "max", "cosine_R@1": "max", "cosine_R@10": "max", "val_loss": "min"}
# One row per run for --results, so a sweep can be tabulated without parsing stdout.
RESULT_COLUMNS = (
    "embed", "batch", "hidden", "dropout", "lr", "seed",
    "best_epoch", "epochs", "R@1", "R@10", "MRR", "val_loss",
    # The seed also picks the paper split, so the baseline moves with it and has to
    # travel alongside the run's own numbers to stay comparable.
    "base_R@1", "base_R@10", "base_MRR",
)


def load_samples(kg_dir: str):
    """subgraphs.jsonl -> per-paper Result texts and per-claim (paper index, Evidence, Claim) texts."""
    results, sample_paper, evidence, claims = [], [], [], []
    with open(Path(kg_dir) / "subgraphs.jsonl") as f:
        for line in f:
            paper = json.loads(line)
            p = len(results)
            results.append(paper["result"]["text"])
            for c in paper["claims"]:
                if not c["evidence"]:  # none in this KG, but a claim needs evidence to be a sample
                    continue
                sample_paper.append(p)
                evidence.append(c["evidence"][0]["text"])  # exactly one evidence per claim
                claims.append(c["text"])
    return results, sample_paper, evidence, claims


def build_cache(kg_dir: str, path: Path, model_name: str, device: torch.device) -> dict:
    """Embed every node text once with a frozen sentence-embedding model and save the vectors."""
    from sentence_transformers import SentenceTransformer

    results, sample_paper, evidence, claims = load_samples(kg_dir)
    model = SentenceTransformer(model_name, device=str(device))
    max_len = model.max_seq_length
    if max_len < MIN_TOKENS:
        raise ValueError(f"{model_name} reads only {max_len} tokens; need at least {MIN_TOKENS}")
    print(f"embedding with {model_name}  (max {max_len} tokens, {model.get_sentence_embedding_dimension()}-d)")

    cache = {"sample_paper": torch.tensor(sample_paper), "model": model_name}
    for name, texts in (("result", results), ("evidence", evidence), ("claim", claims)):
        n_tokens = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
        cut = sum(n > max_len for n in n_tokens)
        print(f"{name:8s} {len(texts):6d} texts, {cut / len(texts):.1%} truncated at {max_len} tokens")
        # Unit length, as MiniLM already outputs, so L2 and cosine rank raw vectors identically.
        cache[name] = model.encode(
            texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"saved {path}")
    return cache


def mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Two-layer MLP, Kaiming-uniform weights and zero biases.

    Dropout sits between the activation and the output layer, so it perturbs the
    hidden features and never the embedding this returns.
    """
    net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )
    for layer in net:
        if isinstance(layer, nn.Linear):
            # GELU has no gain of its own; ReLU's sqrt(2) is the usual stand-in.
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    return net


class Encoder(nn.Module):
    """Node encoder shared across node types: [..., E] -> [..., D]."""

    def __init__(self, in_dim: int, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = mlp(in_dim, hidden, dim, dropout)

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Context embeddings of (Evidence, Result) -> predicted Claim embedding."""

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = mlp(2 * dim, hidden, dim, dropout)

    def forward(self, ctx):  # [BS, 2, D] -> [BS, D]
        return self.net(ctx.flatten(1))


def set_seed(seed: int) -> None:
    """Seed every generator this script can reach, and pin cuDNN to deterministic kernels.

    torch.manual_seed already covers CUDA devices, so manual_seed_all is belt-and-braces;
    it is spelled out because it is the one people look for. random and numpy are seeded
    for the libraries underneath (sentence-transformers, matplotlib), not for this file.
    The cuDNN flags do nothing for an MLP -- there is no convolution here -- but they cost
    nothing and stay correct if the model ever grows one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unit_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MSE after scaling both to length 1 along the last dim: (2 - 2*cos) / D, blind to vector size."""
    return F.mse_loss(F.normalize(a, dim=-1), F.normalize(b, dim=-1))


@torch.no_grad()
def retrieval(queries: torch.Tensor, keys: torch.Tensor, metric: str = "cosine") -> dict:
    """kNN search: queries[i] should retrieve keys[i] among all keys. 1 = best rank.

    metric="cosine" ranks by cosine similarity (higher is closer);
    metric="l2" ranks by Euclidean distance (lower is closer).
    """
    if metric == "cosine":
        score = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T
    elif metric == "l2":
        score = -torch.cdist(queries, keys)  # negated so higher is closer for both metrics
    else:
        raise ValueError(f"unknown metric {metric!r}")
    rank = (score > score.diagonal().unsqueeze(1)).sum(dim=1) + 1
    return {
        "R@1": (rank <= 1).float().mean().item(),
        "R@10": (rank <= 10).float().mean().item(),
        "MRR": (1.0 / rank.float()).mean().item(),
    }


@torch.no_grad()
def evaluate(context_enc, predictor, target_enc, x, claim) -> dict:
    ctx = context_enc(x)  # [N, 2, D]
    pred = predictor(ctx)
    tgt = target_enc(claim)
    out = {
        "val_loss": unit_mse(pred, tgt).item(),
        # Spread of target embeddings across samples; ~0 means collapse.
        "tgt_std": tgt.std(dim=0).mean().item(),
    }
    for metric in METRICS:
        out.update({f"{metric}_{k}": v for k, v in retrieval(pred, tgt, metric).items()})
    return out


def plot(history: list, baseline: dict, path: str):
    """Side by side: the training loss and cosine kNN retrieval vs the evidence-only baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3de"
    blue, orange = "#2a78d6", "#eb6834"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#fcfcfb")
    ax_loss, ax_cos = axes
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_xlabel("epoch", color=muted)

    ax_loss.plot(epochs, [h["train_loss"] for h in history], color=blue, linewidth=2, label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], color=orange, linewidth=2, label="val")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Unit-length MSE(predictor, target encoder)  [training loss]", color=ink, loc="left", fontsize=11)
    ax_loss.set_ylabel("MSE (log scale)", color=muted)
    ax_loss.legend(frameon=False, labelcolor=ink, fontsize=9)

    for key, color in (("R@1", blue), ("R@10", orange)):
        ax_cos.plot(epochs, [h[f"cosine_{key}"] for h in history], color=color, linewidth=2, label=f"JEPA {key}")
        ax_cos.axhline(baseline[f"cosine_{key}"], color=color, linewidth=1.5, linestyle="--", label=f"evidence-only {key}")
    ax_cos.set_ylim(0, 1)
    ax_cos.set_title("Val claim retrieval, kNN by cosine similarity", color=ink, loc="left", fontsize=11)
    ax_cos.set_ylabel("recall", color=muted)
    ax_cos.legend(frameon=False, labelcolor=ink, fontsize=9, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-dir", default=KG_DIR)
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="sentence-transformers model for node texts")
    ap.add_argument("--cache", default=None, help="embedding cache; default scripts/poc/cache/<kg>_<model>.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384, help="embedding size D")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25, help="dropout on the MLP hidden layer; 0 disables")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    ap.add_argument("--ema", type=float, default=0.996, help="target-encoder momentum")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of papers held out")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many epochs with no improvement; 0 disables")
    ap.add_argument("--monitor", default="cosine_MRR", choices=sorted(MONITORS), help="metric early stopping watches")
    ap.add_argument("--drop-result", action="store_true", help="zero the Result input (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="optional checkpoint path")
    ap.add_argument("--results", default=None, help="append one TSV row for this run (for sweeps)")
    ap.add_argument("--plot", default="scripts/poc/train_kg_plot.png", help="figure path; '' to skip")
    args = ap.parse_args()
    if args.patience < 0:
        ap.error("--patience must be 0 or more")

    set_seed(args.seed)
    device = pick_device(args.device)

    model_slug = args.embed_model.split("/")[-1].lower()
    cache_path = Path(args.cache or f"{CACHE_DIR}/{Path(args.kg_dir).name}_{model_slug}.pt")
    if cache_path.exists():
        data = torch.load(cache_path)
    else:
        data = build_cache(args.kg_dir, cache_path, args.embed_model, device)

    # Some caches were written in bfloat16 (a half-precision run); the MLPs are
    # float32, and mixing the two is a hard error, so settle the dtype here.
    for key in ("result", "evidence", "claim"):
        data[key] = data[key].float()

    # Each sample takes its paper's Result vector.
    result = data["result"][data["sample_paper"]]
    if args.drop_result:
        result = torch.zeros_like(result)
    x = torch.stack([data["evidence"], result], dim=1).to(device)  # [N, 2, E]
    claim = data["claim"].to(device)  # [N, E]

    # Split by paper so the same Result text is never in both train and val.
    n_papers = data["result"].shape[0]
    perm = torch.randperm(n_papers, generator=torch.Generator().manual_seed(args.seed))
    is_val = torch.isin(data["sample_paper"], perm[: int(n_papers * args.val_frac)])
    tr = (~is_val).nonzero().squeeze(1).to(device)
    va = is_val.nonzero().squeeze(1).to(device)
    print(
        f"device={device}  embed={args.embed_model} ({x.shape[-1]}-d)  samples train={len(tr)} val={len(va)}"
        f"  papers={n_papers}  dropout={args.dropout}  drop_result={args.drop_result}"
    )

    baseline = {}
    for metric in METRICS:
        ret = retrieval(x[va, 0], claim[va], metric)
        baseline.update({f"{metric}_{k}": v for k, v in ret.items()})
        print(f"evidence-only baseline [{metric:6s}]  R@1 {ret['R@1']:.3f}  R@10 {ret['R@10']:.3f}  MRR {ret['MRR']:.3f}")

    in_dim = x.shape[-1]
    context_enc = Encoder(in_dim, args.dim, args.hidden, args.dropout).to(device)
    predictor = Predictor(args.dim, args.hidden, args.dropout).to(device)
    target_enc = copy.deepcopy(context_enc)
    target_enc.requires_grad_(False)
    # Never undone: the target follows by EMA on the parameters, not by gradient, and a
    # dropped-out target would make the vector the predictor chases random per step.
    target_enc.eval()

    opt = torch.optim.AdamW(
        list(context_enc.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Only tracked when early stopping is on, so a plain run keeps reporting its last epoch
    # and never pays for the per-epoch deepcopy.
    watch = MONITORS[args.monitor] if args.patience else None
    best_score, best_epoch, best_state = None, 0, None
    if watch:
        print(f"early stopping: watching {args.monitor} ({watch}), patience {args.patience} epochs")

    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        context_enc.train()
        predictor.train()
        order = tr[torch.randperm(len(tr), device=device)]
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]

            ctx = context_enc(x[idx])  # [BS, 2, D]
            pred = predictor(ctx)  # [BS, D]
            with torch.no_grad():
                tgt = target_enc(claim[idx])  # [BS, D]

            loss = unit_mse(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

            # Target encoder slowly follows the context encoder.
            with torch.no_grad():
                for p_t, p_c in zip(target_enc.parameters(), context_enc.parameters()):
                    p_t.lerp_(p_c, 1.0 - args.ema)

        # Dropout off, or every validation metric below is noise-inflated.
        context_enc.eval()
        predictor.eval()
        stats = evaluate(context_enc, predictor, target_enc, x[va], claim[va])
        history.append({"epoch": epoch, "train_loss": total / len(tr), **stats})
        print(
            f"epoch {epoch:3d}  loss train {total / len(tr):.6f} val {stats['val_loss']:.6f}"
            f"  tgt_std {stats['tgt_std']:.4f}"
            f"  | cos R@1 {stats['cosine_R@1']:.3f} R@10 {stats['cosine_R@10']:.3f} MRR {stats['cosine_MRR']:.3f}"
            f"  | l2 R@1 {stats['l2_R@1']:.3f} R@10 {stats['l2_R@10']:.3f} MRR {stats['l2_MRR']:.3f}"
        )

        if watch:
            score = stats[args.monitor]
            better = best_score is None or (score > best_score if watch == "max" else score < best_score)
            if better:
                best_score, best_epoch = score, epoch
                # Deepcopy: state_dict() hands back live tensors that training would overwrite.
                best_state = copy.deepcopy(
                    {
                        "context_enc": context_enc.state_dict(),
                        "target_enc": target_enc.state_dict(),
                        "predictor": predictor.state_dict(),
                    }
                )
            elif epoch - best_epoch >= args.patience:
                print(f"early stop at epoch {epoch}: no {args.monitor} gain since epoch {best_epoch} ({best_score:.6g})")
                break

    print(f"trained {len(history)} epochs in {(time.time() - t0) / 60:.1f} min")

    if best_state is not None:
        # Weights from the monitored best, not from wherever patience happened to run out.
        context_enc.load_state_dict(best_state["context_enc"])
        target_enc.load_state_dict(best_state["target_enc"])
        predictor.load_state_dict(best_state["predictor"])
        print(f"restored epoch {best_epoch}  ({args.monitor} {best_score:.6g})")

    if args.results:
        # The restored epoch is the one the weights came from, so report that row.
        row = history[best_epoch - 1] if best_epoch else history[-1]
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists()
        with open(path, "a") as f:
            if header:
                f.write("\t".join(RESULT_COLUMNS) + "\n")
            f.write(
                "\t".join(
                    str(v)
                    for v in (
                        args.embed_model.split("/")[-1], args.batch_size, args.hidden, args.dropout, args.lr, args.seed,
                        row["epoch"], len(history),
                        f"{row['cosine_R@1']:.4f}", f"{row['cosine_R@10']:.4f}", f"{row['cosine_MRR']:.4f}",
                        f"{row['val_loss']:.6g}",
                        f"{baseline['cosine_R@1']:.4f}", f"{baseline['cosine_R@10']:.4f}", f"{baseline['cosine_MRR']:.4f}",
                    )
                )
                + "\n"
            )
        print(f"appended {path}")

    if args.out:
        torch.save(
            {
                "context_enc": context_enc.state_dict(),
                "target_enc": target_enc.state_dict(),
                "predictor": predictor.state_dict(),
                "history": history,
                "best_epoch": best_epoch,  # 0 when early stopping was off
                "args": vars(args),
            },
            args.out,
        )
        print(f"saved {args.out}")

    if args.plot:
        plot(history, baseline, args.plot)


if __name__ == "__main__":
    main()
