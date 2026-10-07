"""JEPA prototype on the materials-science / chem-eng KG with graph inputs and graph targets.

train_kg.py fed the model two loose text vectors. Here each sample is a pair of
small graphs cut from its paper's reasoning chain between Evidence and Claim:

    context graph   paper -> method -> result -> evidence    4 nodes, 3 edges
    target graph                                   claim -> implication   2 nodes, 1 edge

One sample per claim, as before. Every node starts as a frozen text-embedding
vector; a relational graph encoder turns each graph into one D-vector per node:

    context encoder  f_theta : [BS, 4, E] -> [BS, 4, D]
    predictor        g_phi   : [BS, 4, D] -> [BS, 2, D]      one vector per target node
    target encoder   f_xi    : [BS, 2, E] -> [BS, 2, D]      (EMA of f_theta, no grad)
    loss = unit-MSE(g_phi(f_theta(context)), stopgrad(f_xi(target))), averaged over target nodes

f_theta and f_xi are one set of weights (up to the EMA), so the encoder knows
all six node types and the four relations used; which graph it is reading comes
from the node types and edges passed in with it.

The two graphs share no node. An earlier version put Evidence in the target too
(evidence -> claim -> implication); message passing then copied the Evidence into
the target Claim vector, and retrieval matched on that copy instead of the claim
(job 15644766: R@1 fell below baseline once the target's Evidence was blanked).

Validation retrieves held-out samples by kNN, two ways (cosine and L2, R@1,
R@10, MRR):
  claim  the predicted Claim node against all target Claim nodes. Comparable to
         train_kg.py, and the baseline is its evidence-only baseline.
  graph  the mean of the predicted nodes against the mean of each target graph;
         baseline: raw Evidence vs the mean raw target vector.

Run from the repo root. The first run per embedding model reuses train_kg.py's
cached Result/Evidence/Claim vectors and embeds only Paper/Method/Implication:
    python scripts/poc/train_kg_graph.py
    python scripts/poc/train_kg_graph.py --layers 0          # ablation: no message passing
    python scripts/poc/train_kg_graph.py --drop result method # ablation: blank context nodes
    python scripts/poc/train_kg_graph.py --patience 10
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# train_kg.py sits next to this file, and `python scripts/poc/train_kg_graph.py` puts that
# directory on sys.path, so its helpers are shared rather than copied.
from train_kg import CACHE_DIR, EMBED_MODEL, KG_DIR, METRICS, MIN_TOKENS, mlp, pick_device, retrieval, set_seed, unit_mse

# Node types and relations, named as in the KG (dataset_creation/schema.py).
NODE_TYPES = ("paper", "method", "result", "evidence", "claim", "implication")
# Only relations some graph below uses: an unused one would add weights that never train.
RELATIONS = ("has_method", "produces", "grounds", "implies")
PER_PAPER = ("paper", "method", "result")  # one per paper; a sample takes its paper's
PER_CLAIM = ("evidence", "claim", "implication")  # one per claim, i.e. per sample

CONTEXT_NODES = ("paper", "method", "result", "evidence")
CONTEXT_EDGES = ((0, "has_method", 1), (1, "produces", 2), (2, "grounds", 3))
TARGET_NODES = ("claim", "implication")
TARGET_EDGES = ((0, "implies", 1),)

# How one vector is read out of a graph's node embeddings for retrieval.
READOUTS = {
    "claim": lambda h: h[:, TARGET_NODES.index("claim")],
    "graph": lambda h: h.mean(dim=1),
}
MONITORS = {
    f"{r}_cosine_{k}": "max" for r in READOUTS for k in ("MRR", "R@1", "R@10")
} | {"val_loss": "min"}
# R@1/R@10/MRR and base_* are the claim readout, so grid_report.py and seed_report.py
# compare these runs against the same evidence-only baseline as train_kg.py's.
RESULT_COLUMNS = (
    "embed", "batch", "hidden", "layers", "dropout", "lr", "seed",
    "best_epoch", "epochs", "R@1", "R@10", "MRR", "val_loss",
    "base_R@1", "base_R@10", "base_MRR",
    "graph_R@1", "graph_R@10", "graph_MRR", "graph_base_R@1", "graph_base_R@10", "graph_base_MRR",
)


class Graph:
    """A fixed-shape graph shared by every sample in a batch: node types plus typed edges.

    Every relation is also added backwards under its own id, so information flows both
    ways along the chain (the same thing T.ToUndirected does with rev_* edges in PyG).
    """

    def __init__(self, nodes, edges, device):
        self.types = torch.tensor([NODE_TYPES.index(n) for n in nodes], device=device)
        src, dst, rel = [], [], []
        for s, r, d in edges:
            k = RELATIONS.index(r)
            src += [s, d]
            dst += [d, s]
            rel += [2 * k, 2 * k + 1]  # 2k forwards, 2k+1 backwards
        self.src = torch.tensor(src, device=device)
        self.dst = torch.tensor(dst, device=device)
        self.rel = torch.tensor(rel, device=device)
        # Incoming-edge count per node, to average messages. A chain has no isolated nodes.
        self.deg = torch.bincount(self.dst, minlength=len(nodes)).clamp(min=1).float()


class RelationalLayer(nn.Module):
    """One round of message passing with a weight matrix per relation and direction (R-GCN).

    Each node averages the messages W_rel h_src from its neighbours, adds its own
    transformed state, and keeps a residual path so stacked layers start near identity.
    """

    def __init__(self, dim: int, n_rel: int, dropout: float):
        super().__init__()
        self.rel = nn.Parameter(torch.empty(n_rel, dim, dim))
        self.self_loop = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        for w in self.rel:
            nn.init.kaiming_uniform_(w, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.self_loop.weight, nonlinearity="relu")
        nn.init.zeros_(self.self_loop.bias)

    def forward(self, h, g: Graph):  # [BS, N, D] -> [BS, N, D]
        msg = torch.einsum("bed,edk->bek", h[:, g.src], self.rel[g.rel])  # [BS, edges, D]
        agg = torch.zeros_like(h).index_add_(1, g.dst, msg) / g.deg[:, None]
        return self.norm(h + self.drop(F.gelu(self.self_loop(h) + agg)))


class GraphEncoder(nn.Module):
    """Graph of frozen text vectors -> one D-vector per node: [BS, N, E] -> [BS, N, D].

    The per-node MLP is train_kg.py's Encoder; a node-type embedding is added so a
    Claim and an Evidence are told apart; `layers` rounds of message passing follow.
    With layers=0 each node is encoded alone, as in train_kg.py.
    """

    def __init__(self, in_dim: int, dim: int, hidden: int, dropout: float, layers: int):
        super().__init__()
        self.embed = mlp(in_dim, hidden, dim, dropout)
        self.node_type = nn.Embedding(len(NODE_TYPES), dim)
        nn.init.normal_(self.node_type.weight, std=0.02)
        self.layers = nn.ModuleList(RelationalLayer(dim, 2 * len(RELATIONS), dropout) for _ in range(layers))

    def forward(self, x, g: Graph):
        h = self.embed(x) + self.node_type(g.types)
        for layer in self.layers:
            h = layer(h, g)
        return h


class Predictor(nn.Module):
    """Context node embeddings -> one predicted embedding per target node."""

    def __init__(self, n_in: int, n_out: int, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.n_out = n_out
        self.net = mlp(n_in * dim, hidden, n_out * dim, dropout)

    def forward(self, ctx):  # [BS, n_in, D] -> [BS, n_out, D]
        return self.net(ctx.flatten(1)).unflatten(1, (self.n_out, -1))


def load_texts(kg_dir: str):
    """subgraphs.jsonl -> texts per node type, plus each sample's paper index.

    Claims are kept exactly as train_kg.load_samples keeps them, so its cached
    Result/Evidence/Claim vectors line up with these rows and can be reused.
    """
    texts = {k: [] for k in NODE_TYPES}
    sample_paper = []
    with open(Path(kg_dir) / "subgraphs.jsonl") as f:
        for line in f:
            paper = json.loads(line)
            p = len(texts["paper"])
            texts["paper"].append(paper["text"])
            texts["method"].append(paper["method"]["text"])
            texts["result"].append(paper["result"]["text"])
            for c in paper["claims"]:
                if not c["evidence"]:
                    continue
                if not c["implications"]:  # none in this KG; skipping would misalign the reused cache
                    raise ValueError(f"claim {c['id']} has no implication")
                sample_paper.append(p)
                texts["evidence"].append(c["evidence"][0]["text"])  # exactly one per claim
                texts["claim"].append(c["text"])
                texts["implication"].append(c["implications"][0]["text"])  # exactly one per claim
    return texts, sample_paper


def build_cache(kg_dir: str, path: Path, text_cache: Path, model_name: str, device: torch.device) -> dict:
    """Embed every node type once and save; vectors train_kg.py already cached are reused."""
    texts, sample_paper = load_texts(kg_dir)
    cache = {"sample_paper": torch.tensor(sample_paper), "model": model_name}

    if text_cache.exists():
        old = torch.load(text_cache)
        if torch.equal(old["sample_paper"], cache["sample_paper"]):
            for k in ("result", "evidence", "claim"):
                cache[k] = old[k]
            print(f"reused result/evidence/claim vectors from {text_cache}")
        else:
            print(f"{text_cache} has different samples; embedding everything")

    missing = [k for k in NODE_TYPES if k not in cache]
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=str(device))
        max_len = model.max_seq_length
        if max_len < MIN_TOKENS:
            raise ValueError(f"{model_name} reads only {max_len} tokens; need at least {MIN_TOKENS}")
        print(f"embedding {missing} with {model_name}  (max {max_len} tokens, {model.get_sentence_embedding_dimension()}-d)")
        for k in missing:
            n_tokens = [len(ids) for ids in model.tokenizer(texts[k])["input_ids"]]
            cut = sum(n > max_len for n in n_tokens)
            print(f"{k:12s} {len(texts[k]):6d} texts, {cut / len(texts[k]):.1%} truncated at {max_len} tokens")
            # Unit length, like the reused vectors, so L2 and cosine rank raw vectors identically.
            cache[k] = model.encode(
                texts[k], batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
            ).cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"saved {path}")
    return cache


def gather(data: dict, idx: torch.Tensor, drop: list):
    """Node vectors of samples idx: context [n, 4, E], target [n, 2, E].

    Per-paper vectors are looked up here rather than copied into every sample up front,
    which for Qwen would hold ~0.8 GB of duplicates on the GPU.
    """
    p = data["sample_paper"][idx]
    ctx = torch.stack([data[k][p] if k in PER_PAPER else data[k][idx] for k in CONTEXT_NODES], dim=1)
    tgt = torch.stack([data[k][idx] for k in TARGET_NODES], dim=1)
    if drop:
        ctx[:, [CONTEXT_NODES.index(k) for k in drop]] = 0.0
    return ctx, tgt


def retrieval_all(pred: torch.Tensor, tgt: torch.Tensor) -> dict:
    """Every readout x metric of `retrieval`, keyed like 'claim_cosine_MRR'."""
    out = {}
    for r, read in READOUTS.items():
        for metric in METRICS:
            out.update({f"{r}_{metric}_{k}": v for k, v in retrieval(read(pred), read(tgt), metric).items()})
    return out


@torch.no_grad()
def evaluate(context_enc, predictor, target_enc, ctx, tgt, g_ctx, g_tgt) -> dict:
    pred = predictor(context_enc(ctx, g_ctx))  # [N, 2, D]
    target = target_enc(tgt, g_tgt)  # [N, 2, D]
    return {
        "val_loss": unit_mse(pred, target).item(),
        # Spread of target-graph embeddings across samples; ~0 means collapse.
        "tgt_std": target.mean(dim=1).std(dim=0).mean().item(),
        **retrieval_all(pred, target),
    }


@torch.no_grad()
def baseline(ctx: torch.Tensor, tgt: torch.Tensor) -> dict:
    """The same searches untrained: the raw Evidence vector stands in for every prediction."""
    ev = ctx[:, CONTEXT_NODES.index("evidence")]
    return retrieval_all(ev[:, None].expand_as(tgt), tgt)


def plot(history: list, base: dict, path: str):
    """Training loss, then cosine kNN retrieval per readout against its evidence-only baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3de"
    blue, orange = "#2a78d6", "#eb6834"

    fig, axes = plt.subplots(1, 1 + len(READOUTS), figsize=(18, 4.5), facecolor="#fcfcfb")
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

    ax_loss = axes[0]
    ax_loss.plot(epochs, [h["train_loss"] for h in history], color=blue, linewidth=2, label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], color=orange, linewidth=2, label="val")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Unit-length MSE over target nodes  [training loss]", color=ink, loc="left", fontsize=11)
    ax_loss.set_ylabel("MSE (log scale)", color=muted)
    ax_loss.legend(frameon=False, labelcolor=ink, fontsize=9)

    for ax, r in zip(axes[1:], READOUTS):
        for key, color in (("R@1", blue), ("R@10", orange)):
            ax.plot(epochs, [h[f"{r}_cosine_{key}"] for h in history], color=color, linewidth=2, label=f"JEPA {key}")
            ax.axhline(base[f"{r}_cosine_{key}"], color=color, linewidth=1.5, linestyle="--", label=f"evidence-only {key}")
        ax.set_ylim(0, 1)
        ax.set_title(f"Val {r} retrieval, kNN by cosine similarity", color=ink, loc="left", fontsize=11)
        ax.set_ylabel("recall", color=muted)
        ax.legend(frameon=False, labelcolor=ink, fontsize=9, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-dir", default=KG_DIR)
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="sentence-transformers model for node texts")
    ap.add_argument("--cache", default=None, help="embedding cache; default scripts/poc/cache/<kg>_<model>_graph.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384, help="embedding size D")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2, help="message-passing rounds; 0 encodes each node alone")
    ap.add_argument("--dropout", type=float, default=0.25, help="dropout in the MLPs and GNN layers; 0 disables")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    ap.add_argument("--ema", type=float, default=0.996, help="target-encoder momentum")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of papers held out")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many epochs with no improvement; 0 disables")
    ap.add_argument("--monitor", default="claim_cosine_MRR", choices=sorted(MONITORS), help="metric early stopping watches")
    ap.add_argument("--drop", nargs="*", default=[], choices=CONTEXT_NODES, help="zero these context nodes (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="optional checkpoint path")
    ap.add_argument("--results", default=None, help="append one TSV row for this run (for sweeps)")
    ap.add_argument("--plot", default="scripts/poc/plots/train_kg_graph_plot.png", help="figure path; '' to skip")
    args = ap.parse_args()
    if args.patience < 0:
        ap.error("--patience must be 0 or more")
    if args.layers < 0:
        ap.error("--layers must be 0 or more")

    set_seed(args.seed)
    device = pick_device(args.device)

    model_slug = args.embed_model.split("/")[-1].lower()
    text_cache = Path(f"{CACHE_DIR}/{Path(args.kg_dir).name}_{model_slug}.pt")
    cache_path = Path(args.cache or f"{CACHE_DIR}/{Path(args.kg_dir).name}_{model_slug}_graph.pt")
    if cache_path.exists():
        data = torch.load(cache_path)
    else:
        data = build_cache(args.kg_dir, cache_path, text_cache, args.embed_model, device)

    # The Qwen vectors are stored as bfloat16; the model is float32.
    for k in NODE_TYPES:
        data[k] = data[k].float().to(device)
    data["sample_paper"] = data["sample_paper"].to(device)
    g_ctx = Graph(CONTEXT_NODES, CONTEXT_EDGES, device)
    g_tgt = Graph(TARGET_NODES, TARGET_EDGES, device)

    # Split by paper so the same Paper/Method/Result never appears in both train and val.
    n_papers = data["paper"].shape[0]
    perm = torch.randperm(n_papers, generator=torch.Generator().manual_seed(args.seed)).to(device)
    is_val = torch.isin(data["sample_paper"], perm[: int(n_papers * args.val_frac)])
    tr = (~is_val).nonzero().squeeze(1)
    va = is_val.nonzero().squeeze(1)
    ctx_va, tgt_va = gather(data, va, args.drop)
    in_dim = ctx_va.shape[-1]
    print(
        f"device={device}  embed={args.embed_model} ({in_dim}-d)  samples train={len(tr)} val={len(va)}"
        f"  papers={n_papers}  layers={args.layers}  dropout={args.dropout}  drop={args.drop or 'none'}"
    )

    # From the unblanked vectors: the baseline must not depend on the ablation.
    base = baseline(*gather(data, va, []))
    for r in READOUTS:
        for metric in METRICS:
            b = {k: base[f"{r}_{metric}_{k}"] for k in ("R@1", "R@10", "MRR")}
            print(f"evidence-only baseline [{r:5s} {metric:6s}]  R@1 {b['R@1']:.3f}  R@10 {b['R@10']:.3f}  MRR {b['MRR']:.3f}")

    context_enc = GraphEncoder(in_dim, args.dim, args.hidden, args.dropout, args.layers).to(device)
    predictor = Predictor(len(CONTEXT_NODES), len(TARGET_NODES), args.dim, args.hidden, args.dropout).to(device)
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
            ctx, tgt = gather(data, idx, args.drop)

            pred = predictor(context_enc(ctx, g_ctx))  # [BS, 2, D]
            with torch.no_grad():
                target = target_enc(tgt, g_tgt)  # [BS, 2, D]

            loss = unit_mse(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

            # Target encoder slowly follows the context encoder.
            with torch.no_grad():
                for p_t, p_c in zip(target_enc.parameters(), context_enc.parameters()):
                    p_t.lerp_(p_c, 1.0 - args.ema)

        context_enc.eval()
        predictor.eval()
        stats = evaluate(context_enc, predictor, target_enc, ctx_va, tgt_va, g_ctx, g_tgt)
        history.append({"epoch": epoch, "train_loss": total / len(tr), **stats})
        print(
            f"epoch {epoch:3d}  loss train {total / len(tr):.6f} val {stats['val_loss']:.6f}"
            f"  tgt_std {stats['tgt_std']:.4f}"
            + "".join(
                f"  | {r} cos R@1 {stats[f'{r}_cosine_R@1']:.3f} R@10 {stats[f'{r}_cosine_R@10']:.3f}"
                f" MRR {stats[f'{r}_cosine_MRR']:.3f}"
                for r in READOUTS
            )
        )

        if watch:
            score = stats[args.monitor]
            better = best_score is None or (score > best_score if watch == "max" else score < best_score)
            if better:
                best_score, best_epoch = score, epoch
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
        context_enc.load_state_dict(best_state["context_enc"])
        target_enc.load_state_dict(best_state["target_enc"])
        predictor.load_state_dict(best_state["predictor"])
        print(f"restored epoch {best_epoch}  ({args.monitor} {best_score:.6g})")

    if args.results:
        row = history[best_epoch - 1] if best_epoch else history[-1]
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists()
        cos = lambda d, r, k: f"{d[f'{r}_cosine_{k}']:.4f}"  # noqa: E731
        with open(path, "a") as f:
            if header:
                f.write("\t".join(RESULT_COLUMNS) + "\n")
            f.write(
                "\t".join(
                    str(v)
                    for v in (
                        args.embed_model.split("/")[-1], args.batch_size, args.hidden, args.layers, args.dropout,
                        args.lr, args.seed, row["epoch"], len(history),
                        *(cos(row, "claim", k) for k in ("R@1", "R@10", "MRR")),
                        f"{row['val_loss']:.6g}",
                        *(cos(base, "claim", k) for k in ("R@1", "R@10", "MRR")),
                        *(cos(row, "graph", k) for k in ("R@1", "R@10", "MRR")),
                        *(cos(base, "graph", k) for k in ("R@1", "R@10", "MRR")),
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
                "best_epoch": best_epoch,
                "args": vars(args),
            },
            args.out,
        )
        print(f"saved {args.out}")

    if args.plot:
        plot(history, base, args.plot)


if __name__ == "__main__":
    main()
