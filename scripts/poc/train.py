"""Toy JEPA prototype (see tasks.md).

Each sample is a tiny three-node "paper" -- Evidence, Result, Claim -- where
every node is a 512-d signal. The model sees Evidence and Result and predicts
the *embedding* of the Claim. As in JEPA, the loss lives in latent space, not
in signal space.

    k ~ U(K_MIN, K_MAX), r ~ U(R_MIN, R_MAX) per sample
    a = r * k / K_MAX                           (Result amplitude, linked to Evidence)

    evidence = [k] * 512
    result   = a * sin(2*pi * 0.1 * t)
    claim    = a * sin(2*pi * 0.1 * k * t)      (frequency from Evidence, amplitude from Result)

    context encoder  f_theta : [BS, 2, 512] -> [BS, 2, D]   (Evidence, Result)
    predictor        g_phi   : [BS, 2, D]   -> [BS, D]
    target encoder   f_xi    : [BS, 512]    -> [BS, D]      (Claim; EMA of f_theta, no grad)
    loss = MSE(g_phi(f_theta(x)), stopgrad(f_xi(claim)))

Run from the repo root:
    python scripts/poc/train.py --steps 2000
"""

import argparse
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIG_LEN = 512
RESULT_FREQ = 0.1
K_MIN, K_MAX = 0.1, 1.0
R_MIN, R_MAX = 0.5, 1.5


def time_axis(device: torch.device) -> torch.Tensor:
    """512 real-valued time points in [0, 1]."""
    return torch.from_numpy(np.linspace(0.0, 1.0, SIG_LEN)).float().to(device)


def signals(k: torch.Tensor, a: torch.Tensor, device: torch.device):
    """Evidence, Result, Claim for evidence k and amplitude a (both [BS, 1]) -> three [BS, 512]."""
    t = time_axis(device)
    evidence = k.expand(-1, SIG_LEN)
    result = a * torch.sin(2 * math.pi * RESULT_FREQ * t)
    claim = a * torch.sin(2 * math.pi * RESULT_FREQ * k * t)
    return evidence, result, claim


def make_batch(bs: int, device: torch.device):
    """Return x: [BS, 2, 512] (Evidence, Result) and claim: [BS, 512]."""
    k = torch.empty(bs, 1, device=device).uniform_(K_MIN, K_MAX)
    r = torch.empty(bs, 1, device=device).uniform_(R_MIN, R_MAX)
    a = r * k / K_MAX
    evidence, result, claim = signals(k, a, device)
    x = torch.stack([evidence, result], dim=1)
    return x, claim


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class Encoder(nn.Module):
    """Node encoder shared across node types: [..., 512] -> [..., D]."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = mlp(SIG_LEN, hidden, dim)

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Context embeddings of (Evidence, Result) -> predicted Claim embedding."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = mlp(2 * dim, hidden, dim)

    def forward(self, ctx):  # [BS, 2, D] -> [BS, D]
        return self.net(ctx.flatten(1))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot(losses: list, path: str):
    """Left: Result and Claims for a few evidence values. Right: MSE loss per step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpu = torch.device("cpu")
    # k=1 would make the Claim identical to the Result and hide it. Plot with r=1.
    ks = torch.tensor([[2.0], [5.0], [10.0]])
    _, result, claim = signals(ks, ks / K_MAX, cpu)
    t = time_axis(cpu)
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3de"

    fig, (ax_sig, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#fcfcfb")
    for ax in (ax_sig, ax_loss):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)

    ax_sig.plot(t, result[0], color="#2a78d6", linewidth=2, label="Result  sin(2π·0.1·t)")
    for i, color in enumerate(["#eb6834", "#1baf7a", "#eda100"]):
        k = ks[i, 0].item()
        ax_sig.plot(t, claim[i], color=color, linewidth=2, label=f"Claim  k={k:g}")
    ax_sig.set_title("Result and Claim signals  (claim = sin(2π·0.1·k·t))", color=ink, loc="left", fontsize=11)
    ax_sig.set_xlabel("t", color=muted)
    ax_sig.set_ylabel("value", color=muted)
    ax_sig.set_ylim(-1.15, 1.75)
    ax_sig.legend(frameon=False, labelcolor=ink, fontsize=9, loc="upper center", ncol=2)

    ax_loss.plot(range(1, len(losses) + 1), losses, color="#2a78d6", linewidth=1.5)
    ax_loss.set_yscale("log")
    ax_loss.set_title("MSE(predictor, target encoder)", color=ink, loc="left", fontsize=11)
    ax_loss.set_xlabel("step", color=muted)
    ax_loss.set_ylabel("loss (log scale)", color=muted)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512, help="embedding size D")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ema", type=float, default=0.996, help="target-encoder momentum")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="optional checkpoint path")
    ap.add_argument("--plot", default="scripts/poc/train_plot.png", help="figure path; '' to skip")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    context_enc = Encoder(args.dim, args.hidden).to(device)
    predictor = Predictor(args.dim, args.hidden).to(device)
    target_enc = copy.deepcopy(context_enc)
    target_enc.requires_grad_(False)

    opt = torch.optim.Adam(
        list(context_enc.parameters()) + list(predictor.parameters()), lr=args.lr
    )

    print(f"device={device}  params={sum(p.numel() for p in context_enc.parameters()) + sum(p.numel() for p in predictor.parameters()):,}")

    losses = []
    for step in range(1, args.steps + 1):
        x, claim = make_batch(args.batch_size, device)

        ctx = context_enc(x)  # [BS, 2, D]
        pred = predictor(ctx)  # [BS, D]
        with torch.no_grad():
            tgt = target_enc(claim)  # [BS, D]

        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        # Target encoder slowly follows the context encoder.
        with torch.no_grad():
            for p_t, p_c in zip(target_enc.parameters(), context_enc.parameters()):
                p_t.lerp_(p_c, 1.0 - args.ema)

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                cos = F.cosine_similarity(pred, tgt, dim=-1).mean().item()
            print(f"step {step:5d}  loss {loss.item():.6f}  cos {cos:.4f}")

    assert x.shape[1:] == (2, SIG_LEN) and pred.shape[1:] == (args.dim,)

    if args.out:
        torch.save(
            {
                "context_enc": context_enc.state_dict(),
                "target_enc": target_enc.state_dict(),
                "predictor": predictor.state_dict(),
                "args": vars(args),
            },
            args.out,
        )
        print(f"saved {args.out}")

    if args.plot:
        plot(losses, args.plot)


if __name__ == "__main__":
    main()
