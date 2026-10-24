import os
import argparse
from pathlib import Path
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'ZINC'
    # Additional num of worker for data loading
    cfg.num_workers = 8
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Custom log file name
    cfg.logfile = None
    # tree depth for TreeDataset
    cfg.depth = -1
    # Number of cross-validation folds
    cfg.k = 2

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    # Total graph mini-batch size
    cfg.train.batch_size = 128
    # Maximal number of epochs
    cfg.train.epochs = 1000
    # Number of runs with random init
    cfg.train.runs = 4
    # Base learning rate
    cfg.train.lr = 0.001
    # number of steps before reduce learning rate
    cfg.train.lr_patience = 20
    # learning rate decay factor
    cfg.train.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.train.wd = 0.
    # Dropout rate
    cfg.train.dropout = 0.
    # Dropout rate for MLPMixer
    cfg.train.mlpmixer_dropout = 0.
    # A lower bound on the learning rate.
    cfg.train.min_lr = 1e-5
    # optimizer: Adam, SGD, or ASAM (ASAM is a sharpness-aware step on top of SGD)
    cfg.train.optimizer = 'Adam'
    # lightning (default) or legacy (the original hand-written loops)
    cfg.train.backend = 'lightning'
    # auto selects vicreg for papers and distance for the TU / ZINC scripts
    cfg.train.objective = 'auto'
    # full updates every EMA pair; encoder updates only the context encoder
    cfg.train.ema_scope = 'full'
    # 0 keeps the single GPU in cfg.device. >1 turns on DDP across that many GPUs.
    cfg.train.devices = 0
    # auto or ddp
    cfg.train.strategy = 'auto'
    # one seed per GPU, sequential seeds when this is false
    cfg.train.parallel_runs = False
    # 0 disables clipping. The papers run sets this to 1.
    cfg.train.grad_clip = 0.0
    cfg.train.precision = '32-true'
    # Speed switches (docs/training-optimization.md). None of them changes the loss.
    # The VICReg representation reuses the online backbone pass instead of running it twice.
    cfg.train.reuse_online_pass = True
    # EMA target update as one torch._foreach call instead of a Python loop over parameters.
    cfg.train.foreach_ema = True
    # Fused Adam/SGD kernels on CUDA.
    cfg.train.fused_optim = True
    # Batches each DataLoader worker prepares ahead (Lightning backend).
    cfg.train.prefetch_factor = 2
    # SmoothL1 beta. 0.5 matches ZINC/DD/IMDB/REDDIT; MUTAG/PROTEINS/EXP use 1.
    cfg.train.smooth_l1_beta = 0.5
    cfg.train.sim_weight = 1.0
    # import path of the training script, filled in for multi-process seeds
    cfg.train.entry = ''
    # Multiscale training
    cfg.train.multiscale = False
    # ── anti-collapse (VICReg) regularizer weights ──
    cfg.train.var_weight = 1.0      # VICReg variance hinge weight
    cfg.train.cov_weight = 0.04     # VICReg covariance weight
    # Write runs/<group>/seed-<s>/checkpoints/[fold<k>/]last.ckpt after every epoch.
    cfg.train.save_last = True
    # Resume from a Lightning checkpoint: '' (off), 'auto' (last.ckpt of this run, needs logfile), or a path.
    cfg.train.resume = ''

    # ------------------------------------------------------------------------ #
    # Lightning Trainer flags. None keeps Lightning's default.
    # ------------------------------------------------------------------------ #
    cfg.trainer = CN()
    # Run 1 (or N) train and val batches as a smoke test, no checkpoints or probe logs.
    cfg.trainer.fast_dev_run = None
    # Fraction (0.1) or number (20) of batches per epoch.
    cfg.trainer.limit_train_batches = None
    # Same for the validation and test loaders.
    cfg.trainer.limit_val_batches = None
    # Train on a fixed fraction or number of batches to check the model can overfit.
    cfg.trainer.overfit_batches = None
    # Validate (and step the plateau scheduler) every N epochs.
    cfg.trainer.check_val_every_n_epoch = None
    # Stop after this wall time, "DD:HH:MM:SS". Pair with train.resume auto under SLURM.
    cfg.trainer.max_time = None
    # simple, advanced or pytorch
    cfg.trainer.profiler = None
    # cudnn.benchmark. True overrides the deterministic setting in core.trainer.set_seed.
    cfg.trainer.benchmark = None
    # Autograd anomaly detection (slow), for NaN hunting.
    cfg.trainer.detect_anomaly = None
    # Lightning progress bar. The epoch summary line prints either way.
    cfg.trainer.enable_progress_bar = None
    # Steps between logger writes for step-level metrics (train_loss_step).
    cfg.trainer.log_every_n_steps = 50

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # GNN type used, see core.model_utils.pyg_gnn_wrapper for all options
    cfg.model.gnn_type = 'GINEConv'  # change to list later
    # GraphMLPMixer or graph-based multihead attention: [MLPMixer, Hadamard, Standard, Graph, Addictive, Kernel]
    cfg.model.gMHA_type = 'MLPMixer'
    # Hidden size of the model
    cfg.model.hidden_size = 128
    # Number of gnn layers
    cfg.model.nlayer_gnn = 4
    # Number of mlp mixer layers
    cfg.model.nlayer_mlpmixer = 4
    # Pooling type for generaating graph/subgraph embedding from node embeddings
    cfg.model.pool = 'mean'
    # Use residual connection
    cfg.model.residual = True
    # Number of output classes for the papers dataset (coarse field labels)
    cfg.model.papers_nout = 0

    # ------------------------------------------------------------------------ #
    # Positional encoding options
    # ------------------------------------------------------------------------ #
    cfg.pos_enc = CN()
    # Random walk structural encoding
    cfg.pos_enc.rw_dim = 0
    # Laplacian eigenvectors positional encoding
    cfg.pos_enc.lap_dim = 0
    # Patch random walk structural encoding
    cfg.pos_enc.patch_rw_dim = 8
    # Patch PE diffusion steps
    cfg.pos_enc.patch_num_diff = -1

    # ------------------------------------------------------------------------ #
    # Metis patch extraction options
    # ------------------------------------------------------------------------ #
    cfg.metis = CN()
    # Enable Metis partition (otherwise use random partition)
    cfg.metis.enable = True
    # Enable data augmentation
    cfg.metis.online = True
    # The number of partitions
    cfg.metis.n_patches = 32
    # Whether to randomly drop a set of edges before each metis partition
    cfg.metis.drop_rate = 0.3
    # expanding patches with k hop neighbourhood
    cfg.metis.num_hops = 1

    # ------------------------------------------------------------------------ #
    # JEPA options
    # ------------------------------------------------------------------------ #
    cfg.jepa = CN()
    # Use Graph-JEPA
    cfg.jepa.enable = True
    # Number of patches to use as context
    cfg.jepa.num_context = 1
    # Number of patches to use as targets
    cfg.jepa.num_targets = 2  # 4
    # Distance function: 0 = 2d Hyper (Ours), 1 = Euclidean, 2 = Hyperbolic (Poincaré)
    cfg.jepa.dist = 0
    # EMA target-network momentum (anti-collapse)
    cfg.jepa.ema_momentum = 0.996
    # Momentum reached at the last optimization step
    cfg.jepa.ema_momentum_end = 0.999

    # ------------------------------------------------------------------------ #
    # Data and logging
    # ------------------------------------------------------------------------ #
    cfg.data = CN()
    # Full dataset root for this run. Empty uses SCIJEPA_DATA_ROOT or Analysis/dataset.
    cfg.data.root = ''

    cfg.logging = CN()
    # Run directories: <root>/<logfile or dataset-timestamp>/seed-<s>. Relative to the repo root.
    cfg.logging.root = 'runs'
    # Lightning loggers: any of tensorboard, csv, wandb, mlflow.
    cfg.logging.loggers = ['tensorboard']
    # False removes tensorboard from logging.loggers (kept for older configs).
    cfg.logging.tensorboard = True
    # W&B project or MLflow experiment name.
    cfg.logging.project = 'graph-jepa'
    # MLflow tracking URI. Empty writes to <logging.root>/mlruns.
    cfg.logging.mlflow_uri = ''
    # events.jsonl next to the logs (always written; kept for older configs)
    cfg.logging.jsonl = True
    # Folder with the existing dashboard (index.html, runs.json, runs/).
    # Empty uses SCIJEPA_DASHBOARD_ROOT or
    # SCI-JEPA-training/runs/paper_a_results/dashboard.
    cfg.logging.dashboard_root = ''

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def resolve_config(name: str) -> str:
    """Find a YAML config from the repo root or from ``Analysis/``."""
    if not name:
        return name
    raw = Path(name)
    analysis = Path(__file__).resolve().parents[1]
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([
        Path.cwd() / name,
        analysis / name,
        analysis / "train" / "configs" / raw.name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return name


# Short flags for the options people change most. Each one sets a config key.
SHORTCUTS = [
    ("--epochs", "train.epochs", "maximum number of epochs"),
    ("--lr", "train.lr", "base learning rate"),
    ("--batch-size", "train.batch_size", "graphs per batch"),
    ("--runs", "train.runs", "number of seeds"),
    ("--seed", "seed", "run this one seed only"),
    ("--device", "device", "CUDA device index"),
    ("--devices", "train.devices", ">1 trains one seed with DDP on that many GPUs"),
    ("--precision", "train.precision", "32-true, bf16-mixed, 16-mixed"),
    ("--workers", "num_workers", "DataLoader workers"),
    ("--name", "logfile", "run group name; fixed names make --resume auto work"),
    ("--resume", "train.resume", "auto or a checkpoint path"),
    ("--loggers", "logging.loggers", "comma list of tensorboard,csv,wandb,mlflow"),
]

EPILOG = """
Any config key can be set as `KEY VALUE` or `KEY=VALUE` after the flags, e.g.
  python -m train.papers --epochs 5 train.lr 1e-4 metis.n_patches=16
  python -m train.papers --options lr          # keys matching "lr", with values
  python -m train.papers --print-config --lr 3e-4 > my.yaml
  python -m train.papers --config my.yaml trainer.fast_dev_run 1
Order: script defaults, --config file, KEY VALUE pairs, then the flags above.
"""


def flatten_cfg(node, prefix=""):
    """``{'train.lr': 0.001, ...}`` for every leaf of a config node."""
    out = {}
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, CN):
            out.update(flatten_cfg(value, name + "."))
        else:
            out[name] = value
    return out


def option_docs():
    """Comments written above each key in ``set_cfg``, keyed by dotted name."""
    import re

    docs, pending = {}, []
    assign = re.compile(r"^\s*cfg\.([\w.]+)\s*=\s*(.*)$")
    for line in Path(__file__).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and docs:
            break
        if stripped.startswith(("# ---", "# ──")) or not stripped:
            pending = []
            continue
        if stripped.startswith("#"):
            pending.append(stripped.lstrip("# ").strip())
            continue
        match = assign.match(line)
        if not match:
            pending = []
            continue
        inline = match.group(2).split("#", 1)[1].strip() if "#" in match.group(2) else ""
        text = " ".join(pending + ([inline] if inline else []))
        if not match.group(2).startswith("CN()"):
            docs[match.group(1)] = text
        pending = []
    return docs


def print_options(cfg, pattern=""):
    docs = option_docs()
    rows = [(k, v) for k, v in flatten_cfg(cfg).items() if pattern.lower() in k.lower()]
    width = max((len(k) for k, _ in rows), default=0)
    for key, value in rows:
        doc = docs.get(key, "")
        print(f"{key:<{width}}  {value!r:<14}  {doc}".rstrip())


def _split_opts(tokens):
    """Accept both ``key value`` and ``key=value`` in the positional list."""
    out, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if "=" in token and not token.startswith(("[", "(", "'", '"')):
            key, value = token.split("=", 1)
            out.extend([key, value])
            index += 1
        elif index + 1 < len(tokens):
            out.extend([token, tokens[index + 1]])
            index += 2
        else:
            raise SystemExit(f"Config override `{token}` has no value. Use KEY VALUE or KEY=VALUE.")
    return out


def merge_opts(cfg, opts):
    """``cfg.merge_from_list`` with a readable error and a suggestion for typos."""
    import difflib

    keys = list(flatten_cfg(cfg))
    for key in opts[0::2]:
        if key not in keys:
            close = difflib.get_close_matches(key, keys, n=3, cutoff=0.6)
            close += [k for k in keys if k.endswith("." + key) and k not in close]
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise SystemExit(f"Unknown config key `{key}`.{hint} List keys with --options.")
    try:
        cfg.merge_from_list(opts)
    except (AssertionError, ValueError) as exc:
        raise SystemExit(f"Bad config override: {exc}") from None


def _shortcut_value(key, value):
    if key == "logging.loggers":
        names = [name.strip() for name in value.split(",") if name.strip()]
        return repr(names)
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Graph-JEPA training. Defaults live in core/config.py and train/configs/<dataset>.yaml.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', default="",
                        metavar="FILE", help="YAML file merged over the script defaults")
    for flag, key, text in SHORTCUTS:
        parser.add_argument(flag, dest=key, default=None, metavar="V", help=f"{text} ({key})")
    parser.add_argument("--fast-dev-run", nargs="?", const=1, default=None, type=int, metavar="N",
                        help="smoke test on N batches (trainer.fast_dev_run)")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved config as YAML and exit")
    parser.add_argument("--options", nargs="?", const="", default=None, metavar="PATTERN",
                        help="list config keys (matching PATTERN) with values and help, then exit")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs="*",
                        help="config overrides as KEY VALUE or KEY=VALUE")
    return parser


def update_cfg(cfg, args_str=None):
    parser = build_parser()
    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_intermixed_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_intermixed_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if args.config:
        config_path = resolve_config(args.config)
        if not os.path.isfile(config_path):
            raise SystemExit(f"Config file not found: {args.config}")
        cfg.merge_from_file(config_path)

    # Update from command line
    merge_opts(cfg, _split_opts(args.opts))
    flags = []
    for _, key, _ in SHORTCUTS:
        value = getattr(args, key)
        if value is not None:
            flags.extend([key, _shortcut_value(key, value)])
    if args.fast_dev_run is not None:
        flags.extend(["trainer.fast_dev_run", str(args.fast_dev_run)])
    merge_opts(cfg, flags)

    if args.options is not None:
        print_options(cfg, args.options)
        raise SystemExit(0)
    if args.print_config:
        print(cfg.dump(), end="")
        raise SystemExit(0)
    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())