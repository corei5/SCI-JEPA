"""
All hyperparameters and ablation switches for Paper A, in one dataclass.

Every experiment in the plan (E1-E5, E10-E12) is a preset that overrides a
few fields of `PaperAConfig`; see EXPERIMENTS at the bottom.
"""
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Paths come from the environment (set by the run scripts); the defaults are relative to the working directory.
#   PAPER_A_PARQUET    normalized Scientific-Summaries parquet folder(s), comma-separated
#   PAPER_A_INDEX_DIR  KU index cache (built once per data setting)
#   PAPER_A_OUT        results, models, dashboard


@dataclass
class PaperAConfig:
    # ---- data: normalized Scientific-Summaries parquet (see ku_schema.py) ------
    parquet_dirs: tuple = tuple(os.environ.get(
        "PAPER_A_PARQUET", "data/normalized/arxiv").split(","))
    subsets: tuple = ()                # () = all; e.g. ("arxiv-new", "arxiv-old", "titled")
    max_papers: int = 60000            # tier: D0 60k, D1 500k, D2 2M (see TIERS)
    data_seed: int = 0                 # which shards / papers a tier draws
    rows_per_shard: int = 500          # random rows per shuffled shard (0 = whole shards)
    field_balance: str = "temperature" # domain-balanced corpus (ku_graph.select_balanced): temperature (share ~
                                       #   count^field_alpha) | uniform (equal) | cap (natural, at most field_cap
                                       #   per domain) | none (the earlier random rows)
    field_alpha: float = 0.5           # temperature exponent: 1 = natural shares, 0.5 = square root, 0 = equal
    field_cap: int = 0                 # optional maximum papers per domain (0 = no cap)
    field_pool: int = 4                # metadata-only candidates read per selected paper (first pass)
    train_domain_alpha: float = 1.0    # training anchors weighted by domain size^(alpha - 1) (1 = as in the corpus,
                                       #   0 = every domain equally often); an ablation
    primary_average: str = "macro"     # reasoning index used for model selection and findings: macro (mean over
                                       #   domains, each domain weighs the same) | micro (over all queries)
    min_domain_queries: int = 20       # a domain enters the macro average with at least this many test queries
    min_domains: int = 3               # a task is macro-averaged only over >= this many such domains (else micro)
    year_min: int = 1900
    year_max: int = 2100
    index_dir: str = os.environ.get("PAPER_A_INDEX_DIR", "runs/paper_a_index")
    out_dir: str = os.environ.get("PAPER_A_OUT", "runs/paper_a_results")
    model_dir: str = os.environ.get("PAPER_A_MODEL_DIR", "")   # trained models: <model_dir>/<exp>/<arm>_seed<s>/;
                                       # "" = <out_dir>/<exp>/models/
    synthetic: bool = False            # generated records in the parquet schema, for smoke tests
    ku_mode: str = "section"           # section: ~22 KUs per paper (7 sections + claim / evidence /
                                       #   implication KUs) | paper: the WHOLE paper is ONE KU (all
                                       #   sections + claims, ~2,100 tokens); same papers in both modes
                                       # | paper_masked: the target's paper is ONE text of the visible
                                       #   ("given") KUs, encoded together; targets = section KUs
    pm_templates: str = "full"         # paper_masked: "full" (3 reasoning queries + leave-one-out per
                                       #   type, ~13 texts / paper) | "tasks" (the 3 queries only)

    # ---- splits ---------------------------------------------------------------
    split_dir: str = os.environ.get("PAPER_A_SPLIT", "")   # a stored split (sci-jepa split, data/split.py): its
                                       #   train / val / test papers are used as they are (domain-balanced once, never
                                       #   re-sampled or re-split); max_papers then takes the same share of every split.
                                       #   Empty: papers are sampled and split here, as below
    split_mode: str = "time"           # time: train <= T, val T+1, test T+2 | random
    time_T: int = 2023                 # train <= 2023, validation 2024, test 2025 onwards (contamination control:
                                       #   every test paper is later than the text encoder's 2024 cutoff)
    test_open_ended: bool = True       # test = every year >= time_T + 2 (False: only time_T + 2)
    encoder_cutoff_year: int = 2024    # text encoder's training data (preset); warn when T precedes it
    split_seed: int = 42
    split: tuple = (0.8, 0.1, 0.1)     # random mode only

    # ---- knowledge units ------------------------------------------------------
    ku_min_chars: int = 20             # shorter, placeholder or null-statement text is not a KU
    optional_kus: tuple = ()           # e.g. ("figures",); robustness / data_code / ethics are sparse
    min_kus_per_paper: int = 8
    required_kus: tuple = ("result", "claim")
    embedder: str = "gte-modernbert"   # a key of EMBEDDERS (sets the fields below) | hash (smoke tests)
    embed_model: str = "Alibaba-NLP/gte-modernbert-base"   # 8192-token context, 768-d
    embed_max_tokens: int = 2048       # longest KU ~1150 tokens (results); MiniLM's 256 cut 93% of methods
    embed_batch: int = 32              # texts per forward pass (long sections: keep small on a laptop)
    embed_tokens: int = 8192           # padded tokens per forward pass (with embed_batch: the smaller limit wins)
    embed_prefix: str = ""             # some encoders want e.g. "search_document: "
    embed_chunk: int = 8192
    snippet_chars: int = 240           # KU text kept for the dashboard's example tables

    # ---- proxy grounded edges (stand-in until canonicalisation exists) ------
    proxy_types: tuple = ("method",)   # component types matched across papers
    proxy_tau: float = 0.86            # per encoder (EMBEDDERS): the same percentile of cross-paper top-1
                                       # method cosine as MiniLM's 0.80 (p99.3); the build prints the edge count
    proxy_topk: int = 10

    # ---- instance construction (the relation layer) -------------------------
    target_types: tuple = ("context", "question", "method", "procedure", "result", "interpretation",
                           "limitation", "claim", "evidence", "implication")   # all 10 (ALL_TARGET_TYPES)
    mask_schedule: str = "mixed"       # mixed | balanced | round_robin | sequential (see train.mask_plan)
    replay_frac: float = 0.0           # sequential only: share of each batch replaying earlier types
    epochs: float = 40.0               # >0: MAXIMUM run length = epochs full passes over every target KU
                                       # (overrides steps); sequential gives each type `epochs` passes.
                                       # With patience > 0 a run stops earlier (early stopping below)
    min_epochs: int = 10               # early stopping never stops before this epoch (small corpora learn late:
                                       #   500 papers jump after 5-30 epochs); use 1 at full-arXiv scale
    patience: int = 5                  # early stopping: stop when the validation score has not improved for
                                       #   this many epochs (0 = always run every epoch). Needs eval_at_epochs
    select_best: bool = True           # test with the weights of the best validation epoch, not the last
    select_metric: str = "reasoning_index"   # validation score for early stopping / best epoch:
                                       #   reasoning_index (mean of the 5 tasks, x chance) | retrieval
    n_subgraphs: int = 8               # N: E10 sweeps {1, 4, 8, 16}
    hops: int = 2                      # k-hop expansion over the paper graph
    fill_policy: str = "same_subfield" # keep N fixed when citations are sparse (~22% have refs)
    max_comp_per_paper: int = 24       # a paper has ~23 KUs (7 sections + ~4.3 x 3.5 claim KUs)
    mask_policy: str = "token+edges+children+siblings"  # siblings: method<->procedure restate each other
                                       # (same-paper cosine 0.78 vs 0.55 across papers)
    audit_tau: float = 0.87            # paraphrase-leak threshold, per encoder (EMBEDDERS): MiniLM 0.75,
                                       # gte-modernbert 0.87 (same percentile of same-paper similarity)
    audit_drop: bool = False           # drop instances the audit flags
    resample_each_epoch: bool = True   # E4: False = neighbourhoods frozen at epoch 0
    use_degree_features: bool = False  # E4: True = expose the node census to the encoder
    n_probe_edges: int = 4             # role edges hidden from the encoder for the inverse head
    task_mask_p: float = 0.5           # task-aware masking: share of training instances whose own paper shows
                                       #   only a reasoning template's given parts (forward / inverse / abduction)
                                       #   so training matches the five reasoning queries; 0 = leave-one-out only
    motif_p: float = 0.0               # E20 (Graph-JEPA-style): share of training instances whose hidden KU's
                                       #   whole motif (claim+evidence+implications, method+procedure,
                                       #   result+interpretation+limitation, context+question) is predicted
    rwse_steps: int = 0                # E20: random-walk structural encoding size (0 = off; Graph-JEPA uses 15-40)
    baseline: str = ""                 # comparison system instead of a JEPA model (baselines.py): text (frozen
                                       #   index encoder) | bm25 | scincl | bge | minilm | supervised (same-data MLP)
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"   # LLM-Reasoning baseline (baseline="llm"), any HF causal LM
    llm_score: str = "pmi"             # pmi: log p(candidate | paper) - log p(candidate | no paper) | loglik
    llm_dtype: str = "bfloat16"         # on CUDA
    llm_mps_dtype: str = "float32"     # on an Apple GPU (fp16 overflows in Qwen models; set from the check below)
    llm_batch: int = 8                 # candidates per forward pass (on top of the cached prompt)
    llm_ctx_tokens: int = 1536         # visible parts of the paper in the prompt (kept from the end)
    llm_cand_tokens: int = 256         # candidate text scored
    llm_eval_instances: int = 600      # queries per retrieval evaluation (each ranks ~100 candidates)
    llm_shape_bucket: int = 64         # prompt lengths rounded down to multiples of this (oldest context tokens
                                       #   dropped), candidates padded to multiples of 32, batches to llm_batch: the
                                       #   Apple GPU keeps buffers per tensor shape (0 = off)
    llm_allow_cpu: bool = False        # True: run the LLM arm without a CUDA GPU too (Apple GPU / CPU; fine for
                                       #   models up to ~2B, e.g. Qwen/Qwen2.5-1.5B-Instruct on a MacBook)
    llm_models: tuple = ()             # several LLM-Reasoning arms, one per model (Hugging Face ids): BENCH's
                                       #   LLM-Reasoning arm becomes "LLM-Reasoning@<model name>" per entry;
                                       #   () = the single arm with llm_model (run_experiment.expand_llm_arms)
    arms: tuple = ()                   # only these arm tags of each experiment (env ARMS, e.g. the full-corpus run's
                                       #   trained arms); "LLM-Reasoning" also selects every LLM-Reasoning@<model>;
                                       #   () = every arm (run_experiment.select_arms)
    # ---- long runs: resume a training run across jobs (train_one) ---------------
    resume_runs: bool = False          # True: every training run of run_experiment / the pipeline writes a
                                       #   checkpoint (model, optimizer, EMA target, sampler and RNG states,
                                       #   history, best epoch) and a later job continues it from there
    ckpt_every: int = 2000             # with resume_runs: also checkpoint every this many steps (0 = only at
                                       #   epoch ends); a job killed at its time limit loses at most this much
    encoder: str = "transformer"       # context / target encoder and predictor family (model.ENCODERS):
                                       #   transformer (relation-biased attention over every token pair) |
                                       #   rgat (attention only along graph edges; "gnn" = alias) | gine (GINE
                                       #   message passing with relation embeddings, Graph-JEPA's family) |
                                       #   gps (GraphGPS: GINE + global attention in every layer)
    wm_p_action: float = 0.0           # JEPA world model (E19): share of training instances that get an
                                       #   action (swap a KU / flip a supports-contradicts edge); the target
                                       #   is the hidden KU's latent AFTER the action. 0 = off (reference)

    # ---- model --------------------------------------------------------------
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 8
    pred_layers: int = 6
    dropout: float = 0.0
    feat_dim: int = 768                # node features = text-encoder dimension (preset)

    # ---- objective ----------------------------------------------------------
    jepa_loss: str = "l2"              # l2 | cos
    target_scope: str = "paper"        # paper: target encoder sees only the target's own subgraph
    lambda_inv: float = 0.0            # E11: inverse-relation (IDM analogue) weight
    lambda_reg: float = 0.0            # E12: grouping regulariser weight
    reg_grouping: str = "across"       # E12: within | across
    reg_projections: int = 64
    lambda_contrast: float = 0.0       # E5: relational contrast masking margin weight
    contrast_margin: float = 0.5
    lambda_align: float = 0.0          # contrastive alignment term (own it in the intro)
    align_tau: float = 0.1

    # ---- symbolic constraints (E1-E3) -----------------------------------------
    shape_spec: Optional[str] = None   # JSON spec; None = constraint loss off
    lambda_shacl: float = 0.0
    tnorm: str = "product"             # product | lukasiewicz | godel
    shape_depth: Optional[int] = None  # E2: replicate conjuncts to this depth
    shape_factored: bool = True        # E3: per-slot sub-vectors vs one opaque target
    shape_subset: str = "all"          # E1: all | tight | loose

    # ---- optimisation ---------------------------------------------------------
    batch_size: int = 128
    steps: int = 20000
    steps_per_epoch: int = 1000
    lr: float = 3e-4
    warmup: int = 1000                 # linear warm-up steps (the pipeline: up to one epoch, at most 10% of the run)
    lr_schedule: str = "cosine"        # after warm-up: cosine decay to lr_min over the run's step budget, as in
                                       #   I-JEPA / V-JEPA | "constant" (the earlier runs)
    lr_min: float = 1e-6
    weight_decay: float = 0.05
    ema_start: float = 0.996
    ema_end: float = 1.0
    grad_clip: float = 1.0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    device: str = "auto"               # auto: cuda > mps (Apple GPU) > cpu

    # ---- speed (same model and objective; see FAST and README "Training on a MacBook") ----
    attn_impl: str = "manual"          # manual | sdpa: fused attention kernel, identical maths (helps on CUDA,
                                       #   not on Apple GPUs)
    amp: str = "off"                   # off | fp16 | bf16: mixed-precision forward / backward; weights, loss,
                                       #   optimizer and EMA stay fp32 (fp16 uses a gradient scaler)
    prefetch: int = 0                  # >0: build the next training batches in a background thread while
                                       #   the GPU trains (same batches, same order)
    batch_workers: int = 0             # >0: build training batches in this many worker processes (scaling/README.md).
                                       #   The anchors are drawn in the main process as before; every step builds
                                       #   its instances from its own random stream (seed, step), so the batches
                                       #   do not depend on the number of workers and a resumed run is exact.
                                       #   Same model, objective and data distribution; another random stream
                                       #   than batch_workers=0
    tf32: bool = False                 # TF32 matrix products on Ampere+ GPUs (fp32 parts of the model)
    fused_adamw: bool = False          # fused (single-kernel) AdamW on CUDA: same update
    trainer: str = "native"            # native (train.py's own loop) | lightning (training/lightning_loop.py)
    devices: int = 1                   # trainer=lightning: GPUs for ONE run (DDP; the global batch is split)
    targets_per_instance: int = 1      # >1: multi-target training (I-JEPA-style): besides the anchor, hide one
                                       #   KU in each of k-1 neighbouring papers (leak-safe masked, own target
                                       #   view). An epoch stays one pass of PREDICTED KUs, so a run has k x
                                       #   fewer steps. NOT RECOMMENDED: in SPEED (500 papers, equal predicted
                                       #   KUs) k=4 was 4-6x faster but lost accuracy (reasoning index 3.36 vs
                                       #   4.25, intervention shift 0.04 vs 0.47). Kept for ablations
    embed_bucket: int = 512            # pad text-encoder batches to a multiple of this many tokens
                                       #   (64: 4.3x faster embedding, identical vectors)
    embed_dtype: str = "float32"       # float32 | float16: text-encoder precision (features are stored in fp16
                                       #   either way; float16 is 1.9x faster, cosine to fp32 >= 0.99999)

    # ---- evaluation -----------------------------------------------------------
    eval_instances: int = 4000
    eval_bank_papers: int = 0          # 0 = every test paper in the candidate bank
    balanced_k: int = 100              # candidates per query (besides the true KU)
    sim_threshold: float = 0.8         # JEPA models: a prediction counts as CORRECT when its cosine to the target
                                       #   encoder's vector of the true KU, both centred on the candidate bank's
                                       #   mean vector (evaluation/diagnostics.py: centred_cos), is >= this (reported as "accuracy (sim >=
                                       #   0.80)" next to MRR; comparison systems predict no vector: not scored)
    pool_mode: str = "topic"           # topic: distractors from the papers with the most similar problem
                                       # (shortcut-controlled) | subfield: the earlier protocol
    mp_filter: bool = True             # adversarial filtering (validity.audit): drop counterfactuals a context-free
                                       #   detector tells apart from the original by wording alone
    mp_filter_tau: float = 0.9         #   ... when it names the original with at least this probability,
    mp_kind_tau: float = 0.9           #   ... or every edit of a kind it detects in >= this share (e.g. negation)
    minimal_pairs: bool = True         # forward / abduction: add counterfactual variants of the true
                                       # result / claim (ku_schema.counterfactuals) to the candidates
    probe_papers: int = 200            # partition-identity probe
    log_every: int = 200
    eval_every: int = 0                # >0: quick validation pass every N steps (dashboard curves)
    live_eval_instances: int = 300
    neuron_report: bool = True         # per-neuron interpretation at every live eval (dashboard: Neurons)
    neuron_instances: int = 128
    eval_at_epochs: bool = True        # with epochs > 0: validation + neuron report at the end of EVERY epoch
    epoch_eval_instances: int = 1000   # queries per end-of-epoch validation (more = steadier per-type numbers)
    eval_tasks: bool = True            # reasoning-task matrix (tasks.py): forward / inverse / abduction /
                                       # consistency / intervention, at every epoch end (val) and at the end (test)
    task_eval_instances: int = 300     # queries per task
    save_model: bool = True            # write each trained model to <model_dir>/<exp>/<arm>_seed<s>/
    epoch_worldview: bool = True       # False: skip the world-model checks at epoch ends (faster; final test keeps them)
    eval_worldview: bool = True        # structure reliance + action prediction (every epoch end, val; final, test)
                                       # and explained example subgraphs for the dashboard's World model tab

    def override(self, **kw) -> "PaperAConfig":
        bad = [k for k in kw if not hasattr(self, k)]
        if bad:
            raise KeyError(f"unknown config keys: {bad}")
        explicit = set(kw)
        if kw.get("embedder") not in (None, self.embedder):   # switching encoder: its preset brings its
            name = kw["embedder"]                    # own dim / thresholds, unless set explicitly in this call
            if name not in EMBEDDERS:
                raise KeyError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
            kw = {**EMBEDDERS[name], **kw}
        mode = kw.get("ku_mode", self.ku_mode)
        if mode not in ("section", "paper", "paper_masked"):
            raise KeyError(f"unknown ku_mode {mode!r}; choose 'section', 'paper' or 'paper_masked'")
        emb = kw.get("embedder", self.embedder)
        if mode == "paper":                          # one KU per paper: its only type is "paper"
            kw["target_types"], kw["proxy_types"] = ("paper",), ("paper",)
            if self.ku_mode != "paper" or "embedder" in kw:   # entering paper mode: whole-paper settings
                kw = {**kw, **{k: v for k, v in PAPER_MODE.get(emb, PAPER_MODE["*"]).items() if k not in explicit}}
        elif mode == "paper_masked" and self.ku_mode != "paper_masked":
            kw.setdefault("embed_max_tokens", 8192)  # context texts reach ~4k tokens
            if self.ku_mode == "paper":
                kw.setdefault("target_types", ("context", "question", "method", "procedure", "result",
                                               "interpretation", "limitation", "claim", "evidence", "implication"))
                kw.setdefault("proxy_types", ("method",))
                kw = {**{k: v for k, v in EMBEDDERS.get(emb, {}).items() if k in ("proxy_tau", "audit_tau")}, **kw}
            if kw.get("pm_templates", self.pm_templates) == "tasks":
                tt = kw.get("target_types", self.target_types)
                kw["target_types"] = tuple(t for t in tt if t in ("result", "method", "procedure", "claim", "question"))
        elif self.ku_mode in ("paper", "paper_masked") and mode == "section":   # back to section mode
            kw.setdefault("target_types", self.target_types if self.target_types != ("paper",) else
                          ("context", "question", "method", "procedure", "result", "interpretation",
                           "limitation", "claim", "evidence", "implication"))
            kw.setdefault("proxy_types", ("method",))
            kw = {**{k: v for k, v in EMBEDDERS.get(emb, {}).items()
                     if k in ("embed_max_tokens", "proxy_tau", "audit_tau")}, **kw}
        if mode == "paper_masked" and kw.get("pm_templates") == "tasks" and "target_types" not in explicit:
            tt = kw.get("target_types", self.target_types)
            kw["target_types"] = tuple(t for t in tt if t in ("result", "method", "procedure", "claim", "question"))
        for k, v in kw.items():                      # "1000" -> 1000, "a,b" -> ("a", "b")
            cur = getattr(self, k)
            if isinstance(v, str) and not isinstance(cur, str) and cur is not None:
                if isinstance(cur, bool):
                    kw[k] = v.lower() in ("1", "true", "yes")
                elif isinstance(cur, (int, float)):
                    kw[k] = type(cur)(v)             # raises on garbage, which is the point
                elif isinstance(cur, (tuple, list)):
                    kw[k] = tuple(x for x in v.split(",") if x)
            elif isinstance(cur, tuple) and isinstance(v, list):
                kw[k] = tuple(v)
        return dataclasses.replace(self, **kw)

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)


# Text encoders for KU node features. Thresholds are cosine values, and every encoder has its own
# cosine scale (gte-modernbert: two random KUs ~0.53, MiniLM ~0.08), so each preset carries
# proxy_tau / audit_tau calibrated to the SAME percentiles on 600 real papers (13k KUs).
EMBEDDERS = {
    "gte-modernbert": dict(embed_model="Alibaba-NLP/gte-modernbert-base", embed_max_tokens=2048, feat_dim=768,
                           embed_batch=32, embed_prefix="", encoder_cutoff_year=2024,
                           proxy_tau=0.86, audit_tau=0.87),
    "nomic-modernbert": dict(embed_model="nomic-ai/modernbert-embed-base", embed_max_tokens=2048, feat_dim=768,
                             embed_batch=32, embed_prefix="search_document: ", encoder_cutoff_year=2024,
                             proxy_tau=0.86, audit_tau=0.87),   # not calibrated separately
    "bge-m3": dict(embed_model="BAAI/bge-m3", embed_max_tokens=2048, feat_dim=1024, embed_batch=16,
                   embed_prefix="", encoder_cutoff_year=2024, proxy_tau=0.86, audit_tau=0.87),  # not calibrated
    "minilm": dict(embed_model="sentence-transformers/all-MiniLM-L6-v2", embed_max_tokens=256, feat_dim=384,
                   embed_batch=256, embed_prefix="", encoder_cutoff_year=2021, proxy_tau=0.80, audit_tau=0.75),
    "hash": dict(proxy_tau=0.80, audit_tau=0.75),    # synthetic smoke tests; keeps feat_dim
}
# ku_mode="paper": whole-paper texts (median ~2,100 tokens, max ~4,200 on 600 real papers) and
# paper-to-paper thresholds. gte-modernbert: cross-paper top-1 cosine p99.3 = 0.84 (the percentile
# MiniLM's 0.80 had at section level); the leak flag uses the same value (a near-duplicate paper).
PAPER_MODE = {
    "gte-modernbert": dict(embed_max_tokens=8192, proxy_tau=0.84, audit_tau=0.84),
    "hash": dict(proxy_tau=0.80, audit_tau=0.75),
    "*": dict(embed_max_tokens=8192),                # other encoders: thresholds not calibrated
}


# ---------------------------------------------------------------- the ICML paper
# Every run uses the PaperAConfig defaults above (= the paper's configuration); an experiment
# changes ONE thing (its arms). Shown by the notebook as "the fixed configuration".
ICML_FIXED = {
    "KU graph": "each paper -> ~22 KUs: 7 sections + every claim / evidence / implication (ku_mode='section')",
    "text encoder": "gte-modernbert-base, 8192-token context, 768-d, no KU truncated",
    "split": "time split: train <= 2023, val 2024, test 2025-2026 (stored, domain-balanced split: sci-jepa split); "
             "every task also reported for test papers "
             "after the text encoders' pretraining cutoff (contamination check)",
    "hidden KU types": "all 10 (context, question, method, procedure, result, interpretation, limitation, "
                       "claim, evidence, implication)",
    "masking": "mixed (every type in every batch); leak-safe: the hidden KU's paraphrases are hidden too",
    "context": "N = 8 papers per training instance (E10 varies it)",
    "objective": "plain JEPA (L2 to an EMA target encoder); every extra loss off (E11, E12, E5, E1-E3 turn one on)",
    "training": "EPOCHS full passes over every hidden KU (60k run: 10, the same for every arm); AdamW 3e-4 with a "
                "one-epoch warm-up and cosine decay to 1e-6; validation every epoch, test with the best epoch's "
                "weights; optional early stopping (PATIENCE); batch 128 (64 for the local size)",
    "evaluation": "every epoch on val, final on test: balanced MRR vs chance, per KU type, reasoning-task "
                  "matrix (forward, inverse, abduction, consistency, intervention), neuron report",
}
PAPER_EXPERIMENTS = {                  # key -> the question it answers (in paper order)
    "BENCH": "benchmark: 6 comparison systems (frozen text, BM25, SciNCL, BGE-M3, supervised regressor, LLM-Reasoning), "
             "JEPA with 3 GNN encoders (R-GAT, GINE, GraphGPS) + Graph-JEPA, and the 5 JEPA graph-transformer variants "
             "(15 rows per seed; LLM-Reasoning needs a CUDA GPU)",
    "ALL": "every distinct method of every experiment above (duplicates removed), one leaderboard (67 runs per seed)",
    "APPX_STRUCT": "appendix: random-walk structural encodings, alone and with motif + world model (4 runs)",
    "APPX_EVAL": "appendix: old vs redesigned evaluation (candidates, minimal pairs, task-aware masking) (5 runs)",
    "SPEED": "MacBook speed settings: same accuracy as the reference? (fast arithmetic, multi-target training) (6 runs)",
    "E10": "A1 degeneracy: does widening the context (N = 1, 4, 8, 16 papers) remove the identity shortcut?",
    "E11": "repair: inverse-relation head (lambda_inv = 0, 0.1, 0.3, 1)",
    "E12": "repair: grouping regulariser (off / within a paper / across papers)",
    "E4":  "repair: graph construction (frozen neighbourhoods + node census)",
    "E5":  "repair: relational contrast masking (off / on)",
    "E15": "which KU types are learnable: one model masking every type vs one model per type (11 runs)",
    "E17": "how to read a paper: separate KUs vs ONE text of its visible parts (2 runs)",
    "E18": "E15 with the paper read as one text: one model for every part vs one per part (11 runs)",
    "E19": "JEPA world model: learning how actions on the graph (swap a KU, flip an edge) change the hidden KU, "
           "without losing clean accuracy (3 runs)",
    "E20": "Graph-JEPA ideas: sub-graph (motif) targets + random-walk structural encodings, alone and with "
           "the world model, at unchanged clean accuracy (6 runs)",
    "E1":  "shape constraints: SHACL off / loose / tight (Theorem A2) [placeholder shapes until ORKG]",
    "E2":  "t-norm depth decay: product / Lukasiewicz / Goedel x depth 3, 8, 15 (Prop. A3) [placeholder shapes]",
    "E3":  "shape-factored heads vs one head x depth [placeholder shapes]",
}


def resolve_device(name):
    """'auto' -> cuda, else Apple 'mps', else cpu. An unavailable request falls back."""
    import torch
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if name in ("auto", "cuda", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Speed settings that keep the model and objective unchanged (measured on an M5 Pro, see README
# "Training on a MacBook"): fp16 mixed precision (1.1x local, 1.5x S), batches built in a background
# thread while the GPU trains, and 8.4x faster text embedding with the same vectors (cosine >= 0.99999).
FAST = dict(amp="fp16", prefetch=2, embed_bucket=64, embed_dtype="float16")
# FAST on an Ampere / Hopper GPU (A100, H100): bf16 instead of fp16 (no loss scaling), fused attention kernel, TF32
# and fused AdamW. Same model and objective (scaling/README.md has the measurements)
FAST_GPU = dict(FAST, amp="bf16", attn_impl="sdpa", tf32=True, fused_adamw=True)
# + multi-target training (targets_per_instance): a training change. The SPEED experiment showed it LOSES
# accuracy at an equal number of predicted KUs, so it is not used; kept to reproduce that ablation
FAST_MULTI = dict(FAST, targets_per_instance=4)
SLOW = dict(amp="off", prefetch=0, targets_per_instance=1)   # the original code path
_R10 = dict(epochs=10.0, patience=0, select_best=False)       # the SPEED runs' fixed 10-epoch protocol

# Data tiers from the plan (papers drawn at random across shards).
TIERS = {"D0": dict(max_papers=60000), "D1": dict(max_papers=500000), "D2": dict(max_papers=2000000)}

# Model sizes from the plan. S matches D0; M/L need the D1/D2 corpora.
SIZES = {
    "tiny": dict(d_model=64, n_heads=4, enc_layers=2, pred_layers=1),
    "local": dict(d_model=256, n_heads=8, enc_layers=4, pred_layers=2),   # MacBook-sized
    "S": dict(d_model=512, n_heads=8, enc_layers=8, pred_layers=6),
    "M": dict(d_model=768, n_heads=12, enc_layers=12, pred_layers=8),
    "L": dict(d_model=1024, n_heads=16, enc_layers=16, pred_layers=12),
}

# Each experiment is a list of (tag, overrides). The first entry is the
# reference arm that paired tests compare against.
EXPERIMENTS = {
    # BENCH: the benchmark. Every main model variant + the no-training text baseline, one row each,
    # same papers, splits, candidate pools and evaluation; compared in the dashboard's Benchmark tab.
    # Comparison systems first (baselines.py): lexical, two frozen encoders besides the index's, and a
    # supervised regressor on the same data. Then JEPA with graph neural network encoders (relational
    # graph attention, GINE, GraphGPS; Graph-JEPA = GINE + motif / sub-graph targets) and the JEPA
    # variants with the graph transformer. (An LLM ranker needs a GPU server and is not included.)
    "BENCH": [("text_baseline", dict(baseline="text")),
              ("bm25", dict(baseline="bm25")),
              ("scincl", dict(baseline="scincl")),
              ("bge-m3", dict(baseline="bge-m3")),        # long-context encoder (8,192 tokens)
              ("supervised", dict(baseline="supervised")),
              ("LLM-Reasoning", dict(baseline="llm")),     # GPU server only (skipped on a MacBook)
              ("jepa+rgat", dict(encoder="rgat")),
              ("jepa+gine", dict(encoder="gine")),
              ("jepa+gps", dict(encoder="gps")),
              ("graph_jepa_gine", dict(encoder="gine", motif_p=0.5)),
              ("reference", dict()),
              ("paper_as_text", dict(ku_mode="paper_masked")),
              ("world_model", dict(wm_p_action=0.25)),
              ("motif", dict(motif_p=0.5)),
              ("motif+world", dict(motif_p=0.5, wm_p_action=0.25))],
    # Appendix A: random-walk structural encodings (Graph-JEPA), kept out of BENCH: on 500 papers
    # they changed nothing and cost ~17x training time
    "APPX_STRUCT": [("reference", dict()),
                    ("rwse", dict(rwse_steps=16)),
                    ("motif+world", dict(motif_p=0.5, wm_p_action=0.25)),
                    ("motif+world+rwse", dict(rwse_steps=16, motif_p=0.5, wm_p_action=0.25))],
    # Appendix B: what the evaluation redesign changes. old_protocol = subfield candidates, no
    # minimal pairs, no task-aware masking (the first BENCH run's setting)
    "APPX_EVAL": [("text_baseline", dict(baseline="text")),
                  ("text_baseline_old", dict(baseline="text", pool_mode="subfield", minimal_pairs=False)),
                  ("reference", dict()),
                  ("reference_no_task_mask", dict(task_mask_p=0.0)),
                  ("reference_old_protocol", dict(task_mask_p=0.0, pool_mode="subfield", minimal_pairs=False))],
    # SPEED: do the speed settings keep accuracy? reference vs FAST (same model, faster arithmetic) vs
    # multi-target training (4 hidden KUs per instance: the same predicted KUs in 4x fewer steps, or the
    # same number of steps), also for the best variant (motif + world model)
    # (fast_40_epochs = the reference trained on as many predicted KUs as fast+multi4_same_steps)
    "SPEED": [("reference", {**SLOW, **_R10}),
              ("fast", {**FAST, **_R10}),
              ("fast+multi4", {**FAST_MULTI, **_R10}),
              ("fast+multi4_same_steps", {**FAST_MULTI, **_R10, "epochs": 40.0}),
              ("fast_40_epochs", {**FAST, **_R10, "epochs": 40.0}),
              ("motif+world", {**SLOW, **_R10, "motif_p": 0.5, "wm_p_action": 0.25}),
              ("motif+world fast+multi4", {**FAST_MULTI, **_R10, "motif_p": 0.5, "wm_p_action": 0.25}),
              ("motif+world fast+multi4_same_steps", {**FAST_MULTI, **_R10, "motif_p": 0.5, "wm_p_action": 0.25,
                                                      "epochs": 40.0}),
              ("motif+world fast_40_epochs", {**FAST, **_R10, "motif_p": 0.5, "wm_p_action": 0.25, "epochs": 40.0})],
    "E10": [(f"N={n}", dict(n_subgraphs=n)) for n in (1, 4, 8, 16)],
    "E11": [(f"inv={w}", dict(lambda_inv=w)) for w in (0.0, 0.1, 0.3, 1.0)],
    "E12": [("reg=off", dict(lambda_reg=0.0)),
            ("reg=within", dict(lambda_reg=1.0, reg_grouping="within")),
            ("reg=across", dict(lambda_reg=1.0, reg_grouping="across"))],
    "E4": [("frozen+census", dict(resample_each_epoch=False, use_degree_features=True)),
           ("resample+census", dict(resample_each_epoch=True, use_degree_features=True)),
           ("frozen+nocensus", dict(resample_each_epoch=False, use_degree_features=False)),
           ("resample+nocensus", dict(resample_each_epoch=True, use_degree_features=False))],
    "E5": [("contrast=off", dict(lambda_contrast=0.0)),
           ("contrast=on", dict(lambda_contrast=1.0))],
    # E15 = mixed (all types together) next to one model per type: the same budget of
    # epochs each, so the dashboard's All runs tab compares mixed vs each type alone, per type
    "E15": [("mixed", dict(mask_schedule="mixed"))]
           + [(f"only={t}", dict(target_types=(t,), mask_schedule="mixed"))
              for t in ("context", "question", "method", "procedure", "result", "interpretation",
                        "limitation", "claim", "evidence", "implication")],
    # E17: the target's own paper as separate KUs (section) vs as ONE joint text of the visible
    # parts (paper_masked); same papers, same targets, same neighbours, same reasoning-task matrix
    "E17": [("section", dict(ku_mode="section")), ("paper_masked", dict(ku_mode="paper_masked"))],
    # E18 = E15 with the paper read as ONE text (ku_mode="paper_masked"): one model masking every
    # part (mixed) + one model per masked part; compare with E15 (the same in section mode)
    "E18": [("pm_mixed", dict(ku_mode="paper_masked", mask_schedule="mixed"))]
           + [(f"pm_only={t}", dict(ku_mode="paper_masked", target_types=(t,), mask_schedule="mixed"))
              for t in ("context", "question", "method", "procedure", "result", "interpretation",
                        "limitation", "claim", "evidence", "implication")],
    # E19: JEPA world model. A quarter of training instances get an action (swap a KU for another
    # paper's / flip a supports-contradicts edge) and the target is the hidden KU's latent AFTER the
    # action. Accuracy is always measured on clean instances, so it is directly comparable.
    "E19": [("reference", dict(wm_p_action=0.0)),
            ("world_model", dict(wm_p_action=0.25)),
            ("world_model+inverse", dict(wm_p_action=0.25, lambda_inv=0.3))],
    # E20: Graph-JEPA ideas for the KU graph (Skenderi et al.): sub-graph (motif) targets and random-
    # walk structural encodings (encoder input + the hidden nodes' true position for the predictor).
    # Motifs are used in training only; evaluation is the reference protocol, so accuracy compares 1:1.
    "E20": [("reference", dict()),
            ("rwse", dict(rwse_steps=16)),
            ("motif", dict(motif_p=0.5)),
            ("graph_jepa", dict(rwse_steps=16, motif_p=0.5)),
            ("graph_jepa+world", dict(rwse_steps=16, motif_p=0.5, wm_p_action=0.25)),
            ("motif+world", dict(motif_p=0.5, wm_p_action=0.25))],
    "E1": [("shacl=off", dict(lambda_shacl=0.0)),
           ("shacl=loose", dict(lambda_shacl=1.0, shape_subset="loose")),
           ("shacl=tight", dict(lambda_shacl=1.0, shape_subset="tight"))],
    "E2": [(f"{t}@{d}", dict(lambda_shacl=1.0, tnorm=t, shape_depth=d))
           for t in ("product", "lukasiewicz", "godel") for d in (3, 8, 15)],
    "E3": [(f"factored={f}@{d}", dict(lambda_shacl=1.0, shape_factored=f, shape_depth=d))
           for d in (3, 15) for f in (False, True)],
}


def _all_arms():
    """ALL: every distinct arm of every experiment (BENCH first), tagged <experiment>__<arm>; an arm
    whose overrides (ignoring values equal to the defaults) were already listed is skipped. SPEED is
    left out: it compares speed settings of one model, not methods."""
    base, seen, arms = PaperAConfig(), set(), []
    for exp, lst in EXPERIMENTS.items():
        if exp in ("SPEED", "ALL"):
            continue
        for tag, over in lst:
            eff = {k: v for k, v in over.items() if getattr(base, k) != (tuple(v) if isinstance(v, list) else v)}
            key = tuple(sorted((k, repr(v)) for k, v in eff.items()))
            if key in seen:
                continue
            seen.add(key)
            arms.append((f"{exp}__{tag}", dict(over)))
    return arms


EXPERIMENTS["ALL"] = _all_arms()


# ---------------------------------------------------------------- method catalogue (dashboard, PDF report)
_BASELINE_TEXT = {
    "text": "no training: frozen text encoder (gte-modernbert), mean embedding of the visible KUs, cosine to candidates",
    "bm25": "no training: lexical BM25 ranking of the candidates against the visible KUs' words",
    "scincl": "no training: frozen scientific-paper encoder (SciNCL; texts over 512 tokens chunked and averaged), same protocol",
    "bge": "no training: frozen general-purpose encoder (bge-base-en-v1.5), long texts chunked, same protocol",
    "bge-m3": "no training: frozen long-context retrieval encoder (BGE-M3, 8,192 tokens), same protocol as the text baseline",
    "minilm": "no training: small frozen encoder (MiniLM-L6), same protocol as the text baseline",
    "supervised": "supervised MLP trained on the same data to map visible KUs to the hidden KU's embedding "
                  "(no graph, no JEPA)",
    "llm": "no training: an open LLM (Qwen2.5-7B-Instruct by default) ranks the candidates by calibrated "
           "log-likelihood given the visible parts; zero-shot consistency",
}
_ENCODER_TEXT = {
    "rgat": "graph neural network encoder: attention only along graph edges (relational graph attention, R-GAT)",
    "gnn": "graph neural network encoder: attention only along graph edges (relational graph attention, R-GAT)",
    "gine": "graph neural network encoder: GINE message passing with relation embeddings (Graph-JEPA's family)",
    "gps": "graph neural network encoder: GraphGPS (GINE message passing + global attention in every layer)",
}


def method_family(over):
    if over.get("baseline"):
        return "comparison"
    return "gnn" if over.get("encoder", "transformer") in _ENCODER_TEXT else "jepa"


def describe(over):
    """Plain-language description of what an arm changes relative to the reference JEPA model
    (graph transformer, task-aware masking, plain JEPA objective)."""
    if over.get("baseline"):
        b = over["baseline"]
        s = _BASELINE_TEXT.get(b, f"comparison system {b}")
        if b == "llm" and over.get("llm_model"):
            s = s.replace("Qwen2.5-7B-Instruct by default", over["llm_model"])
        if over.get("pool_mode") == "subfield" or over.get("minimal_pairs") is False:
            s += "; old evaluation (subfield candidates, no counterfactual distractors)"
        return s
    d, base = [], PaperAConfig()
    enc = over.get("encoder", "transformer")
    d.append(_ENCODER_TEXT.get(enc, "graph transformer encoder (relation-biased attention over every token pair)"))
    if over.get("ku_mode") == "paper_masked":
        d.append("the own paper read as ONE joint text of its visible parts (paper-as-text)")
    if over.get("motif_p"):
        d.append(f"motif / sub-graph targets: a KU's whole family predicted together ({over['motif_p']:.0%} of instances)")
    if over.get("wm_p_action"):
        d.append(f"world model: actions (swap a KU, flip supports/contradicts) in {over['wm_p_action']:.0%} of instances")
    if over.get("rwse_steps"):
        d.append(f"random-walk structural encodings ({over['rwse_steps']} steps)")
    if "task_mask_p" in over and over["task_mask_p"] == 0:
        d.append("no task-aware masking (trained only with everything else visible)")
    if over.get("pool_mode") == "subfield" or over.get("minimal_pairs") is False:
        d.append("old evaluation (subfield candidates, no counterfactual distractors)")
    if "n_subgraphs" in over and over["n_subgraphs"] != base.n_subgraphs:
        d.append(f"context of {over['n_subgraphs']} paper(s) instead of {base.n_subgraphs}")
    if over.get("lambda_inv"):
        d.append(f"relation decoder trains the encoder too (inverse loss weight {over['lambda_inv']})")
    if over.get("lambda_reg"):
        d.append(f"grouping regulariser ({over.get('reg_grouping', base.reg_grouping)} papers)")
    if "resample_each_epoch" in over or "use_degree_features" in over:
        d.append(("neighbourhoods frozen" if over.get("resample_each_epoch") is False else "neighbourhoods resampled")
                 + (", node census shown" if over.get("use_degree_features") else ", no node census"))
    if over.get("lambda_contrast"):
        d.append("relational contrast: predictions for minimal-pair contexts must differ")
    if "target_types" in over:
        d.append("only one KU type hidden: " + ", ".join(over["target_types"]))
    if over.get("lambda_shacl"):
        bits = [f"SHACL shape constraints ({over.get('shape_subset', 'all')} shapes"]
        if over.get("tnorm"):
            bits.append(f"{over['tnorm']} t-norm")
        if over.get("shape_depth"):
            bits.append(f"depth {over['shape_depth']}")
        if "shape_factored" in over:
            bits.append("factored heads" if over["shape_factored"] else "one head")
        d.append(", ".join(bits) + "; placeholder shapes)")
    if len(d) == 1:
        d[0] += "; the reference setting" if enc == "transformer" else "; otherwise the reference setting"
    return "; ".join(d)
