<!-- Moved from SCI-JEPA-training/Analysis/train/paper_a/README.md; module paths updated to the sci_jepa package. -->
# Paper A — Constraint-Grounded Latent Prediction (Track A code)

Self-contained package. It reads the normalized Scientific-Summaries parquet directly
(`Scientific-Summaries-Data/normalized/<subset>/*.parquet`, the 44-key schema written by
`parquet_to_json.py`) and depends on nothing else in the repo.

```
python -m sci_jepa.smoke_test                              # CPU, synthetic records, ~3 min (--all: all 66 methods)
python -m sci_jepa.training.bench_all --data smoke                  # every method -> live dashboard + PDF report
python -m sci_jepa.training.bench_all --data real --papers 300      # the same on a small slice of real summaries
python -m sci_jepa.reporting.report_pdf runs/paper_a_results/BENCH        # PDF report of any finished experiment
python -m sci_jepa.evaluation.findings runs/paper_a_results/BENCH          # pre-registered hypotheses + conclusion
python -m sci_jepa.reporting.figures runs/paper_a_results/BENCH           # paper figures (PDF + PNG) + LaTeX / CSV tables
python -m sci_jepa.evaluation.validity audit --set max_papers=500     # counterfactual surface-cue audit (also faithfulness,
                                                                #   export, agreement, contamination)
python -m sci_jepa.data.ku_graph --set max_papers=60000         # build + cache the D0 KU graph
python -m sci_jepa.training.train --audit-shapes                    # Theorem A2 audit, no training
python -m sci_jepa.training.train --exp E10 --size S                # one ablation, 5 seeds, paired tests
python -m sci_jepa.training.train --size S --set n_subgraphs=8 lambda_inv=0.3   # one config
bash cluster/grete/run_paper_a.sh submit dist                   # cluster (docs/cluster_grete.md)
```

`--set` takes any field of `PaperAConfig` (`config.py`). Results go to
`$PAPER_A_OUT/<exp>/`: one JSON per arm and seed, plus `summary.json` (mean, std, and a paired
t-test against the first arm on every key metric).

## Live dashboard and graph views

`telemetry.py` streams each run to `<dashboard>/runs/<name>/live.json`. `dashboard.html`
(copied to `<dashboard>/index.html` by `serve.setup_dashboard`) polls it every 2 s.
`graph_viz.py` writes animated vis-network pages (paper subgraph, training instance step by
step, corpus graph). `serve.py` serves the folder on localhost, because browsers block live
fetches from `file://`. Enable streaming with
`train_one(..., live=LiveRun(root, name, cfg, gi))` or `run_experiment(..., live_root=root)`;
set `eval_every` for live validation curves. With `neuron_report` (default on) each validation
pass also runs `neurons.py`, which labels every encoder neuron and predictor output by its
bias-corrected variance share (ω²) for KU type, paper and field and by its within-type
correlation with the paraphrase-leak flag: dead / identity / type / field / leak / content.
A final report runs on the test split. Dashboard tabs: Overview, Explain (rule-based
plain-language findings plus glossary), Training dynamics, Validation, Neurons, Where it learns,
Examples, Graphs, All runs; the x-axis is epoch or step. `paper_a_local.ipynb` wires all of this together.

## Saved models and inference

Every training run writes a model folder (`save_model`, on by default): the single notebook run to
`<model_dir>/single/<run>/`, experiment runs to `<model_dir>/<exp>/<arm>_seed<s>/` (`model_dir` = `PAPER_A_MODEL_DIR`; on Grete `SCI-JEPA/models/<run folder>`; unset = `<out_dir>/<exp>/models/`).
It holds `model.safetensors` (end of training, every module including the EMA target encoder),
`best.safetensors` (best end-of-epoch validation), `config.json` (the full configuration, so the
identical model is rebuilt) and `summary.json` (test metrics, text encoder, data fingerprint).
`inference.SciJEPA.from_pretrained(folder)` loads it with the same frozen text encoder and offers
`embed` (facet embeddings), `predict` (hidden-part representation, optionally under a reasoning
template), `rank` (candidate texts for a hidden part) and `consistency` (supports vs contradicts) for
any paper given as a structured summary. Pass a corpus index (`from_pretrained(folder, index=gi)` or
`attach_index(gi)`) to use the evaluation's context: the `n_subgraphs - 1` most similar training papers
join as neighbouring papers, `rank` encodes every candidate inside the query paper (in the hidden KU's
place, as the evaluation encodes the true KU and its counterfactuals), and `consistency(claim, evidence,
paper=...)` reads the pair inside its paper. Without an index (or with `rank(in_context=False)`) each
text is encoded on its own, the earlier lighter behaviour.

## Training on a MacBook (speed settings)

Speed switches in `PaperAConfig`, all off by default (the original code path). `config.FAST` turns on
the ones that keep the model and objective unchanged; the notebook (`SPEED = "fast"`) and
`scripts/run_local.sh` (`SPEED=fast`, the default) use it.

| Setting | What it does | Effect on results |
|---|---|---|
| `embed_bucket=64` | pads text-encoder batches (sorted by length) to multiples of 64 tokens instead of 512 | identical vectors (cosine 1.000000); 4.3x faster embedding |
| `embed_dtype="float16"` | runs the text encoder in fp16 (features were always stored in fp16) | cosine to fp32 >= 0.99999, 99.9% of top-10 neighbours unchanged; 1.9x faster. Own index cache |
| `amp="fp16"` | mixed-precision forward / backward; weights, losses, optimizer and EMA stay fp32; gradient scaler | predictions cosine >= 0.99999 to fp32 |
| `prefetch=2` | builds the next batches in a background thread while the GPU trains | bit-identical run (smoke test checks it) |
| `targets_per_instance=4` | **multi-target training** (`config.FAST_MULTI`, not part of `FAST`): besides the anchor, one hidden KU in each of 3 neighbouring papers, leak-safe masked, each with its own target view (I-JEPA-style). An epoch stays one pass of predicted KUs, so a run has 4x fewer steps | **loses accuracy** (below): not recommended |
| `attn_impl="sdpa"` | fused attention kernel, same maths | no gain on Apple GPUs (useful on CUDA) |

Also: embedding is **resumable** (every 8,192 texts are saved under `paper_a_data/embed_cache_<fingerprint>/`,
so a long build that stops continues where it stopped), and no step makes a full fp32 copy of the KU
features any more (34 GB at 500k papers).

**Measured on an M5 Pro (20-core GPU, 64 GB), 500-paper index, batch 64** (predicted KUs per second):

| Size | parameters | reference | `FAST` | `FAST_MULTI` (not recommended) |
|---|---|---|---|---|
| local (small) | 5.2 M | 209 | 256 (1.2x) | 914 (4.4x) |
| S | 45.6 M | 60 | 92 (1.5x) | 349 (5.8x) |
| M (medium) | ~130 M | 24 | 42 (1.8x) | 164 (6.9x) |

In a whole 10-epoch run of the small model (500 papers) `FAST` cut the training loop from 5.2 to
4.1 min; the final evaluation (~45 s) is not sped up.

Text embedding: 37.4 ms per KU before, 4.45 ms after (8.4x). What did *not* help on the Apple GPU:
`torch.compile` (slower), padding instances to a fixed length (slower), the fused attention kernel,
one-hot relation biases (slower than the embedding lookup).

**Accuracy check (`--exp SPEED`, 500 papers, size local, test split).**

| Arm | seeds | retrieval x chance | reasoning index | abduction x | intervention shift | training loop |
|---|---|---|---|---|---|---|
| reference (10 epochs) | 3 | 2.32 ± 0.19 | 1.67 ± 0.13 | 1.85 | 0.12 | 5.2 min |
| `FAST` (10 epochs) | 3 | 2.32 ± 0.19 | 1.67 ± 0.13 | 1.85 | 0.12 | 4.1 min |
| `FAST` 40 epochs | 2 | 10.1 | 4.25 | 8.1 | 0.47 | 17-29 min |
| `FAST_MULTI` 10 epochs (same predicted KUs as 10, 4x fewer steps) | 3 | 1.23 | 1.26 | 1.10 | 0.00 | 1.1 min |
| `FAST_MULTI` 40 epochs (same predicted KUs as `FAST` 40, same steps as 10) | 3 | 8.65 | 3.27 | 5.08 | 0.04 | 4.3 min |

`FAST` matches the reference on every metric (differences < 0.005, p >= 0.14): use it. Multi-target
training reaches a lower plateau at the same number of predicted KUs (reasoning index 3.3 vs 4.25,
intervention response almost gone): faster, but it costs accuracy, so it is off. Note that on this
500-paper index the reference was still improving after 10 epochs (1.67 at 10, 4.25 at 40 epochs).

**Several Macs.** One run uses one GPU; the models are small, so splitting a run across machines would
cost more in gradient traffic than it saves. Split the *runs* instead: with a shared `PAPER_A_OUT`
(network drive or synced folder), `SHARD=0/2 EXPS=BENCH bash scripts/run_local.sh` on one Mac and `SHARD=1/2`
on the other (`--shard i/n` in `train.py`) each train every second arm x seed; a last call without
`SHARD` loads every finished run and writes `summary.json`.

**Size of a MacBook run** (FAST, from the measured rates; full arXiv needs more disk than the 89 GB free):

| Corpus | KUs | text embedding | features | one pass, local / S / M (`FAST`) |
|---|---|---|---|---|
| 5,000 papers | 0.11 M | 8 min | 0.2 GB | 7 min / 20 min / 44 min |
| 60,000 (D0) | 1.3 M | 1.6 h | 2 GB | 1.4 h / 3.9 h / 8.6 h |
| 500,000 (D1) | 11 M | 14 h | 17 GB | 12 h / 33 h / 73 h |
| full arXiv, 3.3 M | 73 M | 3.8 days | 112 GB (external SSD; features must also be memory-mapped, not yet implemented, since RAM is 64 GB) | 3.3 / 9.2 / 20 days |

One pass = every KU predicted once (KUs / predicted KUs per second); evaluation and the graph build
come on top. Embedding is the same for every size and is done once.

## KU modes

| `ku_mode` | The hidden unit | How the model sees the hidden unit's own paper |
|---|---|---|
| `section` (default) | one section / claim / evidence / implication KU | the paper's other KUs as separate tokens (leak-safe masking) |
| `paper_masked` | the same KU as `section` | ONE text made of the visible KU types, joined and encoded together |
| `paper` (not in the paper) | a whole paper | not at all: predicted from neighbouring papers only |

`paper_masked` uses query templates (`ku_schema.TEMPLATES`): the three reasoning queries (forward,
inverse, abduction) and `all-<t>` = everything except `t` and the KUs that restate it
(`LEAK_FAMILY`). Training draws a random template that hides the target type; evaluation uses
`all-<t>` and the task templates. The hidden KU, its candidate pool and the neighbouring papers
are identical to `section`, so **E17** (section vs paper_masked) isolates one question: is it
better to read a paper as one text or as a graph of separate KUs? Each paper needs ~13 extra
encodings (`pm_templates="full"`) or 3 (`"tasks"`). For interventions, every val / test paper also
gets counterfactual forward texts with the procedure or method of up to 3 donor papers.
The dashboard's **Compare** tab shows two runs side by side (default: the arms of E17).

## Reasoning-task evaluation matrix

`tasks.py` queries the trained model five ways, at the end of every epoch (validation) and at the
end of training (test). It is on by default (`eval_tasks`, `task_eval_instances`). The parquet has
no separate Problem / Material / Condition fields, so the slots are mapped to KU types:
Problem = context + question, Method = method, Material / Condition = procedure, Result = result,
Claim / Hypothesis = claim (+ question). "given" = only these KU types stay visible in the target's
own paper; neighbouring papers are unchanged.

| Task | Query | Score |
|---|---|---|
| Forward prediction | mask result \| given problem, method, procedure | balanced MRR ÷ chance |
| Experiment design (inverse) | mask method / procedure \| given problem, result | balanced MRR ÷ chance |
| Abduction | mask claim / question \| given result, procedure | balanced MRR ÷ chance |
| Consistency checking | hide claim–evidence edges \| inverse head: supported_by vs challenged_by | balanced accuracy (chance 0.5) |
| Intervention conditioning | forward query with the procedure (condition) or method swapped for a donor paper's | relative shift of the prediction; share moving toward the donor's result |

Limits, stated in the output: there are no supports / contradicts edges *between* papers in the
data, so consistency is within-paper only; the model has no `z_do` input, so interventions are
input swaps. Not applicable with `ku_mode="paper"`.

**Counterfactual distractors** (`minimal_pairs`, built with the index, `ku_schema`): every validation / test
result, claim, method and procedure gets rule-based edits of its own text that keep topic and wording but
change the meaning: direction reversed, negated, number changed, and for methods / procedures (experiment
design) a **substitution**: up to two key technical terms (models, datasets, materials: BERT -> GPT-2) or one
sentence replaced by those of the same section of the topic-matched paper. They join the candidates;
`mp_pair` = how often the true KU outranks each edit (chance 0.5), `mp_by_kind` per kind of edit. A sample
must be checked by experts before publication. With `paper_masked` the given parts are the text the model reads, and interventions re-encode that text with the donor's section. Dashboard: *Reasoning tasks* tab, *All runs* →
"Reasoning tasks: every run".

## Text encoder

KU node features come from a frozen sentence encoder chosen by `config.embedder` (presets in
`config.EMBEDDERS`). The default, `gte-modernbert` (Alibaba-NLP/gte-modernbert-base, 768-d),
has an 8,192-token context and `embed_max_tokens = 2048`. The longest KU measured is ~1,150
tokens (a `result`), so no KU is truncated. The old `minilm` (256 tokens) cut 93% of method, 85% of
procedure and 78% of result KUs. Cosine scales differ between encoders (two random KUs:
gte-modernbert ~0.53, MiniLM ~0.08), so each preset carries its own `proxy_tau` / `audit_tau`.
For gte-modernbert these are 0.86 / 0.87, the same percentiles as MiniLM's 0.80 / 0.75, measured on
600 real papers. The leak bins in the evaluation are relative to `audit_tau`. The index build
prints how many texts were truncated.

## Knowledge units (KUs)

Defined in `ku_schema.py`, measured on arxiv-new, arxiv-old and titled shards (~23 KUs per paper):

| KU type | Source | Per paper |
|---|---|---|
| context, question, method, procedure, result, interpretation, limitation | `research_context`, `research_question_hypothesis`, `methodological_details`, `procedures_architectures`, `key_results`, `interpretation_implications`, `contradictions_limitations` | 1 each (~100% filled, 900–1,600 chars) |
| claim | `claims[].description` (`claims` is a JSON string) | ~4.3 (3–6) |
| evidence | `claims[].supporting_evidence`, `claims[].contradicting_evidence` (the edge says which) | ~4.3 supporting + ~1.5 contradicting |
| implication | `claims[].implications` | ~4.3 |
| figures, robustness, data_code, ethics (off by default) | `key_figures_tables`, `robustness_ablation_notes`, `data_code_availability`, `ethical_considerations` | sparse, placeholder-heavy |

- **Never KUs:** `executive_summary` and `three_takeaways` restate the units we mask.
- **Cleaning:** missing values, placeholders ("N/A", "Not applicable.", "No links provided.") and
  text under `ku_min_chars` are not KUs. `contradicting_evidence` is filtered sentence by sentence
  for null statements ("The paper does not present any contradicting evidence." and ~50
  variants), keeping any real caveat after them. On real shards: 36% of claims have real
  counter-evidence, 42% have none, 18.5% only a null statement, 3.4% a placeholder. Per-type
  outcomes are saved in `build_stats.ku_outcomes`.
- **Intra-paper role edges are imposed by the schema**, not extracted:
  context→question→method→{procedure, result}, result→claim, claim→{evidence, implication},
  result→{interpretation, limitation}. The paper must describe them that way.
- **Paper metadata:** `field_subfield` is split into a coarse field (hub embedding) and a fine
  subfield (hard negatives, neighbourhood fill). Year comes from `source_year`, then `oa_year`,
  then the arXiv id. Citations come from `openalex_id` + `oa_referenced_works`; only ~22% of
  papers have references, so neighbourhoods are filled from the same subfield.
- **Tiers:** D0 60k, D1 500k, D2 2M papers, sampled as `rows_per_shard` random rows from shuffled
  shards. Each shard is one subset and a narrow year range, so taking whole shards would break the
  time split.

## Masking

All 10 core KU types are masked (`target_types`), mixed in every batch (`mask_schedule="mixed"`).
Hidden KUs are drawn without replacement, so one epoch masks every KU exactly once; `epochs`
(default 10) sets the run length. Leak rules: a masked claim hides its evidence and
implications, a masked evidence or implication hides its claim, and a masked method hides its
procedure (and vice versa). E15 / E18 compare this one model with one model per masked type.

The code also has `balanced`, `round_robin` and `sequential` (+ `replay_frac`) schedules. They are
not in the paper: on 500 papers, sequential masking with one epoch never left the average
prediction (1.0x chance), while mixed masking with 10 epochs reached 8.8x chance.

## Objective

```
L = mean_d ||Pred(E(x), z_tau) - sg(E_bar(y))||^2      (per-dimension MSE, so lambdas are size-independent)
  + lambda_shacl L_SHACL + lambda_align L_align + lambda_reg L_reg
  + lambda_inv L_inverse + lambda_contrast L_contrast
```
Every λ defaults to 0, so the reference arm is the plain objective. `L_inverse` and `L_align`
make the model a hybrid rather than a pure JEPA, and the paper should say so.

## The ICML configuration and experiments

Every run uses the `PaperAConfig` defaults (`config.ICML_FIXED` describes them): section KUs,
gte-modernbert, time split train <= 2023, validation 2024, test 2025-2026 (a stored, domain-balanced split: `sci-jepa split`, docs/data.md), all 10 KU types mixed, leak-safe masking, N = 8 papers,
plain JEPA objective, up to 40 epochs with early stopping on the validation reasoning index (patience 5;
the test uses the best epoch's weights: `patience`, `select_best`, `select_metric`), evaluation every
epoch plus the reasoning-task matrix. Each
experiment changes one thing (`config.PAPER_EXPERIMENTS`, in paper order):

| Exp | Arms | Question | Status |
|---|---|---|---|
| **BENCH** | comparison systems: frozen text / BM25 / SciNCL / BGE-M3 / supervised regressor / LLM-Reasoning; JEPA + GNN: R-GAT / GINE / GraphGPS / Graph-JEPA (GINE + motif); JEPA + graph transformer: reference / paper-as-text / world model / motif / motif + world | The benchmark: 15 systems on identical papers, pools, counterfactuals and evaluation; ranked in the dashboard's Benchmark tab | runnable |
| **ALL** | every distinct arm of every experiment below (66, duplicates removed) | One leaderboard of every method, for the appendix and to see which change helps which task | runnable |
| E10 | N ∈ {1, 4, 8, 16} | Widening the context removes the A1 degeneracy | runnable |
| E11 | λ_inv ∈ {0, .1, .3, 1} | Inverse-relation head as an order-one barrier | runnable |
| E12 | regulariser off / within / across subgraph | H-JEPA Prop. 2, transposed to the subgraph axis | runnable |
| E4 | resample × census | Repair mechanism one: graph construction | runnable |
| E5 | contrast off / on | Relational contrast masking | runnable |
| E15 | mixed + one model per KU type (section) | Which KU types are learnable, and does one model for all help? | runnable |
| E17 | section vs paper_masked | Read a paper as separate KUs or as one text of its visible parts? | runnable |
| E18 | mixed + one model per KU type (paper_masked) | E15 with the paper read as one text | runnable |
| E19 | reference / world model / world model + inverse head | Does action-conditioned prediction teach structure without losing accuracy? | runnable |
| E20 | reference / RWSE / motif / both / both + world model | Do Graph-JEPA's sub-graph targets and structural encodings make the model use structure? | runnable |
| E1 | SHACL off / loose / tight | Theorem A2 classification | **placeholder shapes** |
| E2 | product / Łukasiewicz / Gödel × depth 3, 8, 15 | Proposition A3 depth decay | **placeholder shapes, replicated conjuncts** |
| E3 | factored vs single head × depth | Shape-factored heads | **placeholder shapes** |

**BENCH, the benchmark.** One code runs six comparison systems and the five JEPA variants on the same
instances (same leak-safe and task-aware masking), the same topic-matched candidates, the same
counterfactuals and the same five tasks (`baselines.py`):

| System | `baseline` / setting | What it is |
|---|---|---|
| Frozen-Text Similarity | `text` | the index's frozen encoder (gte-modernbert): mean embedding of the visible paper-mates, cosine |
| BM25 | `bm25` | lexical Okapi BM25 (k1 1.2, b 0.75) over the visible paper-mates' texts; LSA vectors for consistency / intervention |
| SciNCL | `scincl` | scientific-document encoder (malteos/scincl, [CLS], 512 tokens; longer texts split into overlapping windows whose vectors are averaged, so nothing is truncated), same protocol as text |
| BGE-M3 | `bge-m3` | long-context retrieval encoder (BAAI/bge-m3, 8,192 tokens, ~568 M parameters), same protocol; replaces bge-base (`bge`, still available, also chunked) |
| Supervised regressor | `supervised` | same-data control: MLP from the visible mean embedding (+ hidden type) to the hidden KU's embedding, InfoNCE on the train split's instances, early stopping on validation; no graph, no JEPA |

JEPA with graph neural network encoders (`encoder`, `model.ENCODERS`; the context and target encoders and the
predictor all use the family, objective and masking unchanged):

| Arm | `encoder` | Architecture |
|---|---|---|
| jepa+rgat | `rgat` (`gnn`) | relational graph attention: attention only along graph edges (same-paper, membership, role, citation, field, proxy), relation-biased |
| jepa+gine | `gine` | GINE (Hu et al., 2020), the encoder family of Graph-JEPA: h <- MLP((1+eps) h + mean over neighbours of (h_j + e_relation)); one hop per layer |
| jepa+gps | `gps` | GraphGPS (Rampasek et al., 2022): a GINE branch and a global relation-biased attention branch in every layer, summed, then an MLP |
| graph_jepa_gine | `gine` + `motif_p=0.5` | Graph-JEPA-style: GNN encoder and sub-graph (motif) targets |

Every run also stores a plain-language description of what it changes (`config.describe`) and its family
(comparison system / JEPA + GNN / JEPA); the dashboard's Benchmark tab has a family filter and a *Methods*
card listing every method with that description, and `report_pdf.py` writes a PDF (overview, both
leaderboards, reasoning index per method, a methods x tasks heatmap, families, method catalogue).

Consistency for every comparison system is a supervised linear probe on its frozen vectors (train
split). The texts for BM25 / SciNCL / BGE-M3 come from the index's text file (`*.texts.pkl`); encoder
vectors are cached under `paper_a_data/baseline_features/`. An LLM ranker is not included: it needs a
GPU server (a 7B model runs on the MacBook but at 1-3 s per query). The dashboard's Benchmark tab
ranks the arms (mean ± std over seeds), names the strongest comparison system, and shows which JEPA
models are above it. Run it with 3-5 seeds at 60k+ papers for the paper: at 500 papers the 23 test
papers saturate the scores.

**E19, the JEPA world model.** A structure test on the 500-paper pilot showed that the reference
model's 8.8x-chance accuracy does not depend on typed edges at all: removing every role, citation,
field and proxy edge left it at 8.8x. The learned relation biases were large only for
`same_paper` / `none`. E19 adds action-conditioned prediction: a quarter of training instances
get an action token (`do_swap`: a KU replaced by another paper's of the same type; `do_flip`:
a supports <-> contradicts edge flipped), linked by the `do` relation to what it changes, and the
target is the hidden KU's latent in the paper AFTER the action. The plain objective is unchanged
and accuracy is always measured on clean instances. `worldview.py` reports structure reliance
(x chance with vs without typed edges), action prediction (direction cos between predicted and
true change) and explained example subgraphs, shown in the dashboard's World model tab.

**E20, Graph-JEPA ideas** (Skenderi et al., *Graph-level representation learning with joint-embedding
predictive architectures*). Two of its components are adapted to the KU graph:
(1) sub-graph targets: in half of the training instances the hidden KU's whole motif is hidden and
predicted (a claim with its evidence and implications; method + procedure; result + interpretation +
limitation; context + question). The hidden nodes keep the links among themselves (the motif's
shape) and lose their links to visible KUs. (2) Random-walk structural encodings (RWSE, 16 steps on
the graph of typed links) are added to every encoder token, and the predictor gets the hidden
nodes' true structural position, as in Graph-JEPA's predictor (their Eq. 5). Graph-JEPA's 2-D
hyperbola target is not used: it compresses each target to one number, which would remove KU
retrieval. Motifs are used in training only, and evaluation is the reference protocol, so accuracy
compares one to one with the reference arm. New layers have their own seeds, so the reference
initialisation is unchanged.

Removed from the experiment list (not in the paper): the masking-schedule comparison, one model
per type without the mixed reference (now E15), and one-KU-per-paper mode (`ku_mode="paper"`,
which hides whole papers, not parts). Their code paths remain and are smoke-tested.

Every arm of an experiment starts from identical encoder and predictor weights: optional modules
(shape heads, the census embedding) use their own seed. The smoke test checks this.

## Research-domain imbalance

arXiv is dominated by a few fields (local sample: Physics 36%, Computer Science 25%, Mathematics 18%,
Astrophysics 11%, every other domain below 4%), and the summaries' field labels are free text (195 distinct
coarse labels in 11.7k papers). Four measures keep large fields from deciding the benchmark:

| Measure | Code | What it does |
|---|---|---|
| Canonical domains | `ku_schema.domain_of`, `DOMAINS` | ordered keyword rules map every label onto 10 arXiv-style domains (Physics, Astrophysics, Mathematics, Computer Science, Statistics, EESS & Engineering, Quantitative Biology & Medicine, Economics & Finance, Chemistry, Other; ~1% Other on arXiv) |
| Balanced corpus | `field_balance="temperature"`, `field_alpha=0.5`, `field_cap`, `field_pool` | a metadata-only first pass reads `field_pool` x `max_papers` candidates and draws `max_papers` with domain shares proportional to size^alpha (0.5 = square root, as for multilingual corpora); nothing unselected is embedded; per-domain pool and selected counts in `build_stats["domain_balance"]` |
| Macro-averaged evaluation | `primary_average="macro"`, `min_domain_queries=20` | every task is scored per domain (`by_domain`); the primary reasoning index is the mean over domains with >= 20 test queries (each domain weighs the same), with the micro index and the worst domain beside it; findings resample within every domain (stratified bootstrap); early stopping selects on the macro index |
| Domain-weighted training (ablation) | `train_domain_alpha` | training anchors drawn with weight size^(alpha-1) per domain (1 = as in the corpus, 0 = every domain equally) |

## Benchmark validity (validity.py)

| Check | How | Outcome on 500 real papers |
|---|---|---|
| Surface cues in the counterfactuals | a pairwise classifier that sees only the true text and its edit (char n-grams, no paper) names the original; 5-fold cross-validation grouped by KU. `mp_filter` (default on) removes edits named with probability >= 0.9 and every edit of a kind detected in >= 90% of cases | negation 100% detectable -> removed (548 of 1,634 edits); direction 83%, number 72%, substitution 59% (near chance 50%: the cleanest) |
| Faithfulness of the LLM-generated KUs | content-word overlap with the best window of the paper's full text (`text_sanitized`; window = 3-12 sentences, twice the KU's length), overlap with the whole paper, numbers found in the paper | 99.3% of numbers found; claims / evidence / results well grounded (window 0.77 / 0.72 / 0.56); limitations least (0.33): the summariser infers them |
| Expert check | `export` writes identical CSV sheets for two annotators (counterfactual items: which option is the paper's, fluency, meaning change; faithfulness items) + a hidden key; `agreement` computes human accuracy, validity rates and Cohen's kappa | to run: ~200 items, two experts |
| Contamination | the split keeps every test paper in 2025 or later (after gte-modernbert's 2024 cutoff); every task is also scored for test papers after the text encoders' pretraining cutoff (`by_period`; gte-modernbert 2024) | reported per system; findings.py check C |

**LLM-Reasoning** (`baseline="llm"`, BENCH tag `LLM-Reasoning`): an open causal LM (`llm_model`, default
Qwen/Qwen2.5-7B-Instruct) scores every candidate of the same pools by calibrated log-likelihood: mean token
log-probability after a prompt with the visible parts of the paper minus the same without the paper (PMI,
removes the preference for fluent or common texts). Consistency: zero-shot P(supports) vs P(contradicts),
calibrated; intervention: does swapping the procedure raise the donor result's score relative to the own
result. The paper prompt is encoded once per query and its key / value cache reused for every candidate (~20x
fewer tokens; identical scores). Runs on a CUDA GPU (bf16) by default; on a MacBook set `llm_allow_cpu`
(notebook: `LLM_ON_MAC`) and a model up to ~2B (`LLM_MODEL`, e.g. Qwen/Qwen2.5-1.5B-Instruct): the Apple GPU
runs it in fp32, because fp16 and bf16 overflow in Qwen models there (checked: fp32 matches the CPU to 1e-4,
bf16 / fp16 change the ranking). `llm_eval_instances` caps the queries (each ranks ~100 candidates). On the
GPU server: `LLM_MODEL=Qwen/Qwen2.5-7B-Instruct EXPS=BENCH bash cluster/grete/run_paper_a.sh submit dist`.

## Paper figures (figures.py)

When an experiment finishes, `figures/` in its folder receives every chart as vector PDF (for LaTeX) and
300-dpi PNG: fig01 reasoning index with 95% bootstrap intervals, fig02 the five tasks (small multiples),
fig03 counterfactual robustness per task and kind, fig04 systems x tasks heatmap, fig05 / fig06 validation
learning curves per family, fig07 contamination (before vs after the encoders' cutoff), fig08 hypotheses
(forest plot, one panel per metric), fig09 world model (intervention response, action direction), fig10
families, fig11 systems x KU types, and per run (`figures/runs/`) the training losses, per-type validation
and tasks per epoch; plus `leaderboard.csv` and a booktabs `leaderboard.tex`. Families use the first three
slots of a colour-blind-safe palette (validated all-pairs); old-protocol arms are marked with a dagger.

## Main findings (findings.py)

Pre-registered hypotheses, fixed before the large-scale runs and tested when an experiment finishes
(`findings.json`, dashboard Benchmark tab, PDF):

| | Hypothesis | Compared |
|---|---|---|
| H1 | the best JEPA model beats the strongest comparison system on the reasoning index | best JEPA (any encoder) vs best of text / BM25 / SciNCL / BGE-M3 / supervised |
| H2 | reading the KU graph with a GNN helps | best JEPA + GNN vs KU-Graph JEPA |
| H3 | action-conditioned training improves the intervention response | world_model vs reference; motif+world vs motif |
| H4 | JEPA resists counterfactuals better | best JEPA vs strongest comparison system, share of edits ranked below the truth |
| H5 | JEPA beats LLM-Reasoning | best JEPA vs LLM-Reasoning |
| C | no contamination: no system loses > 0.5x chance after the encoders' cutoff | every system |

Each run stores its per-query outcomes; a difference is SUPPORTED when the 95% interval of 2,000 bootstrap
resamples of the test queries (pooled over seeds) is above 0, CONTRADICTED when below, INCONCLUSIVE otherwise
(plus a Welch t-test over seeds with >= 2 seeds). Old-protocol arms (subfield pools, no counterfactuals) are
not candidates for "best". The conclusion is written from the outcomes and states the framing: a method paper
(ICML / KDD research track) if H1 is supported, a benchmark paper (NeurIPS Datasets & Benchmarks / KDD
datasets track) otherwise.

## Diagnostics, always reported together

- **chi:** within-subgraph variance share after removing type centroids (H-JEPA Def. 1 scale).
  `chi_raw` (with type centroids) and `chi_per_type` are reported alongside.
- **Partition probe:** linear probe for subgraph identity, split within type, plus the
  between-subgraph variance share. G1 wants the share below 30%.
- **Retrieval:** MRR / Hits@1 / Hits@10 against the full same-type bank (standard) and against
  K = `balanced_k` same-type distractors from the same subfield, then proxy / citation
  neighbours, then the same field (balanced). The pool never depends on the sampled
  neighbourhood, so it is identical across E10's N arms. Its mean size and chance MRR are
  reported. The shortcut margin comes with a bootstrap CI.
- **Relation sensitivity:** the relative change in the prediction when one role edge is rewired
  to another subgraph.
- **Inverse accuracy:** the head is trained in every arm. When λ_inv = 0 it is a detached probe
  that cannot affect the encoder.
- **Online detector:** constraint satisfaction on the true target minus satisfaction against
  a same-type impostor.
- **Nuisance audit:** paraphrase similarity between the target and the visible context, the
  fraction of targets whose type plus subgraph fixes them, and the number of papers per instance.

## Stand-ins for data that does not exist yet

Every result must state these. All of them are marked in the code.

| Plan needs | Used here instead | Replace in |
|---|---|---|
| Grounded edges (canonical Material / Method / Metric) | **Proxy:** method KUs with text-embedding cosine ≥ 0.85 across papers | `graph_index.build_proxy_edges` |
| Declared edges (citation stance, replicates, reuses) | OpenAlex `cites` (no stance) between paper hubs | `ku_graph.build_blob` |
| Claim-dependency DAG and its masking regime | not implemented | — |
| ORKG shape vocabulary | `placeholder_shapes.json` (real counter-evidence, evidence cites a figure, quantitative text, paper type, field) | `--set shape_spec=...json`, same format |
| Text encoder older than T | gte-modernbert-base (2024 data): `time_T < 2024` prints a contamination warning | `config.embedder` |
| Confidence weights (E9), z_do / z_pub conditioning | none | `losses.jepa_loss(weights=)`, `model.Predictor` |
| D-ext (OGBN-MAG) | not implemented | `ku_graph.build_blob(records=...)` |

Time-sliced splits are now real (`split_mode="time"`, train ≤ T, val T+1, test T+2; sweep T).

The placeholder shape spec deliberately includes context-determined constraints (field, paper
type, a result's count of grounded claims, which the visible claims reveal) and validity
constraints (cardinality, datatype). Under Theorem A2 these should not separate the degenerate
branch. A claim's counts of supporting evidence and implications are exactly 1 by the extraction
template, so they carry 0 bits and are not slots. `--audit-shapes` checks each constraint's
tight/loose label against the data.

## Leak-safe masking

The default `mask_policy="token+edges+children"` blanks the masked token's features and drops
its role edges. For a claim it also drops the claim's evidence and implication children, which
paraphrase it (supporting evidence usually restates the claim). `+siblings` additionally hides a
masked method's procedure and vice versa, since the two sections restate each other. The smoke
test asserts both. Paper text never becomes a token: each paper is a hub carrying only its field
embedding.

## Data split

The corpus, its year split and its domain balance are made once by `sci-jepa split` and stored
(`data/splits/<name>/`); training and benchmarking read that split and never sample or split again.
How the split is built, its domain shares and the counts on arXiv + PubMed: docs/data.md.

Truncation by 512-token encoders on real KUs: 0.33% of KUs and 0.28% of tokens (median KU 34 tokens, p99 434),
so chunking removes a caveat rather than changing scores.
