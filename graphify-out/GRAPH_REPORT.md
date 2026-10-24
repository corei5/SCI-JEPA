# Graph Report - SCI-JEPA  (2026-10-21)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2370 nodes · 5567 edges · 147 communities (101 shown, 46 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 317 edges (avg confidence: 0.9)
- Token cost: 20,273 input · 1,011 output

## Graph Freshness
- Built from commit: `d5cdebec`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Config and Data Setup
- CLI Paths Dashboard
- Heterogeneous Graph Builder
- Paper Reasoning JEPA
- Link Prediction Model
- Retrieval Stage Pipeline
- Graph Positional Encoding
- Reasoning Experiments
- Information Theory Metrics
- Paper Data Parsing
- Rebuttal Diagnostic Statistics
- JEPA Training Retrieval
- Analysis Figures
- Attention and BM25 Adapters
- PDF Report Generation
- Pooling and Hyperbolic Encoding
- JEPA Encoder Head
- Lightning Training Engine
- Hyperbolic Space Reasoning
- Retrieval Experiment Analysis
- GJEPA Reasoning Evaluation
- Reference Model Entropy
- Lightning Training Objectives
- Experiment Shell Scripts
- Config Test Infrastructure
- Feature Encoders
- Training Loop Execution
- Graph Encoder Model
- JEPA Graph Model
- GNN Building Blocks
- MLP Mixer Attention
- LLM Rebuttal Scripts
- Contrastive Learning Stages
- Corpus Data Loading
- Graph Transformer Encoder
- Findings Analysis
- Retrieval Phase Simulation
- Knowledge Unit Graph Index
- Paper A Configuration
- Training Callbacks
- Multihead Attention Module
- Multihead Attention Module
- Multihead Attention Module
- JEPA Environment Config
- LLM Rebuttal Script
- Multi-aspect JEPA Encoder
- Evaluation and Encoding
- Gated Graph Convolution
- JEPA Model Core
- Set Transformer Pooling
- Graphormer Encoder Layer
- Hadamard Encoder Layer
- Dataset Split Logic
- Results Ledger Builder
- Cluster Job Scripts
- Parquet Normalization
- Diagnostics CLI Driver
- Logging Probes Seeds Utils
- Config Data Model Loading
- Analysis Local Runner
- JSON Schema Text Extraction
- Graph Attention Convolution
- Residual Gated Conv
- Baseline Comparison Systems
- MCP Tool Config
- Graphify MCP Script
- SCI-JEPA Package
- Evaluation Pipeline
- Training Sampler
- Inference and Embedding
- Infrastructure Reporting
- Training Instance Builder
- Validity and Faithfulness
- Training Telemetry
- Evaluation CLI
- Diagnostics Evaluation
- Parquet Test Suite
- Output Schema Tests
- Experiment Management
- SciKU Bench Paper Guide
- LLM Evaluation System
- Paper A Model
- Training Test Suite
- Neuron Interpretation
- SAM Optimizer Module
- Reporting Figures
- Graph Visualization HTML
- Package Integration Tests
- Record Normalization Tests
- Knowledge Unit Schema
- Parquet Writer
- Hardware Monitor
- Project README
- Infrastructure Monitoring
- Soft Constraint Loss
- System Hardware Info
- Training Loss Functions
- Evaluation Protocol
- Paper Graph Dataset
- Dashboard HTTP Server
- Knowledge Graph Index
- Training Performance Profiling
- Project Setup Guide
- CLI Entry Point
- Technical Documentation
- KU Text Parsing
- VICReg Training Losses
- Environment Check
- Domain and Split Docs
- Parquet Output Tests
- Parquet Schema Tests
- Cluster Operations Guide
- Local Run Environment Config
- Planar SAT Dataset
- Paper Data Pipeline
- GraphGPS Encoder Blocks
- GINE Message Passing
- Set Pooling Layer
- SCI-JEPA Overview Docs
- Training Scaling Report
- Slurm Cluster Setup
- Training Infrastructure Docs
- JEPA Predictor Module
- Project Documentation
- Codebase Migration Guide
- Summary Download Script
- Dashboard Presentation Script
- Graph-JEPA Package
- SCI-JEPA Config Data
- Contributing Guidelines
- Relation Bias Attention
- Architecture Docs
- SCI-JEPA Profiling
- SLURM Job Utilities
- Masked Token Predictor
- Agent Guide Commands
- Relation Bias Gradients
- MCP Graphify Config
- Graph Refresh Script
- Claude Instructions

## God Nodes (most connected - your core abstractions)
1. `train_one()` - 38 edges
2. `PaperAConfig` - 37 edges
3. `Sampler` - 37 edges
4. `build_hetero_graph()` - 32 edges
5. `main()` - 31 edges
6. `to_device()` - 30 edges
7. `evaluate()` - 28 edges
8. `device()` - 26 edges
9. `main()` - 26 edges
10. `get_index()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `Graph-JEPA (`Analysis/`)` --references--> `GraphJepa`  [INFERRED]
  ARCHITECTURE.md → Analysis/core/model.py
- `Graph-JEPA (`Analysis/`)` --references--> `run_k_fold()`  [INFERRED]
  ARCHITECTURE.md → Analysis/core/trainer.py
- `Configuration` --references--> `PaperAConfig`  [INFERRED]
  SCI-JEPA-training/README.md → SCI-JEPA-training/src/sci_jepa/config.py
- `How the split is made (`src/sci_jepa/data/split.py`)` --references--> `domain_of()`  [INFERRED]
  SCI-JEPA-training/docs/data.md → SCI-JEPA-training/src/sci_jepa/data/ku_schema.py
- `Commands` --references--> `figures()`  [INFERRED]
  SCI-JEPA-training/README.md → SCI-JEPA-training/src/sci_jepa/infra/report.py

## Import Cycles
- None detected.

## Communities (147 total, 46 thin omitted)

### Community 0 - "Config and Data Setup"
Cohesion: 0.07
Nodes (25): calculate_stats(), hyperbolic_dist(), unithyper_geodesic(), test(), train(), test(), train(), test() (+17 more)

### Community 1 - "CLI Paths Dashboard"
Cohesion: 0.18
Nodes (13): _context(), _dashboard(), main(), _train(), dashboard_root(), analysis_dir(), data_root(), graphify_dir() (+5 more)

### Community 2 - "Heterogeneous Graph Builder"
Cohesion: 0.19
Nodes (6): analyze_schema(), build_homogeneous_reasoning_graph(), build_ownership(), compute_patch_rwse(), connectivity_report(), main()

### Community 3 - "Paper Reasoning JEPA"
Cohesion: 0.09
Nodes (23): _apply_masked_edges_to(), build_incident_map(), build_paper_target_map(), collapse_stats(), get_partition(), GNNEncoder, JEPA, main() (+15 more)

### Community 4 - "Link Prediction Model"
Cohesion: 0.07
Nodes (15): _agg_aspect(), _agg_flat(), EdgeDecoder, GNNEncoder, link_metrics(), LinkModel, load_claim_reasoner(), main() (+7 more)

### Community 5 - "Retrieval Stage Pipeline"
Cohesion: 0.11
Nodes (20): _align_to_papers(), _bank_ranks(), _cross_cell(), feature_bank(), _geometry(), _get_x(), hetero_features(), main() (+12 more)

### Community 6 - "Graph Positional Encoding"
Cohesion: 0.11
Nodes (13): LapPE(), random_walk(), RWSE(), cal_coarsen_adj(), combine_subgraphs(), combine_subgraphs_jepa(), GraphJEPAPartitionTransform, PositionalEncodingTransform (+5 more)

### Community 7 - "Reasoning Experiments"
Cohesion: 0.12
Nodes (24): bits_carried(), bootstrap_mrr_ci(), build_cost(), build_membership(), build_query_set(), chance_mrr(), _cos_loss(), _fmt_hms() (+16 more)

### Community 8 - "Information Theory Metrics"
Cohesion: 0.12
Nodes (23): eff_rank_only(), compute_tap(), dc_stats(), estimate_rho_real(), fit_bank_transform(), information_ladder(), measured_redundancy(), node_type_degrees() (+15 more)

### Community 9 - "Paper Data Parsing"
Cohesion: 0.10
Nodes (11): _aspect_text(), build_hetero_graph(), _claim_records(), _coarse(), _coerce_summary(), _get_claims_list(), _iter_raw_records(), _paper_id() (+3 more)

### Community 10 - "Rebuttal Diagnostic Statistics"
Cohesion: 0.13
Nodes (23): bits_recovered(), bootstrap_ci(), device(), mrr(), permutation_p(), ranks_from_banks(), recall_at(), spectral_decay_kappa() (+15 more)

### Community 11 - "JEPA Training Retrieval"
Cohesion: 0.17
Nodes (5): train_jepa(), _lex_idx(), _methods(), _query_idx(), stage_faithful()

### Community 12 - "Analysis Figures"
Cohesion: 0.17
Nodes (25): _chance(), fig_centering(), fig_dc_ratio(), fig_embedding_geometry(), fig_encoder_depth(), fig_fix_bars(), fig_information_ladder(), fig_ladder() (+17 more)

### Community 13 - "Attention and BM25 Adapters"
Cohesion: 0.11
Nodes (10): bm25_ranks(), BM25Index, _fingerprint(), get_bm25_index(), hard_pool(), pooled_ranks(), protocol_r_frame(), strip_ngram_overlap() (+2 more)

### Community 14 - "PDF Report Generation"
Cohesion: 0.11
Nodes (12): describe(), method_family(), setup_dashboard(), reasoning_index(), build(), _display_name(), _f(), _g() (+4 more)

### Community 15 - "Pooling and Hyperbolic Encoding"
Cohesion: 0.10
Nodes (6): AttnPool, GINEncoder, mlp3(), MultiStatPool, HyperbolicAnchors, StructBindPool

### Community 16 - "JEPA Encoder Head"
Cohesion: 0.11
Nodes (8): _ema_(), _etkey(), _gpu_peak_gb(), HeteroEncoder, JEPAHead, _sample_rows(), Timer, train_encoder_variant()

### Community 17 - "Lightning Training Engine"
Cohesion: 0.11
Nodes (21): _barrier(), _devices(), _fit_kfold(), fit_one_seed(), _fit_split(), _group_name(), _make_trainer(), _read_summaries() (+13 more)

### Community 18 - "Hyperbolic Space Reasoning"
Cohesion: 0.19
Nodes (20): aspect_presence(), build_patch_index(), build_reasoning_label(), collapse_stats(), eval_patchret(), eval_probe(), lorentz_expmap0(), lorentz_inner() (+12 more)

### Community 19 - "Retrieval Experiment Analysis"
Cohesion: 0.20
Nodes (9): aspect_feature_matrix(), bm25_retrieval(), _fmt(), _pick_device(), sbert_embed(), stage_latent(), stage_oracle(), strip_shared_ngrams() (+1 more)

### Community 20 - "GJEPA Reasoning Evaluation"
Cohesion: 0.13
Nodes (15): build_cost_summary(), eval_patchret_cosine(), _fmt_hms(), _gpu_peak_gb(), _gpu_reset_peak(), main(), mean_std(), verdict() (+7 more)

### Community 21 - "Reference Model Entropy"
Cohesion: 0.14
Nodes (8): effective_rank(), set_seed(), conditional_variance_floor(), _grad_norms(), _mlp(), RefJEPA, train_reference_jepa(), _rank_probe()

### Community 22 - "Lightning Training Objectives"
Cohesion: 0.15
Nodes (12): configure_sharing(), _batch_size(), covariance_loss(), distance_step(), ema_bounds(), ema_update(), interpolate_momentum(), prediction_loss() (+4 more)

### Community 23 - "Experiment Shell Scripts"
Cohesion: 0.16
Nodes (16): arm_err(), CUDA_VISIBLE_DEVICES, HF_HOME, MPLBACKEND, MPLCONFIGDIR, mute_err(), p4_build_extra(), p4_has() (+8 more)

### Community 24 - "Config Test Infrastructure"
Cohesion: 0.19
Nodes (7): set_cfg(), _Cfg, test_data_root_honors_explicit_override(), test_defaults_select_lightning(), test_objective_math(), test_resolve_config_finds_papers_yaml(), test_seeds_match_the_historical_lists()

### Community 25 - "Feature Encoders"
Cohesion: 0.29
Nodes (6): covariance_loss(), AtomEncoder(), BondEncoder(), DiscreteEncoder(), FeatureEncoder(), LinearEncoder()

### Community 26 - "Training Loop Execution"
Cohesion: 0.05
Nodes (37): 14. Performance engineering, 8.1 `train_one(cfg, gi, seed, log, live, save_dir, ckpt_path)`, 8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)`, 8. Training, gpu_step_time(), step(), step(), minimal_pair() (+29 more)

### Community 27 - "Graph Encoder Model"
Cohesion: 0.18
Nodes (3): GraphEncoder, InverseRelationHead, rwse()

### Community 29 - "GNN Building Blocks"
Cohesion: 0.10
Nodes (6): Identity, MLP, VNUpdate, GNN, GCNConv, GINEConv

### Community 30 - "MLP Mixer Attention"
Cohesion: 0.18
Nodes (4): MLPMixer, FeedForward, MixerBlock, MLPMixer

### Community 31 - "LLM Rebuttal Scripts"
Cohesion: 0.13
Nodes (14): GJEPA_ARXIV_PROC, GJEPA_ARXIV_RAW, GJEPA_EXTRACTOR_A, GJEPA_EXTRACTOR_B, GJEPA_ROOT, HF_HOME, MPLBACKEND, MPLCONFIGDIR (+6 more)

### Community 32 - "Contrastive Learning Stages"
Cohesion: 0.20
Nodes (13): _infonce_fullpool(), _infonce_hard(), _mm(), pca2_of(), protocol_r_model(), spectrum_of(), stage_centering(), stage_encoder() (+5 more)

### Community 33 - "Corpus Data Loading"
Cohesion: 0.14
Nodes (10): aspect_features(), CorpusSpec, _dig(), _flat(), get_spec(), _iter_records(), load_graph(), load_texts() (+2 more)

### Community 34 - "Graph Transformer Encoder"
Cohesion: 0.22
Nodes (4): GTEncoderLayer, Graph, Kernel, Standard

### Community 35 - "Findings Analysis"
Cohesion: 0.13
Nodes (18): 12.2 Result JSON (one per arm × seed), analyse(), _arm(), _avg(), _boot_task(), bootstrap(), _ci(), compare() (+10 more)

### Community 36 - "Retrieval Phase Simulation"
Cohesion: 0.24
Nodes (6): _collapsed(), _critical_rho(), collapsed_at(), _measured_rho(), _simulate(), stage_phase()

### Community 37 - "Knowledge Unit Graph Index"
Cohesion: 0.07
Nodes (22): 6.3 `data/ku_graph.py`, Stand-ins for data that does not exist yet, build_proxy_edges(), build_blob(), build_context_texts(), build_minimal_pairs(), HashEmbedder, iter_records() (+14 more)

### Community 38 - "Paper A Configuration"
Cohesion: 0.09
Nodes (24): 5.1 Main groups of `PaperAConfig`, 5.2 Presets, 5.3 Environment variables, 5. Configuration, Benchmark validity (validity.py), Data split, Diagnostics, always reported together, Knowledge units (KUs) (+16 more)

### Community 43 - "JEPA Environment Config"
Cohesion: 0.20
Nodes (9): CUDA_VISIBLE_DEVICES, HF_HOME, PYTHONUNBUFFERED, SCIJEPA_CHECKPOINT_ROOT, SCIJEPA_DASHBOARD_ROOT, SCIJEPA_DATA_ROOT, SENTENCE_TRANSFORMERS_HOME, JEPA-rea.sh script (+1 more)

### Community 44 - "LLM Rebuttal Script"
Cohesion: 0.20
Nodes (9): HF_HOME, MPLBACKEND, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONPATH, PYTHONUNBUFFERED, SENTENCE_TRANSFORMERS_HOME, run_LLM_rebuttal_w126.sh script (+1 more)

### Community 45 - "Multi-aspect JEPA Encoder"
Cohesion: 0.18
Nodes (3): GraphJEPAMulti, oracle_retrieval_test(), relhetero_pool()

### Community 46 - "Evaluation and Encoding"
Cohesion: 0.11
Nodes (19): 16. Extending the code, 9.1 Main evaluation: `evaluation/diagnostics.py`, 9.2 Reasoning tasks: `evaluation/tasks.py`, 9. Evaluation, build_view(), centred_cos(), encode_papers(), add_domains() (+11 more)

### Community 52 - "Dataset Split Logic"
Cohesion: 0.19
Nodes (10): _as_text(), ku_fix_drops(), main(), make_split(), parquet_files(), parse_args(), read_rows(), report() (+2 more)

### Community 53 - "Results Ledger Builder"
Cohesion: 0.33
Nodes (4): build_ledger(), _g(), stage_ledger(), to_latex()

### Community 54 - "Cluster Job Scripts"
Cohesion: 0.08
Nodes (25): DATASET, MKL_NUM_THREADS, OMP_NUM_THREADS, PAPER_A_PARQUET, PYTHONUNBUFFERED, run_paper_a.sh script, TOKENIZERS_PARALLELISM, tunnel() (+17 more)

### Community 55 - "Parquet Normalization"
Cohesion: 0.11
Nodes (12): _canon(), _check_paths(), _env(), free_arrow_memory(), load_progress(), peak_mb(), _progress_path(), read_parquet_folder() (+4 more)

### Community 56 - "Diagnostics CLI Driver"
Cohesion: 0.10
Nodes (13): _banner(), _corpus_has_json(), _has_ok(), main(), _parse_ints(), _parse_stages(), _result_dict(), StageTimeout (+5 more)

### Community 57 - "Logging Probes Seeds Utils"
Cohesion: 0.10
Nodes (14): config_logger(), _Tee, seeds_for(), collapse_diagnostics(), count_parameters(), _effective_rank(), _ema_range(), k_fold() (+6 more)

### Community 58 - "Config Data Model Loading"
Cohesion: 0.34
Nodes (6): resolve_config(), update_cfg(), create_dataset(), create_model(), extract_embeddings(), main()

### Community 59 - "Analysis Local Runner"
Cohesion: 0.40
Nodes (4): PATH, PYTHONUNBUFFERED, SCIJEPA_DATA_ROOT, run_local.sh script

### Community 60 - "JSON Schema Text Extraction"
Cohesion: 0.40
Nodes (5): dump_json_schema(), walk(), extract_text(), load_texts(), _walk_collect()

### Community 63 - "Baseline Comparison Systems"
Cohesion: 0.07
Nodes (26): 9.3 Comparison systems: `evaluation/baselines.py`, _agg(), BM25System, _by_kind(), _consistency_probe(), feats(), _consistency_unsup(), DenseSystem (+18 more)

### Community 71 - "Evaluation Pipeline"
Cohesion: 0.08
Nodes (18): render_all(), _save(), task_summary(), board(), _bool(), display(), domain_balance_table(), _env() (+10 more)

### Community 72 - "Training Sampler"
Cohesion: 0.08
Nodes (14): 6.5 `data/instances.py`, ablate_structure(), add_action(), Sampler, action_eval(), _capture(), examples(), _explain() (+6 more)

### Community 73 - "Inference and Embedding"
Cohesion: 0.20
Nodes (3): Saved models and inference, collate_views(), SciJEPA

### Community 74 - "Infrastructure Reporting"
Cohesion: 0.12
Nodes (20): compute(), figures(), timeline(), hardware(), infra_pages(), llm_figure(), llm_pages(), _llm_rows() (+12 more)

### Community 75 - "Training Instance Builder"
Cohesion: 0.12
Nodes (12): _add_text_token(), build_instance(), _drop_tokens(), _extra_anchors(), _hard_negative(), _leak_family(), motif_of(), _paper_tokens() (+4 more)

### Community 76 - "Validity and Faithfulness"
Cohesion: 0.17
Nodes (14): 9.4 Other evaluation modules, agreement(), apply_filter(), audit(), _cell(), contamination(), export(), faithfulness() (+6 more)

### Community 77 - "Training Telemetry"
Cohesion: 0.14
Nodes (3): _atomic_dump(), _clean(), LiveRun

### Community 78 - "Evaluation CLI"
Cohesion: 0.14
Nodes (15): resolve_device(), cmd_model(), cmd_results(), cmd_system(), fmt(), g(), get_gi(), headline() (+7 more)

### Community 79 - "Diagnostics Evaluation"
Cohesion: 0.15
Nodes (12): _boot_diff(), _breakdowns(), _chance(), chi_stats(), effective_rank(), evaluate(), _mrr(), partition_probe() (+4 more)

### Community 80 - "Parquet Test Suite"
Cohesion: 0.13
Nodes (4): json_files(), parquet_files(), _record_files(), TestEndToEnd

### Community 82 - "Experiment Management"
Cohesion: 0.20
Nodes (11): fit_one_seed_cfg(), _atomic(), ensure_dashboard(), Experiment, json_default(), _now(), _package_dashboard(), _public_config() (+3 more)

### Community 83 - "SciKU Bench Paper Guide"
Cohesion: 0.10
Nodes (19): 10The A\* paper, 11What reviewers will ask, 12Work plan, 1Short answers, 2The five reasoning tasks, 3The key finding: in their current form, three tasks are solved without reasoning, 4Making the tasks require reasoning, 5Why no-training baselines are required (+11 more)

### Community 86 - "Training Test Suite"
Cohesion: 0.13
Nodes (13): get_index(), data_fingerprint(), embed_cache_dir(), synthetic_records(), _jepa_losses(), Stop, test_batch_workers_independent_of_worker_count_and_resumable(), dies_at_step_8() (+5 more)

### Community 87 - "Neuron Interpretation"
Cohesion: 0.12
Nodes (13): collate(), _corr(), _label(), _layer_verdict(), neuron_report(), omega2(), _top_group(), _type_centre() (+5 more)

### Community 88 - "SAM Optimizer Module"
Cohesion: 0.18
Nodes (3): ASAM, SAM, GraphJepaLitModule

### Community 89 - "Reporting Figures"
Cohesion: 0.33
Nodes (9): export(), cell(), _fig_legend(), _footnote(), _hbar(), _label(), _legend(), _old() (+1 more)

### Community 90 - "Graph Visualization HTML"
Cohesion: 0.29
Nodes (10): corpus_html(), _edge(), _hub_node(), instance_html(), _ku_node(), paper_html(), short_id(), _snip() (+2 more)

### Community 91 - "Package Integration Tests"
Cohesion: 0.17
Nodes (5): all_modules(), test_cli_help(), test_module_imports(), test_paths_come_from_the_environment(), test_smoke()

### Community 92 - "Record Normalization Tests"
Cohesion: 0.19
Nodes (3): normalize_record(), _safe_filename(), TestNormalizer

### Community 93 - "Knowledge Unit Schema"
Cohesion: 0.15
Nodes (9): active_templates(), _case(), compose_text(), counterfactuals(), openalex_key(), paper_type(), referenced(), substitution() (+1 more)

### Community 96 - "Project README"
Cohesion: 0.14
Nodes (14): 1. Environment, 2. Data, 3. Run the pipeline, 🙌 Acknowledgements, 💾 Caches, ⚙️ Configuration, ✨ Features, 🚀 Getting Started (+6 more)

### Community 97 - "Infrastructure Monitoring"
Cohesion: 0.19
Nodes (6): _atomic(), progress_dir(), write_progress(), area_of(), build_sections(), main()

### Community 98 - "Soft Constraint Loss"
Cohesion: 0.23
Nodes (3): _keep(), ShapeBank, tnorm()

### Community 99 - "System Hardware Info"
Cohesion: 0.28
Nodes (8): collect(), _cpu(), _gpus(), _run(), _slurm(), _storage(), _torch(), _versions()

### Community 100 - "Training Loss Functions"
Cohesion: 0.14
Nodes (9): variance_loss(), 7.3 Modules, align_loss(), contrast_margin(), epps_pulley(), grouping_regulariser(), inverse_loss(), jepa_loss() (+1 more)

### Community 101 - "Evaluation Protocol"
Cohesion: 0.15
Nodes (13): 1. The data unit: knowledge units, 2. Split by time: no future papers in training, 3. A query: mask one KU, predict it, 4. Ranking against hard candidates, 5. The five reasoning tasks, 6. Comparison systems: same queries, same candidates, 7. Model diagnostics (models only), 8. Seeds, best epoch and hypotheses (+5 more)

### Community 102 - "Paper Graph Dataset"
Cohesion: 0.13
Nodes (4): _coerce_summary(), _extract_node_texts(), PapersDataset, parse_paper()

### Community 103 - "Dashboard HTTP Server"
Cohesion: 0.21
Nodes (3): _Quiet, register_graph(), start_server()

### Community 107 - "Project Setup Guide"
Cohesion: 0.18
Nodes (11): Commands, Configuration, Documentation, Install, Long runs on a cluster (e.g. 60,000 papers), Quick start, Repository layout, Reproducibility (+3 more)

### Community 108 - "CLI Entry Point"
Cohesion: 0.29
Nodes (5): fetch_assets(), main(), usage(), ensure_vendor(), test_cli_lists_every_command()

### Community 109 - "Technical Documentation"
Cohesion: 0.13
Nodes (14): 10. Reporting, dashboard and infrastructure monitoring, 12.1 Layout, 12. Outputs and file formats, 13. Reproducibility and determinism, 15. Testing, 17. The 60 k run configuration, 18. Troubleshooting, 1. What the code does (+6 more)

### Community 110 - "KU Text Parsing"
Cohesion: 0.20
Nodes (7): clean(), ku_flags(), paper_ku(), parse_claims(), record_to_kus(), add(), strip_null_statements()

### Community 111 - "VICReg Training Losses"
Cohesion: 0.25
Nodes (5): covariance_loss(), prediction_loss(), test(), train(), variance_loss()

### Community 112 - "Environment Check"
Cohesion: 0.32
Nodes (3): page(), step(), free_port()

### Community 113 - "Domain and Split Docs"
Cohesion: 0.22
Nodes (8): 6.1 `data/ku_schema.py`, 6.2 `data/split.py`, 6.4 `data/graph_index.py`, 6. Data layer, Research-domain imbalance, domain_of(), paper_year(), split_field()

### Community 115 - "Parquet Schema Tests"
Cohesion: 0.21
Nodes (8): build_unified_schema(), balanced_quota(), domain_weights(), test_mixed_column_types_become_text(), test_quota_follows_shares_and_redistributes(), test_quota_zero_means_everything(), test_same_shares_in_every_split(), test_uniform_and_natural_weights()

### Community 116 - "Cluster Operations Guide"
Cohesion: 0.25
Nodes (8): Browser, Grete facts that matter, Long experiments: chained 12-hour jobs, Make the data split, Once, Running on a Slurm cluster (GWDG Grete), Runs, Speed settings

### Community 117 - "Local Run Environment Config"
Cohesion: 0.25
Nodes (7): PAPER_A_INDEX_DIR, PAPER_A_OUT, PAPER_A_PARQUET, PYTHONUNBUFFERED, PYTORCH_ENABLE_MPS_FALLBACK, run_local.sh script, TOKENIZERS_PARALLELISM

### Community 119 - "Paper Data Pipeline"
Cohesion: 0.29
Nodes (7): 1. Download, 2. Normalize, 3. Split (train / validation / test), domain-balanced, Data: download, normalize, split, How the split is made (`src/sci_jepa/data/split.py`), Paper ids, Why the balance is needed: arXiv + PubMed per split

### Community 120 - "GraphGPS Encoder Blocks"
Cohesion: 0.22
Nodes (6): 7.1 Attention with relation biases, 7.2 Encoder families (`make_blocks`), 7. Model, Block, GPSBlock, make_blocks()

### Community 121 - "GINE Message Passing"
Cohesion: 0.18
Nodes (3): _edges(), GINEBlock, GINEMessage

### Community 123 - "SCI-JEPA Overview Docs"
Cohesion: 0.33
Nodes (6): Code graph, Data paths, Install, Logs and the dashboard, SCI-JEPA, Train

### Community 124 - "Training Scaling Report"
Cohesion: 0.33
Nodes (6): 1. Where the time went (before), 2. What was changed, 3. Measured effect (after), 4. Accuracy check (before vs after), 5. Files, Scaling SCI-JEPA training to 60k papers

### Community 125 - "Slurm Cluster Setup"
Cohesion: 0.40
Nodes (4): 11.1 Scripts, 11.2 One job in `dist` mode, 11.3 Dashboard access, 11. Running on a Slurm cluster

### Community 126 - "Training Infrastructure Docs"
Cohesion: 0.40
Nodes (5): How compute is counted, Long runs: checkpoints and chained jobs, Outputs for the paper, Training infrastructure, What is recorded

### Community 129 - "Codebase Migration Guide"
Cohesion: 0.50
Nodes (4): Changed after the migration (intended), Migration from SCI-JEPA-training/Analysis, What changed for users, Where things went

### Community 136 - "Contributing Guidelines"
Cohesion: 0.29
Nodes (7): Branches, Code graph, Contributing, Pull requests, What not to commit, Working from a terminal, Working from the GitHub web UI

### Community 138 - "Architecture Docs"
Cohesion: 0.33
Nodes (6): Architecture, Generated views (Graphify), Graph-JEPA (`Analysis/`), How the pieces connect, Paper A (`SCI-JEPA-training/`), The papers pipeline (`Analysis/JEPA-rea.sh`)

### Community 142 - "Agent Guide Commands"
Cohesion: 0.50
Nodes (4): Agent guide, Commands, graphify, Rules for this repository

## Knowledge Gaps
- **66 isolated node(s):** `PAPER_A_INDEX_DIR`, `PAPER_A_OUT`, `PAPER_A_PARQUET`, `PYTHONUNBUFFERED`, `PYTORCH_ENABLE_MPS_FALLBACK` (+61 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 914 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SCI-JEPA: technical code documentation` connect `Technical Documentation` to `Paper A Configuration`, `Project Documentation Files`, `Evaluation and Encoding`, `Domain and Split Docs`, `GraphGPS Encoder Blocks`, `Training Loop Execution`, `Slurm Cluster Setup`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **What connects `PAPER_A_INDEX_DIR`, `PAPER_A_OUT`, `PAPER_A_PARQUET` to the rest of the system?**
  _66 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Config and Data Setup` be split into smaller, more focused modules?**
  _Cohesion score 0.07399577167019028 - nodes in this community are weakly interconnected._
- **Why does `PaperAConfig` connect `Paper A Configuration` to `Config and Data Setup`, `Knowledge Unit Graph Index`, `SCI-JEPA Config Data`, `Evaluation Pipeline`, `Inference and Embedding`, `Project Setup Guide`, `Validity and Faithfulness`, `PDF Report Generation`, `Evaluation and Encoding`, `Evaluation CLI`, `Dataset Split Logic`, `Training Test Suite`, `Training Loop Execution`, `Baseline Comparison Systems`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Should `Paper Reasoning JEPA` be split into smaller, more focused modules?**
  _Cohesion score 0.08826945412311266 - nodes in this community are weakly interconnected._
- **Why does `GraphJepa` connect `JEPA Graph Model` to `Feature Encoders`, `Config Data Model Loading`, `GNN Building Blocks`, `Architecture Docs`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `Link Prediction Model` be split into smaller, more focused modules?**
  _Cohesion score 0.07307692307692308 - nodes in this community are weakly interconnected._