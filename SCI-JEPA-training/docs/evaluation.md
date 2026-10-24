# SCI-JEPA Paper A: evaluation

`sci-jepa evaluate` (`src/sci_jepa/evaluate.py`) evaluates models and comparison systems on the Paper A benchmark. It calls the same code, settings and random seeds as training, so its numbers match the ones written at the end of training. Re-evaluating a saved model reproduces its training-time test score exactly.

## Setup

`pip install -e .` from the repository root (README.md). A CUDA GPU is needed to evaluate models and LLM-Reasoning. The `results` command runs on any CPU. Data: a stored split (`sci-jepa split`, docs/data.md) or the normalized parquet.

## Commands

```bash
# 1. A trained model folder (model.safetensors, best.safetensors, config.json, summary.json)
sci-jepa evaluate model  <model_dir> --weights best \
       --split-dir data/splits/d0 --index-dir runs/paper_a_index

# 2. A comparison system, on the same papers, candidates, counterfactuals and tasks as a model
sci-jepa evaluate system bm25 --config <model_dir>/config.json \
       --split-dir data/splits/d0 --index-dir runs/paper_a_index
#    systems: text | bm25 | scincl | bge | bge-m3 | minilm | supervised | llm (--llm-model <HF id>)

# 3. A finished experiment folder (<arm>_seed<s>.json files): leaderboard + hypothesis tests, no GPU
sci-jepa evaluate results runs/paper_a_results/BENCH
```

Each command writes `<out>.json` (every metric, including per-query scores) and `<out>.md` (a readable report). `results` also writes `<out>_runs.csv`. Use `--split val` for the validation split and `--out` to choose the file name.

`--split-dir` (or `--parquet` without a stored split) and `--index-dir` point the script at your copy of the data. A model's `config.json` stores the paths of the machine it was trained on. If the KU index isn't in `--index-dir`, it is rebuilt from the parquet. The same settings give the same papers, splits and candidates; the script checks the data fingerprint against the model's `summary.json` and warns if they differ.

## How the evaluation works

**In short: yes, it predicts the masked part.** One knowledge unit (KU) of a paper is hidden. The model predicts its representation from what stays visible. The prediction is then used to *rank* candidates: the true hidden KU against about 100 hard distractors and edited copies of the truth. Good models put the truth first.

### 1. The data unit: knowledge units

Every paper's structured summary is split into typed KUs: context, research question, method, procedure, result, interpretation, limitation, claim, supporting/contradicting evidence, and implication. They are linked by typed edges, e.g. a claim is *grounded* in a result, *supported by* evidence, and *implies* an implication. Each KU is encoded once by a frozen text encoder (gte-modernbert).

### 2. Split by time: no future papers in training

Papers are split by year: train ≤ 2023, validation 2024, test 2025–2026, with the same research-domain mix in every split (`sci-jepa split`, docs/data.md). Everything below is measured on **test** papers the model never saw. Every test paper is later than the text encoder's training cutoff (2024), and every task is also scored per publication period (`by_period`), to check for contamination.

### 3. A query: mask one KU, predict it

A test instance is the target paper plus N − 1 = 7 related papers as context (N = 8). One KU of the target paper is masked. Its text is removed, and so are the edges that would give it away (e.g. a claim's own evidence: "leak-safe masking"). The model's context encoder reads what is left. Its predictor outputs a vector for the masked position, a guess of what the hidden KU's representation should be.

### 4. Ranking against hard candidates

The prediction is compared, by cosine similarity, with the target encoder's vectors of the following candidates:

* **the true hidden KU**;
* **100 distractors of the same KU type** (`balanced_k`), taken from the papers whose *problem* (context + question) is most similar to the target paper. Every distractor is on-topic, so matching the topic is not enough;
* **counterfactual edits of the true KU**: its own text with the meaning changed but the wording kept (direction reversed, number changed, key term or sentence substituted from a related paper). Edits that a classifier can spot from wording alone are removed beforehand (`mp_filter`; this removes all negations).

The rank of the true KU among them gives:

* **MRR** = mean of 1 / rank; **Hits@1 / Hits@10** = share ranked first / in the top 10;
* **chance MRR** = the MRR of a random ranking of the same pool (≈ 0.05 for 100 candidates);
* **× chance** = MRR ÷ chance MRR, the score used everywhere: 1 = no better than random, ≥ 1.5 = learned;
* **counterfactual robustness** (`mp_pair`) = how often the truth outranks each of its own edits. Chance is 0.5. This is the hard part: a system that only matches topic and wording sits near 0.5.

### 5. The five reasoning tasks

The same model is queried five ways. In each, only the listed KU types stay visible in the target paper. The neighbouring papers stay as they are.

| Task | Hidden | Visible in the paper | Score |
|---|---|---|---|
| Forward prediction | result | problem, method, procedure | MRR ÷ chance |
| Experiment design (inverse) | method / procedure | problem, result | MRR ÷ chance |
| Abduction | claim / question | result, procedure | MRR ÷ chance |
| Consistency checking | the claim–evidence edges | everything else | balanced accuracy of *supports* vs *contradicts* (chance 0.5) |
| Intervention | (forward query) | the procedure or method swapped for another paper's | share of predictions that move toward the donor paper's real result (chance 0.5) |

**Reasoning index** = the mean of the five scores on one × chance scale (consistency and intervention are divided by 0.5). The **macro** index, the primary number, averages over research domains first, so large fields such as Physics or Computer Science don't dominate. Domains with fewer than 20 test queries are left out of the average.

### 6. Comparison systems: same queries, same candidates

Each comparison system answers exactly the same queries against the same candidate pools and counterfactuals. Its query is built from the *visible* KUs. Its ranking uses its own similarity:

| System | How it ranks |
|---|---|
| Frozen text | mean gte-modernbert vector of the visible KUs, cosine |
| BM25 | lexical Okapi BM25 over the visible texts |
| SciNCL, BGE-M3 | scientific / long-context encoders, same protocol as frozen text |
| Supervised | an MLP trained on the train split to map the visible mean vector to the hidden KU's vector (no graph, no JEPA) |
| LLM-Reasoning | a causal LM (e.g. Qwen2.5) scores each candidate by calibrated log-likelihood given the visible paper text (PMI: the score with the paper minus the score without it) |

For consistency, each comparison system gets a logistic-regression probe trained on its frozen vectors (train split).

**Several LLMs.** `LLM_MODELS` (or `sci-jepa run --llm-models ...`) replaces BENCH's one LLM-Reasoning arm with one arm per model, `LLM-Reasoning@<model>`. Every model ranks the same candidates with the same calibrated scoring, from raw text with no chat template, so differences come from the model alone. The default set covers four families, all freely downloadable without a licence gate:

| Family | Model | Parameters | Licence | Note |
|---|---|---|---|---|
| Qwen (Alibaba) | `Qwen/Qwen2.5-7B-Instruct` | 7.6 B | Apache-2.0 | |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | 7.3 B | Apache-2.0 | |
| OLMo (AI2) | `allenai/OLMo-2-1124-7B-Instruct` | 7.3 B | Apache-2.0 | training data is public: contamination can be checked |
| Phi (Microsoft) | `microsoft/Phi-3.5-mini-instruct` | 3.8 B | MIT | a smaller size point |

Llama and Gemma need a licence accepted on Hugging Face first. Each LLM runs in bf16 on one 40 GB slice. `figures/fig13_llms` and the report's LLM page compare them with the best JEPA model.

### 7. Model diagnostics (models only)

* **chi**: within-paper variance share of the target vectors (≈ 0.01 = collapsed, ≈ 0.12 = healthy);
* **partition probe**: how well a linear probe recovers *which paper* a KU came from (a shortcut signal);
* **inverse-relation accuracy**: whether the model recovers the type of a hidden edge;
* **structure reliance**: × chance with vs without the typed edges (does the model use the graph?);
* **nuisance audit**: share of queries where a visible KU is ≥ 0.75 cosine to the hidden one (a paraphrase leak the masking could not remove).

### 8. Seeds, best epoch and hypotheses

Each arm is trained with several seeds. The test uses the weights of the **best validation epoch**, chosen on the macro reasoning index with early stopping. `results` reports mean ± std over seeds. It also tests the pre-registered hypotheses (`src/sci_jepa/evaluation/findings.py`) with 2,000 bootstrap resamples of the test queries pooled over seeds, stratified by domain. A difference is SUPPORTED if its 95% interval is above 0, CONTRADICTED if below, and INCONCLUSIVE otherwise:

| | Hypothesis |
|---|---|
| H1 | the best JEPA model beats the strongest comparison system (reasoning index) |
| H2 | reading the KU graph with a GNN beats the graph transformer |
| H3 | action-conditioned training improves the intervention response |
| H4 | the best JEPA model resists counterfactuals better than the strongest comparison system |
| H5 | the best JEPA model beats the strongest LLM (with several LLMs: also each one, H5-<model>) |
| C | no system loses more than 0.5× chance after the encoder's cutoff (contamination) |

## Reading the numbers

* With few test papers (e.g. 50 at a 500-paper run) scores are noisy and comparison systems saturate (their × chance is very high). Use a split with ≥ 5,000 test papers and 3–5 seeds for conclusions.
* × chance scores from comparison systems are high mostly because distractors are on-topic but *other* papers: wording overlap with the true paper wins. **Counterfactual robustness** removes that advantage, because the edits share the wording. Read the two together.
* The candidate pools, counterfactuals and query instances depend only on the data settings and seeds, never on the model. Every system is ranked on identical queries.
