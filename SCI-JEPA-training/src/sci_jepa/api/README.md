# Sci-JEPA API, first version

A FastAPI server for a trained SCI-JEPA model. It follows the draft *Sci-JEPA API Specification* (11 Nov 2026) for sections 1 to 7. There is no authentication yet.

## Run it

```bash
cd SCI-JEPA-training
uv pip install --python .venv/bin/python -e '.[api]'     # once: fastapi + uvicorn

# local / interactive node
.venv/bin/python -m sci_jepa.api --model <model folder> --index auto --port 8000
# or: sci-jepa api --model <model folder>

# a 2080 Ti on the cluster (submit from the repository root, so log/ exists)
MODEL=<model folder> sbatch SCI-JEPA-training/deploy/api/serve_2080ti.sbatch
```

`<model folder>` is a folder that training wrote (`train.save_model`), holding `config.json` and `model.safetensors`. With `--index auto`, the server finds the KU index the model was trained on under `$PAPER_A_INDEX_DIR`. You can also pass the index `.pt` file, or `--index none` to run the model without a corpus (only consistency, edit check and the intervention shift then work).

The interactive docs are at `/v1/docs`, and the OpenAPI file is at `/v1/openapi.json`. `/v1/health` answers straight away. `/v1/ready` returns 503 until the model, the index and the search are all loaded.

**First start with the full corpus.** The server builds one vector per paper, the centred mean of its KUs' text embeddings. That takes one pass over `comp_x` (61M × 768), which can take a while on NFS. The result is cached as `api_paper_vecs_<index>.npz` next to the index, or in `--cache-dir` if the index folder is read-only.

## Endpoints

| Spec section | Endpoints | State |
|---|---|---|
| 2 System | `GET /v1/health`, `/v1/ready`, `/v1/info`, `/v1/models`, `/v1/tasks` | done |
| 3 Main flow | `POST /v1/ask`, `POST /v1/sessions/{id}/answers`, `GET`/`DELETE /v1/sessions/{id}` | done; sessions in memory |
| 4 Building blocks | `POST /v1/parse`, `POST /v1/validate` | done |
| 5 Tasks | `POST /v1/predict/{forward,inverse,abduction}`, `/v1/consistency`, `/v1/edit-check`, `/v1/intervention` | done |
| 6 Papers | `GET /v1/papers/{id}`, `/v1/papers/search`, `/v1/kus/{id}`, `POST /v1/ask/paper` | done; search matches ID prefixes only (the index has no titles) |
| 7 Input checks | format, content, PII | done; names are not yet detected as PII |
| 1 Conventions | X-Request-Id, RFC 9457 errors, rate limits + `RateLimit-*` headers, `Idempotency-Key`, `X-Model-Id` | done (`guard.py`); per client address or bearer token |
| 10 Service metrics | `GET /metrics` (Prometheus) | done: requests, latency, GPU queue, OOMs, CUDA memory |
| 8, 9, 10 (quality) | ORKG login, API keys, dashboard, drift and quality monitoring | not in this version |

## Robustness on one GPU (`guard.py`)

One model on one GPU answers one request at a time. The server protects itself as follows:

| Risk | What the server does |
|---|---|
| Burst of requests | GPU work goes through a queue of `--gpu-queue` (16) waiting requests. A request that finds the queue full, or waits longer than `--gpu-wait` (30 s), gets **503 `overloaded`** with `Retry-After`. uvicorn refuses connections far beyond that (`limit_concurrency`). |
| Health checks starved | `/v1/health`, `/v1/ready` and `/metrics` run on the event loop, not in the thread pool, so they answer while every worker thread waits for the GPU. The thread pool has room for the queue plus 24 threads. |
| CUDA out of memory | The CUDA cache is emptied and the request retried once. If it fails again: **503 `gpu_out_of_memory`**, `Retry-After: 5`. The server stays ready. |
| CUDA fault (illegal memory access, device assert) | The CUDA context is broken: `/v1/ready` turns 503 and model requests get **503 `gpu_fault`**. With `--exit-on-cuda-fault` (set in the sbatch script) the process exits with code 3 and the sbatch loop restarts it (at most `MAX_RESTARTS`, 5). |
| One client hogging the model | Per client per minute: `--rate-gpu` (60) on model endpoints, `--rate-read` (600) on the others; **429 `rate_limited`** with `Retry-After`. `/v1/health`, `/v1/ready` and `/metrics` are never limited. Behind your own reverse proxy, pass `--trust-proxy`. |
| Client retries | `Idempotency-Key` on any POST: the same key and body replays the first answer (`Idempotent-Replayed: true`), the same key with another body is 422, a retry during the first run is 409. 5xx answers are not stored, so a retry runs again. `DELETE /v1/sessions/{id}` also deletes stored replays of that session. |
| Huge bodies | Bodies above `--max-body` (64 KB) get **413** before any parsing, including chunked uploads. |
| Memory growth | At most `--max-sessions` (10,000) open sessions (503 `too_many_sessions` above); expired sessions and idempotency entries are swept. |
| Concurrent answers to one session | Each session has its own lock; a second answer while the first runs gets 409 `session_busy`. |
| Warm-up racing requests | The warm-up holds the model lock. |
| Log injection | A client `X-Request-Id` is kept only if it matches `[A-Za-z0-9._:-]{1,128}`. |

All this state lives in the one server process. Several replicas behind a load balancer would need shared storage (for example Redis) for the limiter, idempotency keys and sessions.

## How answers are ranked

1. **Shortlist.** The server embeds the user's parts with the model's frozen text encoder and keeps the 64 most similar corpus papers (`--shortlist`), after the year and field filters.
2. **Score.** It then scores those papers' KUs of the answer type:
   - `jepa`: the cosine between Sci-JEPA's predicted vector for the hidden part and each candidate's target-encoder key, with the paper encoded alone. This is the rule the evaluation uses.
   - `text`: the cosine of the frozen text embeddings.
   - `bm25`: BM25 over the shortlist.
3. **Combine.** `hybrid` = 0.6 `jepa` + 0.3 `text` + 0.1 `bm25`, each min-max scaled within the request. Scores are relative within one request.

## Measured on the cluster (12 Nov 2026)

Setup: R-GAT model `arxiv_full_M/BENCH/jepa+rgat_seed0` on one RTX 2080 Ti (`gpu-t2080ti-01`), with index `b40bd0e47bd9` (2,066,075 papers, 44,653,582 KUs).

**Start-up**

| Step | Time |
|---|---|
| Load the model | 115 s (mostly hashing the 922 MB weights file over NFS) |
| Load the index | 319 s |
| Build the paper vectors, first start only | 2.5 min, cached as a 6.3 GB `api_paper_vecs_*.npz` |

**Latency per request, after warm-up**

| Request | Time |
|---|---|
| Forward, inverse or abduction | 0.6 to 0.9 s |
| Consistency, with 5 related evidence KUs | 1.2 s |
| Edit check | 0.06 to 0.5 s |

**Accuracy.** `/v1/ask/paper` with `reveal_truth` was run on 24 validation papers per task. The true KU was answer #1 for 71% (forward), 83% (inverse) and 79% (abduction). This is not the benchmark protocol: here the candidates are every answer-type KU in the 64 shortlisted papers.

## IDs and fields

- **Paper IDs.** `paper_id` is the arXiv ID when the paper has one. Corpus papers without one get `p_<n>`, an index row that is stable for one index version. The index stores an internal file path for these papers, and the API never shows it.
- **Fields.** `field` is one of the spec's 10 canonical domains (the index's `paper_domain`), and the filters use it too. `subfield` is the fine label from the summary.

## Known gaps

- **Splitter.** The splitter in `/v1/ask` and `/v1/parse` is a cue-word placeholder (`textparts.split_parts`), not the language model the spec calls for. `understood` always shows what it read, and `corrections` fixes it.
- **Answer text.** Answers show the index's 240-character snippets. Use `--full-texts` to load the full texts, which needs a lot of memory at 61M KUs.
- **Shortlist limits.** A paper with more than 24 KUs (`max_comp_per_paper`) is encoded from a fixed sample of 24, so its other KUs cannot be answers.
- **Edit check without context.** Without `context`, `/v1/edit-check` reports how far apart the model puts the two findings (`margin` = 1 − cosine, separated at ≥ 0.02, a starting value). With `context` (problem, method, conditions), it uses the evaluation's rule: does the prediction prefer the original?
- **Intervention scale.** `shift` is the distance between the predicted results before and after the change. It is scaled by the mean distance to the predicted results of 16 corpus papers.
- **Thresholds.** The thresholds are the spec's starting values, except `problem.min_words`, which is 6 instead of 8 so that the spec's own examples pass.
