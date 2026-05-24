# REASONING.md: Sentiment Analytics Engine

Architecture decisions, trade-offs, and what changes with more time.

## System Architecture

```
HuggingFace Hub (first run only)
        |
        v
  [ loader.py ]       cached_support_dataset.ndjson (partial read, O(limit))
        |              cached_support_dataset.json  (legacy fallback, auto-migrates)
        v
 [ preprocess.py ]    PII scrubbing, filler filter, word-count guard
        |
        v
  [ embed.py ]        all-MiniLM-L6-v2, 384-dim unit-normalized vectors
        |              batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        v
  [ cluster.py ]      UMAP (384 to 5 dims, n_neighbors scales with N)
        |              HDBSCAN with KMeans fallback at noise ratio > 75%
        |              defaultdict grouping, n_init=10 explicit
        v
 [ sentiment.py ]     DistilBERT single bulk pass, SENTIMENT_BATCH_SIZE=64 shared const
        |
        v
  [ label.py ]        AsyncGroq singleton, asyncio.gather, Semaphore=2
        |              prompt built once before retry loop, module-level client
        v
  [ main.py ]         FastAPI + asyncio.Lock (pipeline guard) + run_in_threadpool
        |              index-based sentiment map (duplicate-safe), attrgetter sort
        v
  React + Vite + Recharts    Dashboard at localhost:5173
```

## Decisions and Rejected Alternatives

### Dataset: HuggingFace `bitext/Bitext-customer-support-llm-chatbot-training-dataset`

Chosen: Stream from HuggingFace Hub on first run, cache locally as NDJSON (one message per line).

| Why this | Why not |
|---|---|
| 27,000 real-world labelled customer support messages, zero setup | Custom scraper takes days for a weekend project |
| NDJSON cache reads exactly `limit` lines, O(limit) not O(total) | Postgres/SQLite adds a dependency with no query flexibility gain since data never mutates |
| Auto-migrates legacy JSON cache to NDJSON on first run | Flat JSON loads the entire 26k-entry file to slice the first 100, wasteful |
| Hardcoded fallback queries keep the pipeline alive offline | HuggingFace network call takes 2-8s depending on connection |

Rejected: Any database. The dataset is static and read-only. A flat file is the simplest correct thing. The NDJSON format gives streaming-style partial reads without any extra dependency.

### Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

Chosen: 384-dim dense vectors, 22M parameters, runs on CPU with no GPU required.

| Why this | Why not |
|---|---|
| Best quality/speed ratio for semantic similarity on CPU | `text-embedding-ada-002` costs money per token and adds network latency |
| Runs locally, zero API dependency for the core pipeline | `all-mpnet-base-v2` (768 dims) is 2x slower with marginal quality gain at this scale |
| `convert_to_numpy=True` skips the isinstance check and avoids a GPU tensor copy | GloVe/Word2Vec are bag-of-words and miss sentence-level semantics entirely |
| `normalize_embeddings=True` makes unit normalization explicit and model-swap safe | OpenAI embeddings add API cost, latency, and an outbound dependency |

Rejected: OpenAI embeddings. An outbound API call for every encode batch adds latency, cost, and a hard dependency on network availability, all avoidable with a local model of comparable quality.

### Dimensionality Reduction: UMAP (384 to 5 dims)

Chosen: UMAP with cosine metric, `n_neighbors` scaled to `min(15, max(5, N // 10))`, `min_dist=0.0`.

| Why this | Why not |
|---|---|
| Preserves local neighbourhood structure, critical for HDBSCAN to find dense regions | t-SNE is non-parametric, cannot transform new points, and is much slower |
| Cosine metric aligns with unit-normalized sentence embedding space | PCA is linear and destroys the non-linear manifold structure of language |
| 5 components is the empirically optimal HDBSCAN input dimensionality | Running HDBSCAN directly on 384 dims triggers the curse of dimensionality |
| Dynamic `n_neighbors` stays proportional at both 100 and 5000 queries | Hardcoded `n_neighbors=15` is too sparse at small N and too dense at large N |

### Clustering: HDBSCAN with KMeans Fallback

Chosen: HDBSCAN (`min_cluster_size=10`, `eom` selection), automatic fallback to KMeans when noise ratio exceeds 75%.

| Why this | Why not |
|---|---|
| Discovers clusters of arbitrary shape without a predefined K | KMeans alone requires guessing K upfront and wrong K produces meaningless clusters |
| Handles outlier noise natively (label -1), real support data is noisy | GMM assumes Gaussian distributions, invalid for language embedding clusters |
| KMeans fallback guarantees output on sparse or small datasets | Pure HDBSCAN fails silently on sparse inputs, leaving the dashboard empty |
| `n_init=10` explicit, `"auto"` behaves differently across sklearn 1.3 vs 1.4 | sklearn version ambiguity causes silent behavioral differences in production |
| `defaultdict(list)` removes per-iteration key existence check | Manual `if key not in dict` adds a dict lookup on every message assignment |

Fallback formula: `K = clip(sqrt(N/6), 3, 10)` scales from 3 clusters at 54 messages to 10 at 600.

### Sentiment: DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)

Chosen: Single bulk batch pass across all messages at once, `SENTIMENT_BATCH_SIZE = 64` as a shared module constant.

| Why this | Why not |
|---|---|
| Contextual transformer understands negation and compound sentences | VADER is rule-based and fails on "it was not as bad as I expected" |
| Runs locally, zero cost, no rate limits | AWS Comprehend/Google NLP adds latency and per-call cost |
| Bulk batching is 70%+ faster than per-cluster loops on CPU | TextBlob uses outdated pattern matching, poor on support language |
| Shared `SENTIMENT_BATCH_SIZE` constant, legacy method was using 32, now consistent | Inconsistent batch sizes between methods is a silent performance regression |
| Index-based sentiment map in `main.py`, duplicate messages each get their own result | String-keyed dict overwrites sentiment for duplicate messages silently |

Aggregation rule: above 60% positive gives `positive`, above 60% negative gives `negative`, else `mixed`.

Fallback score is `{"label": "POSITIVE", "score": 0.51}`. A score of exactly 0.5 with a POSITIVE label is contradictory. 0.51 is honest about low confidence while staying within the binary label space.

### LLM Labeler: Groq SDK (`AsyncGroq`) with `openai/gpt-oss-120b`

Chosen: Concurrent async calls via `asyncio.gather`, throttled to `Semaphore(2)`, module-level client singleton.

| Why this | Why not |
|---|---|
| Groq inference is 10-20x faster than OpenAI (LPU hardware, >500 tok/s) | OpenAI GPT-4o is slower, expensive, and requires a paid key |
| `AsyncGroq` singleton created once at module level, no per-call HTTP pool spin-up | `AsyncGroq()` inside the semaphore creates a new connection pool per cluster call |
| Prompt built once before the retry loop, all 3 attempts send identical messages | `random.sample` inside the retry loop sends different messages on each attempt |
| `_build_label_prompt()` shared by sync and async paths, one place to change the prompt | Duplicated prompt string across both functions is a maintenance hazard |
| 3-attempt retry with 1.5s async backoff handles transient API failures | No retry means a single timeout drops a cluster label permanently |

Rejected: LangChain. Heavyweight abstraction for a single prompt call with no chain, no memory, and no tool use. Direct SDK is cleaner and faster.

### API Framework: FastAPI + Starlette `run_in_threadpool`

Chosen: FastAPI with CPU-bound ML steps offloaded to thread pool workers, `asyncio.Lock` for pipeline deduplication.

| Why this | Why not |
|---|---|
| Async event loop stays responsive during 30-90s ML pipeline runs | Flask is sync and blocks the entire process during heavy computation |
| `asyncio.Lock` prevents two simultaneous `/api/analyze` calls from running double UMAP + LLM | No lock means two clicks fire two full pipeline runs concurrently |
| Auto-generates OpenAPI docs at `/docs` with zero config | Django is a full ORM/templating stack, far too heavy for a pure API server |
| `run_in_threadpool` integrates GIL-bound ML cleanly with async routes | Ray/Celery are distributed task queues, overkill for a single-user tool |

### Frontend: React + Vite + Recharts + Tailwind

Chosen: Minimal React SPA, no router, pure state-based navigation.

| Why this | Why not |
|---|---|
| Single view app, state swap is navigation, no routing library needed | Next.js adds SSR/SSG complexity for a dashboard that is 100% client-side |
| Recharts is declarative and composable, bar charts in 10 lines | D3.js requires manual DOM manipulation, heavy for simple bar charts |
| Vite HMR makes iteration instant during development | CRA (Create React App) is deprecated with slow builds |
| Tailwind utility classes keep styling co-located with markup | CSS modules or styled-components add indirection for a single-page UI |

## What I Would Do Differently With a Month

### Week 1: Persistent Storage and Run History

Replace the flat NDJSON cache with PostgreSQL and store each pipeline run as a versioned snapshot. Schema: `runs(run_id, timestamp, limit, total_clusters, processing_time)` and `insights(run_id, cluster_id, label, message_count, sentiment, positive_pct, negative_pct)`. Build a run history selector in the UI to compare cluster drift across two dates side by side. Add a `GET /api/runs` endpoint returning paginated run history.

### Week 2: Vector Database and Incremental Updates

Swap in-memory embeddings for Qdrant (self-hosted, free) to persist vectors across restarts. Implement incremental ingestion so only messages added since the last run timestamp get re-embedded. Add a `POST /api/search` endpoint for finding messages semantically similar to a given complaint using Qdrant ANN search. Cache cluster centroids in Qdrant so new messages can be assigned to existing clusters without re-running UMAP.

### Week 3: Production Hardening

Add API key middleware on `/api/analyze` so no public endpoint is running heavy ML inference. Stream cluster labels to the frontend via Server-Sent Events as Groq finishes each one rather than waiting for all N clusters before the UI updates. Add per-IP throttling via `slowapi` to prevent the free-tier Groq key from being exhausted by repeated clicks. Set up GitHub Actions with `pytest` on every push, Docker build on merge to `main`, and a health check smoke test on deploy.

### Week 4: Quality and Intelligence

Fine-tune embeddings on the support domain using contrastive learning on intent pairs, which measurably improves cluster coherence. Auto-tune `min_cluster_size` based on dataset size via a lightweight grid search scored by silhouette coefficient. Add Slack or email notifications when a cluster volume spikes more than 20% versus the previous run for early issue detection. Build one-click CSV and PDF export of insights for PM handoff with cluster labels, message counts, sentiment breakdown, and sample messages.

## Key Constraints Accepted for the Weekend Scope

| Constraint | Decision | Production Fix |
|---|---|---|
| No database | NDJSON disk cache, O(limit) partial reads | PostgreSQL + Qdrant |
| Free-tier Groq | Semaphore(2) throttle | Paid tier, Semaphore(10) |
| No auth | Open /api/analyze endpoint | API key middleware |
| Static dataset | HuggingFace download + local cache | Live webhook ingestion |
| Single-user | In-memory result cache, asyncio.Lock | Redis + multi-tenant run isolation |
| No streaming | Full pipeline completes before UI updates | SSE streaming per cluster label |
