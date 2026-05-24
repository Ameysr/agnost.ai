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
