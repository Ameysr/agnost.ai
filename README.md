# Agnost AI Sentiment Analytics Engine

An AI-powered conversational analytics system that converts raw customer service transcripts into accurate, clustered intent insights and sentiment metrics using a multi-step machine learning pipeline and high-speed LLM topic labeling.

Built to handle real production challenges like high-dimensional vector search degradation, data noise, PII exposure, and slow inference speeds.

## What It Does

Converts raw conversational text into structured intent clusters with dynamic topic descriptions. Scrubs PII (emails and phone numbers) and strips conversational filler words deterministically. Embeds customer queries into 384-dimensional semantic spaces via Sentence-Transformers, avoids the curse of dimensionality by projecting vectors to a lower-dimensional space before clustering, and groups similar messages dynamically without needing a pre-programmed cluster count. Uses an ultra-fast LLM API pipeline to generate action-oriented topic summaries for each cluster and performs dual-threshold batch sentiment classification over all conversations.

## Key Features

### 1. Dense Vector Embeddings
Uses a preloaded, globally cached Sentence-Transformer model (all-MiniLM-L6-v2) for 384-dimensional dense semantic representation. Encoding runs with `batch_size=64`, `convert_to_numpy=True` to skip redundant tensor copies, and `normalize_embeddings=True` to guarantee unit normalization regardless of which model is loaded.

### 2. Curse of Dimensionality Mitigation
Applies non-linear UMAP to project high-dimensional embeddings down to 5 components using a cosine metric. `n_neighbors` scales dynamically with dataset size (`min(15, max(5, N // 10))`) so the local neighbourhood is meaningful at both 100 and 5000 queries.

### 3. Dynamic Density Clustering
Replaces rigid K-Means with HDBSCAN density clustering. HDBSCAN dynamically discovers natural clusters and isolates noisy queries into an "Uncategorized" bucket (cluster ID -1). Cluster grouping uses `defaultdict(list)` to avoid a per-iteration key existence check. Includes a robust fallback to scikit-learn's HDBSCAN implementation for Windows environments where the native package fails to compile.

Adaptive KMeans fallback activates when HDBSCAN noise ratio exceeds 75%. Cluster count is computed as `K = clip(sqrt(N/6), 3, 10)` with `n_init=10` set explicitly to avoid silent behavioral differences across scikit-learn versions.

### 4. Smart Topic Synthesizer
Uses GPT-OSS-120B on the Groq API (achieving over 500 tokens per second) to generate precise, action-oriented topic labels. The prompt is built once before the retry loop so all three retry attempts send identical messages to the LLM. The `AsyncGroq` client is a module-level singleton so no new HTTP connection pool is created per cluster call. Labels are formatted consistently as `Action Verb - Specific Issue` (e.g., "Requesting refund - order never delivered").

### 5. Multi-Layer Security and PII Sanitization
Deterministic cleaning guards customer privacy using compiled regular expressions to scrub emails, phone numbers, bot signatures, and filler words before semantic modeling occurs. Word-count filtering uses `str.count(' ') + 1` instead of `str.split()` to avoid allocating a list just to count words.

### 6. Dual-Threshold Sentiment Classifier
Employs a preloaded DistilBERT-base-uncased sentiment pipeline with a shared `SENTIMENT_BATCH_SIZE = 64` constant used across all inference paths. Sentiment is resolved via an index-based map rather than a string-keyed dict, so duplicate messages (common in support datasets) each receive their own correct sentiment result instead of the last occurrence overwriting earlier ones. Queries scoring below 60% confidence in either direction are categorized as "Mixed."

### 7. Startup Lifespan Preloading
Preloads heavy PyTorch model weights into memory once at server startup via FastAPI's lifespan context. Eliminates cold-start latency on subsequent requests.

### 8. Pipeline Concurrency Guard
An `asyncio.Lock` on the `/api/analyze` endpoint prevents two simultaneous pipeline runs from executing concurrently. A second request while a run is in progress receives HTTP 429 immediately rather than triggering a duplicate UMAP + HDBSCAN + LLM execution.

### 9. NDJSON Partial-Read Dataset Cache
The dataset is cached locally as newline-delimited JSON (one message per line). Reading 100 queries stops after 100 lines, O(limit), instead of loading all 26,000 entries into memory to slice the first 100. On first run the legacy JSON cache is automatically migrated to NDJSON format.

### 10. Dark-Mode Interactive Dashboard
A React 18 and Tailwind CSS dashboard featuring interactive Recharts bar charts colored dynamically by sentiment split ratios, stats cards, and collapsible accordions containing clean conversation samples.

### 11. Performance Monitor
Calculates and displays total ingested message count, cluster count, pipeline processing time in seconds, and message-per-second throughput.

## Architecture

```
                              User
                               |
                        [React + Vite]
                         Dashboard UI
                               |
                          HTTP / REST
                               |
                        [FastAPI Server]
                               |
               +---------------+------------------+
               |                                  |
         [Lifespan Loader]                  [Memory Cache]
      (Preloads ML weights)             (Returns instant view)
               |
         [asyncio.Lock pipeline guard]
         (Rejects concurrent /api/analyze calls with 429)
               |
         [Analytical Ingest Pipeline 6 Stages]
         1. Ingest     (NDJSON cache, O(limit) partial read)
         2. Preprocess (PII scrub, word-count filter, no list alloc)
         3. Embed      (all-MiniLM-L6-v2, batch=64, unit-normalized)
         4. Reduce     (UMAP cosine, n_neighbors scales with N)
         5. Cluster    (HDBSCAN + KMeans fallback, defaultdict grouping)
         6. Label+Sentiment (AsyncGroq singleton + DistilBERT index-map)
               |
    +----------+-----------+----------+
    |          |           |          |
[HuggingFace] [DistilBERT] [Groq API] [MiniLM L6]
 NDJSON Cache  Sentiment   LLM Label   Embeddings
```

## Pipeline Deep Dive

| Node | What It Does | Key Implementation Detail |
| :--- | :--- | :--- |
| Ingestion | Reads up to `limit` lines from NDJSON cache. Falls back to legacy JSON cache (auto-migrates to NDJSON). Falls back to HuggingFace Hub download. Falls back to hardcoded mock queries. | O(limit) partial read. Stops after `limit` lines, never loads the full 26k dataset into memory. |
| Preprocessing | Scrubs emails and phones via compiled regex, removes filler phrases, filters messages under 5 words. | Word count via `str.count(' ') + 1`. Avoids allocating a list just to count. |
| Embedding | Encodes clean queries into 384-dim vectors in batches of 64. | `convert_to_numpy=True` skips tensor copy. `normalize_embeddings=True` is explicit, not implicit. |
| Dimensionality Reduction | UMAP projects 384 dims to 5 dims using cosine metric. | `n_neighbors = min(15, max(5, N // 10))` scales with dataset size, not hardcoded. |
| Density Clustering | HDBSCAN groups dense regions. KMeans fallback activates if noise ratio exceeds 75%. | `n_init=10` explicit. `defaultdict(list)` for grouping. `K = clip(sqrt(N/6), 3, 10)` for fallback. |
| Topic Labeling | AsyncGroq singleton sends cluster samples to GPT-OSS-120B concurrently (Semaphore=2, 3-attempt retry). | Prompt built once before retry loop. All attempts send identical messages. |
| Sentiment Scoring | DistilBERT bulk pass over all messages at once. Results resolved by index, not by message string. | Index-based map handles duplicate messages correctly. Shared `SENTIMENT_BATCH_SIZE = 64` constant. |

## Running Instructions

### One Command Setup with Docker

The fastest way to run the full stack. Requires Docker and Docker Compose.

```bash
# 1. Clone the repo and enter the project root
git clone <repo-url>
cd agnost.ai

# 2. Add your Groq API key to the backend env file
cp sentiment-engine/backend/.env.example sentiment-engine/backend/.env
# Edit sentiment-engine/backend/.env and set GROQ_API_KEY=gsk_your_key_here

# 3. Start everything
docker compose up --build
```

That's it. Docker builds both containers, installs all dependencies, and starts the services.

On first startup the backend downloads the DistilBERT and MiniLM models from HuggingFace (~500MB total). This takes 2-5 minutes once. The HuggingFace cache is stored in a named Docker volume so subsequent `docker compose up` runs skip the download entirely.

Once the backend prints `All heavy pipeline ML models successfully loaded into memory and ready`, the frontend is available at `http://localhost:5173`.

To stop everything: `docker compose down`

To stop and wipe the model cache volume: `docker compose down -v`

### Manual Setup (without Docker)

### Prerequisites

Python 3.11+, Node.js 18+ and npm, and a Groq API key from [console.groq.com](https://console.groq.com).

### Backend Setup

```bash
cd sentiment-engine/backend

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

On Windows, `hdbscan` may fail to compile. The pipeline automatically falls back to scikit-learn's built-in HDBSCAN so you can skip it and continue:

```bash
pip install -r requirements.txt --ignore-installed hdbscan
```

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_groq_key_here
```

```bash
uvicorn main:app --host 127.0.0.1 --port 8050 --reload
```

Wait for the startup sequence to print `All heavy pipeline ML models successfully loaded into memory and ready.` The backend is now listening at `http://127.0.0.1:8050`. Interactive API docs are at `http://127.0.0.1:8050/docs`.

### Frontend Setup

Open a second terminal:

```bash
cd sentiment-engine/frontend

npm install

# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Edit `.env` and point the API URL to port 8050. The example file defaults to 8000 which is wrong:

```env
VITE_API_URL=http://localhost:8050
```

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/api/analyze` | Run the full pipeline. Body: `{"limit": 100}`. Returns 429 if a run is already in progress. |
| GET | `/api/insights` | Return cached results from the last run. Returns 404 if no run has completed. |
| GET | `/api/health` | Return server status and model load state. |
| GET | `/docs` | Interactive Swagger UI. |
