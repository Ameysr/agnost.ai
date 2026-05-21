# Sentiment Analytics Engine 📊

A production-grade conversational intent discovery and sentiment clustering pipeline built for **Agnost AI**. This engine extracts user intents and complaints from raw customer service transcripts, clusters them semantically, dynamically names topic categories using an LLM, aggregates user sentiments, and visualizes them on a sleek, dark-mode real-time dashboard.

---

## 🛠️ System Overview & Architecture

The Sentiment Analytics Engine is engineered to solve a common product-management challenge: **turning unstructured chat transcripts into actionable intent summaries**. The pipeline is modularized and executes as follows:

```
  [Raw Conversations JSON/CSV]
               ↓
    [Preprocessing & Cleaning]      <- Strips bots, fillers, PII (email/phone)
               ↓
   [Sentence Embedding (MiniLM)]    <- Preloaded once; 384-dim dense encoding (batch_size=64)
               ↓
  [UMAP Dimensionality Reduction]   <- Projects high-dim space to 5 components (cosine distance)
               ↓
       [HDBSCAN Clustering]        <- Identifies dense shapes, filters noise to "Uncategorized"
               ↓
  [LLM Topic Labeling via Groq]     <- Connects to llama3-8b-8192 for action-based names
               ↓
   [Sentiment Scoring (DistilBERT)] <- Classifies user sentiments in batches of 32
               ↓
    [FastAPI Memory-Cached Server]  <- Caches heavy pipelines, serves endpoints instantly
               ↓
  [React 18 + TS + Tailwind CSS]    <- Sleek dashboard featuring Recharts intent volumes
```

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.11+** installed
- **Node.js 18+** & **npm** installed
- A free **Groq API Key** (Obtain in 10 seconds at [console.groq.com](https://console.groq.com/))

### 1. Backend Setup
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd sentiment-engine/backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables. Copy the `.env.example` template:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and paste your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_key_here
   ```
5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   The backend will preload the embedding and sentiment models on startup. Once you see `Application startup complete.` in your logs, the server is listening at `http://127.0.0.1:8000`. You can inspect the interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

---

### 2. Frontend Setup
1. Open a second terminal window and navigate to the frontend directory:
   ```bash
   cd sentiment-engine/frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Create the environment file:
   ```bash
   cp .env.example .env
   ```
4. Boot up the Vite developer server:
   ```bash
   npm run dev
   ```
5. Open your web browser and navigate to `http://localhost:5173`.

---

## 🎨 Architectural & Design Decisions

### 1. Why HDBSCAN over K-Means?
- **No Fixed Clusters ($K$):** Standard K-Means requires specifying $K$ upfront, which is impossible in a dynamic customer feedback setting where the number of topics shifts daily. HDBSCAN dynamically discovers the optimal number of clusters based on density patterns.
- **Noise Awareness:** Real conversations contain filler, spam, or outliers. K-Means forces every outlier into a cluster, polluting clean topics. HDBSCAN identifies noise and segregates it into a special outlier group (labeled as `cluster_id: -1`), which we label as `"Uncategorized"`.
- **Arbitrary Shapes:** K-Means assumes spherical clusters. Conversation semantics are complex and construct arbitrary shapes in vector space; HDBSCAN handles variable-density, non-spherical shapes flawlessly.

### 2. Why UMAP before Clustering?
- **Curse of Dimensionality:** In 384-dimensional vector spaces (output of Sentence-Transformers), distance metrics like Euclidean space degrade because "everything becomes equally far apart." HDBSCAN struggles to identify core density in high dimensions.
- **Non-Linear Projection:** Principal Component Analysis (PCA) only does linear projections. UMAP preserves both local and global semantic structural relations while projecting vectors down to 5 dimensions, which HDBSCAN can then cluster rapidly and accurately.

### 3. Why all-MiniLM-L6-v2?
- **Extreme Speed vs Quality:** `all-MiniLM-L6-v2` delivers over 80% of the semantic clustering quality of large BERT models while running **5x to 10x faster** on standard consumer CPUs.
- **Compact Dimensions:** The output embeddings are 384-dimensional (compared to 768 or 1536 for larger models), leading to low memory overhead, extremely fast UMAP calculations, and swift operational throughput.

### 4. Why Groq for LLM Labeling?
- **Blazing Fast Throughput:** Groq API running `llama3-8b-8192` achieves speeds of over **500 tokens per second**. This keeps our pipeline from bottle-necking on API responses.
- **Accurate Action Verb Summaries:** Llama3 is highly receptive to system prompt constraints, consistently delivering the exact `"[Action verb] — [specific issue]"` format (e.g., `"Requesting refund — order never delivered"`) without redundant conversational explanations.

### 5. Conversational Scaling to 10M+ Records
At 10M conversations, we would move from in-memory caching to Redis with TTL, replace batch UMAP+HDBSCAN with online clustering (BIRCH or streaming k-means), store embeddings in a vector DB (Qdrant or Weaviate) for incremental nearest-neighbor assignment, and run the pipeline on a nightly Airflow DAG rather than on-demand.

---

## 📸 Sample Visual Dashboard Flow

When you load the Agnost AI Sentiment Analytics Dashboard, you are presented with:
1. **The Launch Center:** If no analysis has run yet, you see a glowing, glassmorphic CTA prompting you to trigger the conversational engine.
2. **Dynamic ML Ingest Steps:** Triggering the run locks the UI in a blur modal showing a rolling ticker detailing the active pipelines (Scrubbing PII, Embedding batches, UMAP projection, Groq calls).
3. **KPI Statistics Cards:** Displays high-level summaries including cleaned message volume, active clustered intent groups, processing times, and system message-per-second throughput.
4. **Interactive Intent Distribution Chart:** A Recharts bar chart showing message counts per topic, dynamically colored by overall sentiment (emerald green for positive, rose red for negative, amber yellow for mixed). Hovering over any bar reveals a custom dark tooltip showing the full LLM cluster description.
5. **Insights Grid:** Modular, expandable cards displaying:
   - Dynamic, action-oriented topic title.
   - Percentage of total conversation share.
   - Dual-colored horizontal split progress track comparing positive and negative ratios.
   - Expanding message accordions that display actual conversation samples in italicized message bubbles.
