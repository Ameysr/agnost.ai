import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Limit PyTorch to 2 CPU threads to prevent system overheating
import torch
torch.set_num_threads(2)

# Import our pipeline modules and schemas
from starlette.concurrency import run_in_threadpool
from data.loader import load_customer_support_dataset
from pipeline.preprocess import preprocess_conversations
from pipeline.embed import embed_messages, load_embedding_model
from pipeline.cluster import cluster_messages
from pipeline.label import generate_cluster_label_with_llm, label_all_clusters_async
from pipeline.sentiment import analyze_cluster_sentiment, load_sentiment_pipeline, analyze_messages_sentiment_bulk
from models.schemas import AnalyzeRequest, AnalysisResult, ClusterInsight

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sentiment_engine")

# Load environment variables
load_dotenv()

# Global memory cache for pipeline results
_analysis_cache: AnalysisResult | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events to preload the heavy machine learning models
    once at server startup to prevent latency on subsequent requests.
    """
    logger.info("Initializing Sentiment Analytics Engine startup sequence...")
    try:
        # Load Sentence-Transformer embedder
        load_embedding_model()
        
        # Load DistilBERT sentiment pipeline
        load_sentiment_pipeline()
        
        app.state.models_loaded = True
        logger.info("All heavy pipeline ML models successfully loaded into memory and ready.")
    except Exception as e:
        logger.critical(f"Startup sequence failed. Could not preload models: {str(e)}")
        app.state.models_loaded = False
        
    yield
    logger.info("Shutting down Sentiment Analytics Engine. Releasing cached resources.")

app = FastAPI(
    title="Sentiment Analytics Engine API",
    description="Production-grade conversational cluster feedback analyzer for Agnost AI",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Vite dev server (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(body: AnalyzeRequest = AnalyzeRequest()):
    """
    Executes the entire end-to-end Sentiment Analytics pipeline using production-grade thread offloading:
    1. Downloads/Loads raw conversation dataset (uses local cache for instant retrieval if available)
    2. Cleans text and filters PII / short items (threaded)
    3. Encodes queries into 384-dim dense vectors on optimal GPU/CPU (threaded)
    4. Projects vectors and clusters with HDBSCAN and KMeans fallback if quality is low (threaded)
    5. Scores all queries in a single high-throughput batch pass (threaded)
    6. Summarizes topic intent with safe-throttled AsyncGroq concurrent API calls
    7. Caches the result in memory
    """
    global _analysis_cache
    
    # Assert models are preloaded successfully
    if not getattr(app.state, "models_loaded", False):
        logger.error("POST /api/analyze called but models not preloaded.")
        raise HTTPException(
             status_code=503,
             detail="Machine learning components are unavailable (startup loading failed)."
        )
        
    start_time = time.time()
    logger.info(f"Received request to run optimized pipeline with limit={body.limit}")
    
    try:
        # Stage 1: Load Raw Dataset (uses local cache internally)
        raw_queries = await run_in_threadpool(load_customer_support_dataset, limit=body.limit)
        
        # Stage 2: Preprocess & Filter (CPU-heavy, offloaded to thread pool)
        clean_queries = await run_in_threadpool(preprocess_conversations, raw_queries)
        if not clean_queries:
            logger.warning("Preprocessing filtered out all raw queries.")
            raise HTTPException(
                status_code=500,
                detail="All messages were filtered out during preprocessing. Check data quality or increase limit."
            )
            
        # Stage 3: Embedding generation (CPU/GPU-heavy, offloaded to thread pool)
        embeddings = await run_in_threadpool(embed_messages, clean_queries)
        
        # Stage 4: Dimensionality Reduction and Clustering (CPU-heavy, offloaded to thread pool)
        clusters = await run_in_threadpool(cluster_messages, embeddings, clean_queries)
        
        # Stage 5: Unified Bulk Sentiment scoring (Inference bulk batch pass offloaded to thread pool)
        all_sentiments = await run_in_threadpool(analyze_messages_sentiment_bulk, clean_queries)
        
        # Map each query string back to its predicted sentiment dict
        sentiment_map = {msg: res for msg, res in zip(clean_queries, all_sentiments)}
        
        # Stage 6: Concurrent Async LLM Labeling (Network-bound, safe-throttled using asyncio)
        labels = await label_all_clusters_async(clusters)
        
        # Compile all cluster insights
        insights = []
        total_conversations = len(clean_queries)
        
        for cluster_id, cluster_msgs in clusters.items():
            count = len(cluster_msgs)
            pct = round((count / total_conversations) * 100, 1)
            sample_messages = cluster_msgs[:3]
            
            # Lookup concurrent async generated label
            label = labels.get(cluster_id, f"Cluster {cluster_id} - label unavailable")
            
            # Aggregate preloaded bulk sentiment records for messages in this cluster
            pos_count = 0
            neg_count = 0
            for msg in cluster_msgs:
                res = sentiment_map.get(msg, {"label": "POSITIVE"})
                label_val = res["label"].upper()
                if "POSITIVE" in label_val:
                    pos_count += 1
                else:
                    neg_count += 1
                    
            cluster_len = len(cluster_msgs)
            pos_pct = round((pos_count / cluster_len) * 100, 1) if cluster_len > 0 else 0.0
            neg_pct = round((neg_count / cluster_len) * 100, 1) if cluster_len > 0 else 0.0
            
            # Define overall sentiment class based on 60% thresholds
            if neg_pct > 60.0:
                overall_sentiment = "negative"
            elif pos_pct > 60.0:
                overall_sentiment = "positive"
            else:
                overall_sentiment = "mixed"
                
            insight = ClusterInsight(
                cluster_id=cluster_id,
                label=label,
                message_count=count,
                percentage_of_total=pct,
                sentiment=overall_sentiment,
                positive_pct=pos_pct,
                negative_pct=neg_pct,
                sample_messages=sample_messages
            )
            insights.append(insight)
            
        # Sort insights by volume (message_count) descending
        insights.sort(key=lambda x: x.message_count, reverse=True)
        
        # Compute processing duration
        duration = round(time.time() - start_time, 2)
        
        # Build response payload
        result = AnalysisResult(
            total_conversations=total_conversations,
            total_clusters=len(clusters),
            processing_time_seconds=duration,
            insights=insights
        )
        
        # Cache results in-memory
        _analysis_cache = result
        logger.info(f"Optimized pipeline executed successfully in {duration}s. Results cached.")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("An unhandled exception occurred in the analytical pipeline.")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed: {str(e)}"
        )

@app.get("/api/insights", response_model=AnalysisResult)
async def get_insights():
    """
    Returns the cached analytics results of the last pipeline execution.
    If no pipeline has been run yet, returns a 404.
    """
    global _analysis_cache
    if _analysis_cache is None:
        logger.info("GET /api/insights requested but cache is empty.")
        raise HTTPException(
            status_code=404,
            detail="Run /analyze first. No cached analysis results found."
        )
    return _analysis_cache

@app.get("/api/health")
async def health():
    """
    Simple health check endpoint returning system status and preloaded models flag.
    """
    models_loaded = getattr(app.state, "models_loaded", False)
    return {
        "status": "ok",
        "model_loaded": models_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8050, reload=True)
