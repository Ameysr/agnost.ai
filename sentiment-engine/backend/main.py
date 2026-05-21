import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import our pipeline modules and schemas
from data.loader import load_customer_support_dataset
from pipeline.preprocess import preprocess_conversations
from pipeline.embed import embed_messages, load_embedding_model
from pipeline.cluster import cluster_messages
from pipeline.label import generate_cluster_label_with_llm
from pipeline.sentiment import analyze_cluster_sentiment, load_sentiment_pipeline
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
    Executes the entire end-to-end Sentiment Analytics pipeline:
    1. Downloads/Loads raw conversation dataset (Bitext HuggingFace)
    2. Cleans text, strips agent, normalizes and filters PII / short items
    3. Encodes queries into 384-dim dense vectors (all-MiniLM-L6-v2)
    4. Projects vectors to 5 dims with UMAP and clusters with HDBSCAN
    5. Summarizes topic intent with Llama3 on Groq API
    6. Scores message sentiments with DistilBERT and aggregates
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
    logger.info(f"Received request to run full pipeline with limit={body.limit}")
    
    try:
        # Stage 1: Load Raw Dataset
        raw_queries = load_customer_support_dataset(limit=body.limit)
        
        # Stage 2: Preprocess & Filter
        clean_queries = preprocess_conversations(raw_queries)
        if not clean_queries:
            logger.warning("Preprocessing filtered out all raw queries.")
            raise HTTPException(
                status_code=500,
                detail="All messages were filtered out during preprocessing. Check data quality or increase limit."
            )
            
        # Stage 3: Embedding
        embeddings = embed_messages(clean_queries)
        
        # Stage 4: Clustering
        clusters = cluster_messages(embeddings, clean_queries)
        
        # Stage 5 & 6: LLM Labeling and Sentiment Aggregation per Cluster
        insights = []
        total_conversations = len(clean_queries)
        
        for cluster_id, cluster_msgs in clusters.items():
            count = len(cluster_msgs)
            pct = round((count / total_conversations) * 100, 1)
            
            # Sub-sample 3 messages for the React dashboard frontend
            # We take the first 3 after preprocessing
            sample_messages = cluster_msgs[:3]
            
            # Call Groq to generate action-oriented label
            label = generate_cluster_label_with_llm(cluster_id, cluster_msgs)
            
            # Analyze sentiment distribution
            sentiment_summary = analyze_cluster_sentiment(cluster_msgs)
            
            insight = ClusterInsight(
                cluster_id=cluster_id,
                label=label,
                message_count=count,
                percentage_of_total=pct,
                sentiment=sentiment_summary["overall"],
                positive_pct=sentiment_summary["positive_pct"],
                negative_pct=sentiment_summary["negative_pct"],
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
        logger.info(f"Pipeline executed successfully in {duration}s. Results cached.")
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
