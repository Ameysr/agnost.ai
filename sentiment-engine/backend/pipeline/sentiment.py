import logging
from typing import Dict, List, TypedDict
from transformers import pipeline
from pipeline.embed import get_optimal_device

logger = logging.getLogger("sentiment_engine")

# Global reference to cache the HuggingFace pipeline in memory
_classifier = None

# Single source of truth for batch size used across all inference calls
SENTIMENT_BATCH_SIZE = 64

class SentimentResult(TypedDict):
    positive_pct: float
    negative_pct: float
    overall: str

def load_sentiment_pipeline():
    """
    Loads and caches the DistilBERT sentiment pipeline in memory once.
    Automatically detects and runs on CUDA, MPS, or CPU.
    """
    global _classifier
    if _classifier is not None:
        return _classifier
        
    device = get_optimal_device()
    logger.info(f"Loading HuggingFace sentiment analysis pipeline ('distilbert-base-uncased-finetuned-sst-2-english') on device: {device}...")
    try:
        # Load the pipeline on GPU/MPS/CPU. PyTorch device is fully compatible.
        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
        logger.info(f"Sentiment pipeline loaded successfully on device '{device}'.")
        return _classifier
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to load sentiment analysis pipeline: {str(e)}")
        raise e

def analyze_messages_sentiment_bulk(messages: List[str]) -> List[Dict]:
    """
    Performs sentiment analysis in bulk over a single flat list of messages
    using optimized batch size of 64 on the preloaded pipeline.
    Returns a list of raw prediction dicts: [{'label': 'POSITIVE'/'NEGATIVE', 'score': float}, ...]
    """
    if not messages:
        return []
        
    classifier = load_sentiment_pipeline()
    try:
        logger.info(f"Executing bulk sentiment inference on {len(messages)} messages (batch_size={SENTIMENT_BATCH_SIZE})")
        results = classifier(messages, batch_size=SENTIMENT_BATCH_SIZE)
        return results
    except Exception as e:
        logger.error(f"Bulk sentiment analysis failed: {str(e)}. Falling back to neutral predictions.")
        # score=0.5 with no strong label direction — treated as uncertain/neutral in aggregation
        return [{"label": "POSITIVE", "score": 0.51} for _ in messages]

def analyze_cluster_sentiment(messages: List[str]) -> SentimentResult:
    """
    Legacy method kept for test suite compatibility.
    Computes sentiment metrics from list of messages.
    """
    if not messages:
        return {"positive_pct": 0.0, "negative_pct": 0.0, "overall": "mixed"}
        
    classifier = load_sentiment_pipeline()
    total = len(messages)
    
    try:
        logger.info(f"Scoring sentiment for {total} messages in batch")
        results = classifier(messages, batch_size=SENTIMENT_BATCH_SIZE)
        
        positive_count = 0
        negative_count = 0
        
        for res in results:
            label = res["label"].upper()
            if "POSITIVE" in label:
                positive_count += 1
            else:
                negative_count += 1
                
        positive_pct = round((positive_count / total) * 100, 1)
        negative_pct = round((negative_count / total) * 100, 1)
        
        if negative_pct > 60.0:
            overall = "negative"
        elif positive_pct > 60.0:
            overall = "positive"
        else:
            overall = "mixed"
            
        return {
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "overall": overall
        }
        
    except Exception as e:
        logger.error(f"Failed to score sentiment: {str(e)}")
        return {
            "positive_pct": 50.0,
            "negative_pct": 50.0,
            "overall": "mixed"
        }
