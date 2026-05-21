import logging
from typing import Dict, List, TypedDict
from transformers import pipeline

logger = logging.getLogger("sentiment_engine")

# Global reference to cache the HuggingFace pipeline in memory
_classifier = None

class SentimentResult(TypedDict):
    positive_pct: float
    negative_pct: float
    overall: str

def load_sentiment_pipeline():
    """
    Loads and caches the DistilBERT sentiment pipeline in memory once.
    """
    global _classifier
    if _classifier is not None:
        return _classifier
        
    logger.info("Loading HuggingFace sentiment analysis pipeline ('distilbert-base-uncased-finetuned-sst-2-english')...")
    try:
        # Load the pipeline. CPU is standard, and we enable native batching if supported
        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("Sentiment pipeline loaded successfully.")
        return _classifier
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to load sentiment analysis pipeline: {str(e)}")
        # We don't raise 503 here directly so we can catch it at main.py startup lifespan
        raise e

def analyze_cluster_sentiment(messages: List[str]) -> SentimentResult:
    """
    Performs sentiment analysis on a list of messages within a single cluster.
    Runs the HuggingFace pipeline with optimal batching.
    
    Categorizes the cluster overall:
    - >60% positive => "positive"
    - >60% negative => "negative"
    - Otherwise => "mixed"
    """
    if not messages:
        return {"positive_pct": 0.0, "negative_pct": 0.0, "overall": "mixed"}
        
    classifier = load_sentiment_pipeline()
    total = len(messages)
    
    try:
        logger.info(f"Scoring sentiment for {total} messages in batch")
        # Run the pipeline over the entire list in one go (batch size 32 is optimal for memory/speed on CPU)
        results = classifier(messages, batch_size=32)
        
        positive_count = 0
        negative_count = 0
        
        for res in results:
            label = res["label"].upper()
            if "POSITIVE" in label:
                positive_count += 1
            else:
                negative_count += 1
                
        # Calculate percentages
        positive_pct = round((positive_count / total) * 100, 1)
        negative_pct = round((negative_count / total) * 100, 1)
        
        # Determine overall cluster category based on the 60% rules
        if negative_pct > 60.0:
            overall = "negative"
        elif positive_pct > 60.0:
            overall = "positive"
        else:
            overall = "mixed"
            
        logger.info(f"Sentiment result calculated: Pos={positive_pct}%, Neg={negative_pct}%, Overall='{overall}'")
        
        return {
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "overall": overall
        }
        
    except Exception as e:
        logger.error(f"Failed to score sentiment for cluster: {str(e)}")
        # Fail gracefully by returning mixed
        return {
            "positive_pct": 50.0,
            "negative_pct": 50.0,
            "overall": "mixed"
        }
