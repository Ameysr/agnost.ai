import logging
import numpy as np
import torch
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("sentiment_engine")

# Global reference to cache the model in memory
_model = None

def get_optimal_device() -> str:
    """
    Detects the best available hardware accelerator for PyTorch operations.
    Returns 'cuda' for Nvidia GPUs, 'mps' for Apple Silicon, or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the all-MiniLM-L6-v2 model in memory on the optimal hardware device.
    Raises an HTTP 503 error if the model fails to load.
    """
    global _model
    if _model is not None:
        return _model
        
    device = get_optimal_device()
    logger.info(f"Loading SentenceTransformer model 'all-MiniLM-L6-v2' once at startup on device: {device}")
    try:
        # Load the model on detected device
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"SentenceTransformer model loaded successfully on device '{device}'.")
        return _model
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to load SentenceTransformer model: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model is currently unavailable or failed to load: {str(e)}"
        )

def embed_messages(messages: list[str]) -> np.ndarray:
    """
    Generates high-quality sentence embeddings for a list of clean customer queries.
    Uses batch processing of 64 for optimal performance.
    """
    if not messages:
        return np.empty((0, 384))
        
    model = load_embedding_model()
    try:
        logger.info(f"Generating 384-dimensional sentence embeddings for {len(messages)} messages (batch_size=64)")
        # Perform encoding in batches of 64
        embeddings = model.encode(messages, batch_size=64, show_progress_bar=False)
        
        # Ensure it is returned as a NumPy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
        logger.info(f"Embeddings generated successfully. Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sentence embeddings: {str(e)}"
        )
