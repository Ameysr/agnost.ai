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
    - convert_to_numpy=True avoids a potential extra memory copy when the model
      runs on GPU and returns a torch.Tensor.
    - normalize_embeddings=True guarantees unit-normalized vectors regardless of
      which model is loaded (all-MiniLM-L6-v2 normalizes by default, but being
      explicit protects against silent regressions if the model is swapped).
    """
    if not messages:
        return np.empty((0, 384))

    model = load_embedding_model()
    try:
        logger.info(f"Generating 384-dimensional sentence embeddings for {len(messages)} messages (batch_size=64)")
        embeddings = model.encode(
            messages,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,       # skip isinstance check + avoids extra copy
            normalize_embeddings=True,   # explicit unit normalization
        )
        logger.info(f"Embeddings generated successfully. Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sentence embeddings: {str(e)}"
        )
