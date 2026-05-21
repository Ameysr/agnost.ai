import logging
from typing import Dict, List
import numpy as np
from fastapi import HTTPException
import umap

logger = logging.getLogger("sentiment_engine")

def cluster_messages(embeddings: np.ndarray, clean_messages: List[str]) -> Dict[int, List[str]]:
    """
    Performs clustering on sentence embeddings:
    1. Reduces dimension to 5 using UMAP (recommending cosine metric).
    2. Clusters using HDBSCAN (euclidean metric, 'eom' cluster selection).
    3. Groups noise/outlier points (-1) into an 'Uncategorized' bucket.
    
    Returns a dictionary mapping cluster_id (int) to a list of message strings.
    """
    if len(clean_messages) == 0:
        return {}
        
    logger.info(f"Starting dimensionality reduction and clustering on {len(clean_messages)} messages")
    
    try:
        # Step 1: Dimensionality Reduction using UMAP
        # HDBSCAN suffers in high-dimensional spaces (384 dims) due to the curse of dimensionality
        logger.info("Applying UMAP dimensionality reduction (n_components=5, n_neighbors=15, metric='cosine')")
        
        # We set random_state for reproducible clustering results
        reducer = umap.UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(f"UMAP reduction finished. Reduced shape: {reduced_embeddings.shape}")
        
    except Exception as e:
        logger.error(f"UMAP dimensionality reduction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Dimensionality reduction failed in cluster pipeline: {str(e)}"
        )
        
    try:
        # Step 2: Clustering using HDBSCAN
        logger.info("Applying HDBSCAN clustering (min_cluster_size=10, metric='euclidean', selection_method='eom')")
        
        cluster_labels = None
        
        # Try to import and use the standard hdbscan library
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=10,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            cluster_labels = clusterer.fit_predict(reduced_embeddings)
            logger.info("HDBSCAN clustering finished using standard hdbscan library.")
        except ImportError:
            # Fall back to scikit-learn's built-in HDBSCAN if hdbscan package couldn't compile on Windows
            logger.warning("hdbscan library not found, falling back to sklearn.cluster.HDBSCAN")
            from sklearn.cluster import HDBSCAN as SKHDBSCAN
            clusterer = SKHDBSCAN(
                min_cluster_size=10,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            cluster_labels = clusterer.fit_predict(reduced_embeddings)
            logger.info("HDBSCAN clustering finished using scikit-learn fallback.")
            
        # Group messages by their cluster ID
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(cluster_labels):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(clean_messages[idx])
            
        # Summarize cluster statistics
        logger.info(f"Clustering completed. Found {len(clusters)} clusters (including noise cluster -1 if present).")
        for cid, msgs in clusters.items():
            name = "Uncategorized/Noise" if cid == -1 else f"Cluster {cid}"
            logger.info(f"  - {name}: {len(msgs)} messages")
            
        return clusters
        
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Clustering failed in cluster pipeline: {str(e)}"
        )
