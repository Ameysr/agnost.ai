import logging
from collections import defaultdict
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
        # HDBSCAN suffers in high-dimensional spaces (384 dims) due to the curse of dimensionality.
        # n_neighbors is scaled proportionally to dataset size so the local neighborhood
        # is meaningful at both small (100) and large (5000+) query counts.
        n_neighbors = min(15, max(5, len(clean_messages) // 10))
        logger.info(f"Applying UMAP (n_components=5, n_neighbors={n_neighbors}, metric='cosine')")

        reducer = umap.UMAP(
            n_components=5,
            n_neighbors=n_neighbors,
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
            
        # STEP 3: Quality Evaluation & Adaptive Fallback to KMeans
        total_items = len(cluster_labels)
        noise_count = int(np.sum(cluster_labels == -1))
        noise_ratio = noise_count / total_items if total_items > 0 else 0.0
        unique_labels = set(cluster_labels)
        non_noise_clusters = len(unique_labels - {-1})
        
        logger.info(f"Clustering evaluation metrics -> Total Items: {total_items}, Noise Count: {noise_count} ({noise_ratio:.1%}), Clean Clusters: {non_noise_clusters}")
        
        # If HDBSCAN produces 0 clusters (excluding noise) or leaves > 75% of messages as noise, K-Means is activated
        if non_noise_clusters == 0 or noise_ratio > 0.75:
            logger.warning(f"HDBSCAN resulted in poor quality (noise ratio {noise_ratio:.1%} > 75% or 0 clusters). Dynamically falling back to KMeans clustering...")
            from sklearn.cluster import KMeans
            # Determine dynamic number of clusters: k = sqrt(total_items / 6), bounded between 3 and 10
            k = int(np.clip(np.sqrt(total_items / 6), 3, 10))
            logger.info(f"Executing KMeans fallback with K={k} clusters...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_embeddings)
            logger.info("KMeans fallback clustering completed successfully (no noise categories produced).")
            
        # Group messages by their cluster ID using defaultdict to avoid
        # a per-iteration key existence check.
        clusters: Dict[int, List[str]] = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[int(label)].append(clean_messages[idx])
            
        # Summarize cluster statistics
        logger.info(f"Clustering completed. Found {len(clusters)} clusters.")
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
