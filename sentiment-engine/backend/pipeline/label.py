import os
import random
import time
import logging
import asyncio
from typing import Dict, List
from groq import Groq, AsyncGroq

logger = logging.getLogger("sentiment_engine")

async def generate_cluster_label_with_llm_async(
    cluster_id: int, 
    messages: List[str], 
    semaphore: asyncio.Semaphore
) -> str:
    """
    Generates a single-sentence PM-ready summary for a list of messages in a cluster
    using the AsyncGroq client concurrently.
    Respects rate limits using a strict asyncio.Semaphore concurrency limit.
    """
    if cluster_id == -1:
        return "Uncategorized - Miscellaneous user queries"
        
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("gsk_your_groq_api_key"):
        logger.warning(f"GROQ_API_KEY is missing or invalid. Skipping async LLM for Cluster {cluster_id}")
        return f"Cluster {cluster_id} - label unavailable due to missing API key"

    sampled_messages = random.sample(messages, min(10, len(messages)))
    formatted_messages = "\n".join([f"- {msg}" for msg in sampled_messages])
    
    prompt = f"""You are a product analytics assistant. 
Below are customer messages from a single conversation cluster.
Summarize what this cluster of users wants or is complaining about
in ONE sentence. Format: "[Action verb] - [specific issue]"
Examples: 
"Requesting refund - order never delivered"
"Reporting bug - payment fails at checkout"

Messages:
{formatted_messages}

Respond with ONLY the one-line summary. No explanation. No markdown styling. No extra words."""

    # Execute inside semaphore concurrency lock to protect free tier rate limits
    async with semaphore:
        client = AsyncGroq()
        for attempt in range(3):
            try:
                logger.info(f"Sending Cluster {cluster_id} to Groq API concurrently (Attempt {attempt + 1}/3)")
                response = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1,
                    max_completion_tokens=512,
                    top_p=1,
                    reasoning_effort="medium",
                    stop=None
                )
                
                label = response.choices[0].message.content.strip()
                label = label.strip('"').strip("'").strip("`").strip()
                
                if label:
                    logger.info(f"Successfully generated async label for Cluster {cluster_id}: '{label}'")
                    return label
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/3 failed for Cluster {cluster_id}: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(1.5) # Wait 1.5s (async-friendly) before retrying
                    
    logger.error(f"All 3 async LLM labeling attempts failed for Cluster {cluster_id}. Using fallback.")
    return f"Cluster {cluster_id} - label unavailable"

async def label_all_clusters_async(clusters: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Asynchronously labels all clusters in parallel while safe-throttling requests
    via asyncio.Semaphore(2) to comply with Groq free-tier rate limits.
    """
    logger.info(f"Initiating concurrent labeling for {len(clusters)} clusters...")
    
    # Standard free tier safety semaphore of 2 concurrent connections
    semaphore = asyncio.Semaphore(2)
    
    tasks = []
    cluster_ids = list(clusters.keys())
    
    for cid in cluster_ids:
        task = generate_cluster_label_with_llm_async(cid, clusters[cid], semaphore)
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    
    labels: Dict[int, str] = {}
    for cid, label in zip(cluster_ids, results):
        labels[cid] = label
        
    return labels

def generate_cluster_label_with_llm(cluster_id: int, messages: List[str]) -> str:
    """
    Legacy synchronous PM-ready summary generator kept for unit test compatibility.
    """
    if cluster_id == -1:
        return "Uncategorized - Miscellaneous user queries"
        
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("gsk_your_groq_api_key"):
        logger.warning(f"GROQ_API_KEY is missing or invalid. Skipping LLM for Cluster {cluster_id}")
        return f"Cluster {cluster_id} - label unavailable due to missing API key"

    sampled_messages = random.sample(messages, min(10, len(messages)))
    formatted_messages = "\n".join([f"- {msg}" for msg in sampled_messages])
    
    prompt = f"""You are a product analytics assistant. 
Below are customer messages from a single conversation cluster.
Summarize what this cluster of users wants or is complaining about
in ONE sentence. Format: "[Action verb] - [specific issue]"
Examples: 
"Requesting refund - order never delivered"
"Reporting bug - payment fails at checkout"

Messages:
{formatted_messages}

Respond with ONLY the one-line summary. No explanation. No markdown styling. No extra words."""

    client = Groq()
    
    for attempt in range(3):
        try:
            logger.info(f"Sending Cluster {cluster_id} to Groq API (Attempt {attempt + 1}/3)")
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_completion_tokens=512,
                top_p=1,
                reasoning_effort="medium",
                stop=None
            )
            
            label = response.choices[0].message.content.strip()
            label = label.strip('"').strip("'").strip("`").strip()
            
            if label:
                logger.info(f"Successfully generated label for Cluster {cluster_id}: '{label}'")
                return label
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/3 failed for Cluster {cluster_id}: {str(e)}")
            if attempt < 2:
                time.sleep(1)
                
    logger.error(f"All 3 LLM labeling attempts failed for Cluster {cluster_id}. Using fallback.")
    return f"Cluster {cluster_id} - label unavailable"

def label_all_clusters(clusters: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Legacy synchronous labeling loop kept for test suite compatibility.
    """
    logger.info(f"Generating labels for {len(clusters)} clusters...")
    labels: Dict[int, str] = {}
    for cluster_id, messages in clusters.items():
        labels[cluster_id] = generate_cluster_label_with_llm(cluster_id, messages)
    return labels
