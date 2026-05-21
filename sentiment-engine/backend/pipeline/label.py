import os
import random
import time
import logging
from typing import Dict, List
from groq import Groq

logger = logging.getLogger("sentiment_engine")

def generate_cluster_label_with_llm(cluster_id: int, messages: List[str]) -> str:
    """
    Generates a single-sentence PM-ready summary for a list of messages in a cluster
    using the Groq Llama3-8b-8192 model.
    Implements retry logic (2 retries with 1s delay) and a standard fallback.
    """
    # Noise cluster (-1) does not represent a cohesive group, so we bypass LLM
    if cluster_id == -1:
        return "Uncategorized — Miscellaneous user queries"
        
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Degrade gracefully if no Groq API Key is configured
    if not api_key or api_key.startswith("gsk_your_groq_api_key"):
        logger.warning(f"GROQ_API_KEY is missing or invalid. Skipping LLM for Cluster {cluster_id}")
        return f"Cluster {cluster_id} — label unavailable due to missing API key"

    # Sample up to 10 random messages to stay well within token limits and keep speed high
    sampled_messages = random.sample(messages, min(10, len(messages)))
    formatted_messages = "\n".join([f"- {msg}" for msg in sampled_messages])
    
    prompt = f"""You are a product analytics assistant. 
Below are customer messages from a single conversation cluster.
Summarize what this cluster of users wants or is complaining about
in ONE sentence. Format: "[Action verb] — [specific issue]"
Examples: 
"Requesting refund — order never delivered"
"Reporting bug — payment fails at checkout"

Messages:
{formatted_messages}

Respond with ONLY the one-line summary. No explanation. No markdown styling. No extra words."""

    client = Groq(api_key=api_key)
    
    # Try the LLM call, retry up to 2 times (total 3 attempts) with a 1-second delay
    for attempt in range(3):
        try:
            logger.info(f"Sending Cluster {cluster_id} to Groq API (Attempt {attempt + 1}/3)")
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a direct, concise product feedback summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=64
            )
            
            label = response.choices[0].message.content.strip()
            # Clean up the output if the model added wrapping quotes or trailing periods
            label = label.strip('"').strip("'").strip("`").strip()
            
            # Simple check to make sure the model didn't return an empty string
            if label:
                logger.info(f"Successfully generated label for Cluster {cluster_id}: '{label}'")
                return label
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/3 failed for Cluster {cluster_id}: {str(e)}")
            if attempt < 2:
                time.sleep(1) # wait 1s before retrying
                
    # Fallback if all attempts failed
    logger.error(f"All 3 LLM labeling attempts failed for Cluster {cluster_id}. Using fallback.")
    return f"Cluster {cluster_id} — label unavailable"

def label_all_clusters(clusters: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Iterates through all clusters and generates a representative label for each.
    """
    logger.info(f"Generating labels for {len(clusters)} clusters...")
    labels: Dict[int, str] = {}
    
    for cluster_id, messages in clusters.items():
        labels[cluster_id] = generate_cluster_label_with_llm(cluster_id, messages)
        
    return labels
