import os
import random
import time
import logging
import asyncio
from typing import Dict, List
from groq import Groq, AsyncGroq

logger = logging.getLogger("sentiment_engine")

# Module-level singleton clients — created once, reused across all calls.
# Avoids spinning up a new HTTP connection pool for every cluster label request.
_sync_groq_client: Groq | None = None
_async_groq_client: AsyncGroq | None = None


def _get_sync_client() -> Groq:
    global _sync_groq_client
    if _sync_groq_client is None:
        _sync_groq_client = Groq()
    return _sync_groq_client


def _get_async_client() -> AsyncGroq:
    global _async_groq_client
    if _async_groq_client is None:
        _async_groq_client = AsyncGroq()
    return _async_groq_client


def _build_label_prompt(messages: List[str]) -> str:
    """
    Builds the LLM prompt for cluster labeling.
    Extracted into a single function so both sync and async paths share
    identical prompt logic — change it once, affects both.

    Sampling is done here (before any retry loop) so all retry attempts
    send the exact same messages to the LLM, making retries true retries
    rather than different requests.
    """
    sampled = random.sample(messages, min(10, len(messages)))
    formatted = "\n".join([f"- {msg}" for msg in sampled])
    return (
        "You are a product analytics assistant.\n"
        "Below are customer messages from a single conversation cluster.\n"
        "Summarize what this cluster of users wants or is complaining about\n"
        "in ONE sentence. Format: \"[Action verb] - [specific issue]\"\n"
        "Examples:\n"
        "\"Requesting refund - order never delivered\"\n"
        "\"Reporting bug - payment fails at checkout\"\n\n"
        f"Messages:\n{formatted}\n\n"
        "Respond with ONLY the one-line summary. No explanation. No markdown styling. No extra words."
    )


async def generate_cluster_label_with_llm_async(
    cluster_id: int,
    messages: List[str],
    semaphore: asyncio.Semaphore
) -> str:
    """
    Generates a single-sentence PM-ready summary for a cluster using AsyncGroq.
    - Uses the module-level AsyncGroq singleton (no per-call client creation).
    - Prompt is built once before the retry loop so all attempts are identical.
    - Respects rate limits via asyncio.Semaphore.
    """
    if cluster_id == -1:
        return "Uncategorized - Miscellaneous user queries"

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("gsk_your_groq_api_key"):
        logger.warning(f"GROQ_API_KEY missing or invalid. Skipping async LLM for Cluster {cluster_id}")
        return f"Cluster {cluster_id} - label unavailable due to missing API key"

    # Build prompt once — all retry attempts send the same messages
    prompt = _build_label_prompt(messages)
    client = _get_async_client()

    async with semaphore:
        for attempt in range(3):
            try:
                logger.info(f"Sending Cluster {cluster_id} to Groq API concurrently (Attempt {attempt + 1}/3)")
                response = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                    max_completion_tokens=512,
                    top_p=1,
                    reasoning_effort="medium",
                    stop=None
                )

                label = response.choices[0].message.content.strip()
                label = label.strip('"').strip("'").strip("`").strip()

                if label:
                    logger.info(f"Generated async label for Cluster {cluster_id}: '{label}'")
                    return label

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/3 failed for Cluster {cluster_id}: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(1.5)  # async-friendly backoff

    logger.error(f"All 3 async LLM attempts failed for Cluster {cluster_id}. Using fallback.")
    return f"Cluster {cluster_id} - label unavailable"


async def label_all_clusters_async(clusters: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Asynchronously labels all clusters in parallel, throttled via
    asyncio.Semaphore(2) to comply with Groq free-tier rate limits.
    """
    logger.info(f"Initiating concurrent labeling for {len(clusters)} clusters...")

    semaphore = asyncio.Semaphore(2)
    cluster_ids = list(clusters.keys())

    tasks = [
        generate_cluster_label_with_llm_async(cid, clusters[cid], semaphore)
        for cid in cluster_ids
    ]
    results = await asyncio.gather(*tasks)

    return dict(zip(cluster_ids, results))


def generate_cluster_label_with_llm(cluster_id: int, messages: List[str]) -> str:
    """
    Legacy synchronous PM-ready summary generator kept for unit test compatibility.
    - Uses the module-level Groq singleton.
    - Prompt is built once before the retry loop.
    """
    if cluster_id == -1:
        return "Uncategorized - Miscellaneous user queries"

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("gsk_your_groq_api_key"):
        logger.warning(f"GROQ_API_KEY missing or invalid. Skipping LLM for Cluster {cluster_id}")
        return f"Cluster {cluster_id} - label unavailable due to missing API key"

    # Build prompt once — all retry attempts send the same messages
    prompt = _build_label_prompt(messages)
    client = _get_sync_client()

    for attempt in range(3):
        try:
            logger.info(f"Sending Cluster {cluster_id} to Groq API (Attempt {attempt + 1}/3)")
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_completion_tokens=512,
                top_p=1,
                reasoning_effort="medium",
                stop=None
            )

            label = response.choices[0].message.content.strip()
            label = label.strip('"').strip("'").strip("`").strip()

            if label:
                logger.info(f"Generated label for Cluster {cluster_id}: '{label}'")
                return label

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/3 failed for Cluster {cluster_id}: {str(e)}")
            if attempt < 2:
                time.sleep(1)

    logger.error(f"All 3 LLM attempts failed for Cluster {cluster_id}. Using fallback.")
    return f"Cluster {cluster_id} - label unavailable"


def label_all_clusters(clusters: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Legacy synchronous labeling loop kept for test suite compatibility.
    """
    logger.info(f"Generating labels for {len(clusters)} clusters...")
    return {
        cluster_id: generate_cluster_label_with_llm(cluster_id, messages)
        for cluster_id, messages in clusters.items()
    }
