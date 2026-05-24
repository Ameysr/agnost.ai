import os
import math
import json
import logging
from typing import List
from datasets import load_dataset

logger = logging.getLogger("sentiment_engine")

# Define persistent local cache path relative to this file
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CACHE_DIR, "cached_support_dataset.json")
# Newline-delimited JSON cache: one message per line for O(limit) reads instead of O(total)
NDJSON_CACHE_PATH = os.path.join(CACHE_DIR, "cached_support_dataset.ndjson")

def _read_ndjson_cache(limit: int) -> List[str]:
    """
    Reads up to `limit` lines from the NDJSON cache file.
    O(limit) reads — stops as soon as we have enough messages instead of
    loading the entire dataset into memory first.
    """
    messages = []
    with open(NDJSON_CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
            if len(messages) >= limit:
                break
    return messages


def _write_ndjson_cache(all_messages: List[str]) -> None:
    """
    Writes messages to the NDJSON cache: one JSON-encoded string per line.
    Allows partial reads without loading the full file.
    """
    with open(NDJSON_CACHE_PATH, "w", encoding="utf-8") as f:
        for msg in all_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def load_customer_support_dataset(limit: int = 500) -> List[str]:
    """
    Loads raw customer messages.
    Priority order:
      1. NDJSON cache  — O(limit) partial read, no full-file load
      2. Legacy JSON cache — full load with slice (kept for backward compat)
      3. HuggingFace Hub download — writes NDJSON cache for future runs
      4. Hardcoded fallback — offline resilience
    """
    # 1. Fast path: NDJSON cache allows reading exactly `limit` lines
    if os.path.exists(NDJSON_CACHE_PATH):
        logger.info(f"NDJSON cache found. Reading up to {limit} messages (partial read)...")
        try:
            messages = _read_ndjson_cache(limit)
            logger.info(f"Loaded {len(messages)} messages from NDJSON cache (read only what was needed).")
            return messages
        except Exception as e:
            logger.warning(f"Failed to read NDJSON cache, trying legacy JSON cache: {str(e)}")

    # 2. Legacy JSON cache fallback (loads full file — kept for backward compat)
    if os.path.exists(CACHE_PATH):
        logger.info(f"Legacy JSON cache found at {CACHE_PATH}. Reading and migrating to NDJSON...")
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                all_messages = json.load(f)
            logger.info(f"Loaded {len(all_messages)} messages from legacy cache. Migrating to NDJSON...")
            try:
                _write_ndjson_cache(all_messages)
                logger.info("Migration to NDJSON cache complete. Future reads will be faster.")
            except Exception as migrate_err:
                logger.warning(f"NDJSON migration failed (non-fatal): {str(migrate_err)}")
            return all_messages[:limit]
        except Exception as e:
            logger.warning(f"Failed to read legacy JSON cache, falling back to network download: {str(e)}")

    # 3. Download from HuggingFace Hub
    logger.info("No valid local cache found. Downloading from HuggingFace Hub...")
    try:
        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

        if "train" not in dataset:
            available_splits = list(dataset.keys())
            logger.warning(f"Train split not found. Available: {available_splits}. Using: {available_splits[0]}")
            split_data = dataset[available_splits[0]]
        else:
            split_data = dataset["train"]

        all_messages = list(split_data["instruction"])
        logger.info(f"Downloaded {len(all_messages)} messages from HuggingFace Hub.")

        # Write NDJSON cache for fast partial reads on future runs
        try:
            logger.info(f"Writing NDJSON cache to {NDJSON_CACHE_PATH}...")
            _write_ndjson_cache(all_messages)
            logger.info("NDJSON cache written successfully.")
        except Exception as write_err:
            logger.error(f"Failed to write NDJSON cache: {str(write_err)}")

        return all_messages[:limit]

    except Exception as e:
        logger.error(f"Failed to load dataset from HuggingFace: {str(e)}")

    # 4. Hardcoded fallback for offline resilience
    logger.info("Using hardcoded fallback queries for resilience.")
    fallback_queries = [
        "How do I cancel my subscription and get a refund?",
        "I want to track my order number #10382, it has not arrived yet.",
        "Can you help me reset my password? I am locked out.",
        "The payment failed at checkout, can I try another card?",
        "My package arrived but it is damaged and broken.",
        "Do you offer international shipping to the UK?",
        "I want to delete my account and erase my personal data.",
        "I got double charged for my monthly subscription fees.",
        "Is there a phone number I can call for live customer support?",
        "The mobile app keeps crashing whenever I try to log in.",
        "I need to change my delivery address for order 4928.",
        "How do I apply a promo code to my existing order?",
        "Are there any discounts available for students or teachers?",
        "My discount code is not working at checkout, please help.",
        "I did not receive a confirmation email for my purchase."
    ]
    # Single multiplication instead of a while-extend loop
    repeats = math.ceil(limit / len(fallback_queries))
    return (fallback_queries * repeats)[:limit]

