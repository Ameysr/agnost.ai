import os
import json
import logging
from typing import List
from datasets import load_dataset

logger = logging.getLogger("sentiment_engine")

# Define persistent local cache path relative to this file
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CACHE_DIR, "cached_support_dataset.json")

def load_customer_support_dataset(limit: int = 500) -> List[str]:
    """
    Loads raw customer messages.
    First tries to read from a local disk cache for instant startup and offline resilience.
    If no cache exists, downloads from HuggingFace Hub, caches it locally, and returns the queries.
    """
    # 1. Try reading from local disk cache
    if os.path.exists(CACHE_PATH):
        logger.info(f"Local dataset cache found at {CACHE_PATH}. Reading cached dataset...")
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                all_messages = json.load(f)
            logger.info(f"Successfully loaded {len(all_messages)} messages from local disk cache.")
            return all_messages[:limit]
        except Exception as e:
            logger.warning(f"Failed to read local cache, falling back to network download: {str(e)}")

    logger.info(f"No valid local cache found. Attempting to download from HuggingFace dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    try:
        # Load the dataset from HuggingFace Hub
        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        
        # Access the main train split
        if "train" not in dataset:
            available_splits = list(dataset.keys())
            logger.warning(f"Train split not found. Available splits: {available_splits}. Using the first split: {available_splits[0]}")
            split_data = dataset[available_splits[0]]
        else:
            split_data = dataset["train"]
            
        # Extract the instruction column (raw user messages) and cast to a list for JSON serializability
        all_messages = list(split_data["instruction"])
        total_available = len(all_messages)
        logger.info(f"Successfully loaded dataset from HuggingFace. Total records: {total_available}")
        
        # Cache the entire dataset locally for future offline runs
        try:
            logger.info(f"Writing dataset to persistent disk cache at {CACHE_PATH}...")
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(all_messages, f, ensure_ascii=False, indent=2)
            logger.info("Persistent disk cache successfully written.")
        except Exception as write_err:
            logger.error(f"Failed to write local dataset cache: {str(write_err)}")
            
        return all_messages[:limit]

    except Exception as e:
        logger.error(f"Failed to load dataset from HuggingFace: {str(e)}")
        # Provide a fallback mock list of customer support messages in case of API failure or network issues
        logger.info("Using local fallback mock customer queries for resilience.")
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
        # Multiply to reach a reasonable size if a larger amount was requested
        result = []
        while len(result) < limit:
            result.extend(fallback_queries)
        return result[:limit]

