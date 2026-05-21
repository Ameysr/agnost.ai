import logging
from typing import List
from datasets import load_dataset

logger = logging.getLogger("sentiment_engine")

def load_customer_support_dataset(limit: int = 500) -> List[str]:
    """
    Loads raw customer messages from the Bitext HuggingFace dataset.
    Extracts the 'instruction' field representing the user's queries.
    """
    logger.info(f"Attempting to load {limit} records from HuggingFace dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset")
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
            
        # Extract the instruction column (raw user messages)
        messages = split_data["instruction"]
        total_available = len(messages)
        logger.info(f"Successfully loaded dataset. Total records available: {total_available}")
        
        # Apply the limit
        sampled_messages = messages[:limit]
        logger.info(f"Sampled {len(sampled_messages)} messages for analysis.")
        
        return sampled_messages

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
