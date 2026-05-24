import re
import logging
from typing import List

logger = logging.getLogger("sentiment_engine")

# Regular expressions for PII detection
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}")

# Standalone filler phrases to be discarded if they appear by themselves
FILLER_PHRASES = {
    "hi", "hello", "okay", "ok", "thank you", "thanks", "hey", "bye", "goodbye", "yes", "no", "please"
}

def clean_message(message: str) -> str:
    """
    Cleans a single user message:
    - Removes agent/bot prefixes if present.
    - Strips email and phone PII.
    - Standardizes whitespace and converts to lowercase.
    """
    if not message:
        return ""
    
    # Strip agent or bot labels if they are somehow embedded (e.g. "User: hello" or "Agent: hi")
    # Keep only user turns, discarding agent messages entirely
    lower_msg = message.strip()
    if lower_msg.lower().startswith(("agent:", "bot:", "system:", "assistant:")):
        return ""
        
    if lower_msg.lower().startswith("user:"):
        message = message[5:]
        
    # Replace email PII with empty space
    message = EMAIL_REGEX.sub(" ", message)
    
    # Replace phone PII with empty space
    message = PHONE_REGEX.sub(" ", message)
    
    # Lowercase and normalize whitespace
    message = " ".join(message.lower().split())
    
    return message

def preprocess_conversations(messages: List[str]) -> List[str]:
    """
    Preprocesses a list of raw conversations:
    - Cleans individual messages.
    - Filters out standalone filler words.
    - Filters out messages with fewer than 5 words.
    """
    logger.info(f"Preprocessing starting on {len(messages)} raw messages")
    cleaned_messages = []
    
    for idx, raw_msg in enumerate(messages):
        # 1. Clean basic content (PII, labels, casing)
        cleaned = clean_message(raw_msg)
        if not cleaned:
            continue
            
        # 2. Filter standalone filler phrases (e.g., "hello", "thank you")
        if cleaned in FILLER_PHRASES:
            continue
            
        # 3. Filter messages with fewer than 5 words.
        # count(' ') + 1 is faster than split() because it avoids allocating a list.
        if cleaned.count(" ") + 1 < 5:
            continue

        cleaned_messages.append(cleaned)
        
    logger.info(f"Preprocessing completed. Kept {len(cleaned_messages)} out of {len(messages)} messages.")
    return cleaned_messages
