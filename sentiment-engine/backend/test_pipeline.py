import sys
import os

# Append current directory to Python load paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Limit PyTorch to 2 CPU threads to prevent CPU overheating
import torch
torch.set_num_threads(2)

import unittest
import numpy as np
from pipeline.preprocess import preprocess_conversations
from pipeline.embed import embed_messages, load_embedding_model
from pipeline.cluster import cluster_messages
from pipeline.sentiment import analyze_cluster_sentiment, load_sentiment_pipeline

class TestSentimentAnalyticsPipeline(unittest.TestCase):
    """
    Unit test suite validating text cleaning mathematical projections,
    dense sentence embeddings, HDBSCAN clustering, and DistilBERT sentiment aggregates.
    """

    def setUp(self):
        # Prevent logging noise in test outputs
        import logging
        logging.getLogger("sentiment_engine").setLevel(logging.WARNING)

    def test_preprocessing_and_pii_scrubbing(self):
        """
        Validates text scrubbing logic:
        - Drops messages under 5 words.
        - Wipes PII emails and phone numbers.
        - Ignores standalone greetings/fillers.
        - Discards non-user bot turns.
        """
        raw_queries = [
            "Hi there!",                                                        # standalone filler -> dropped
            "Agent: Hello customer support here",                              # agent message -> dropped
            "I want a refund for my order.",                                   # kept (>= 5 words)
            "My debit card payment failed at check-out, please support me.",    # kept (>= 5 words)
            "Help reset my password! email me at amey@example.com right now.",  # PII email stripped, kept (>= 5 words)
            "hello",                                                            # filler -> dropped
            "call +1-555-0199 for shipping status"                             # PII phone stripped, less than 5 words -> dropped
        ]
        
        cleaned = preprocess_conversations(raw_queries)
        
        # Verify sizes and strings
        self.assertEqual(len(cleaned), 3)
        self.assertIn("i want a refund for my order.", cleaned)
        self.assertIn("my debit card payment failed at check-out, please support me.", cleaned)
        # Check that email was scrubbed and rest kept
        self.assertIn("help reset my password! email me at right now.", cleaned)

    def test_embeddings_dimensions(self):
        """
        Validates that Sentence-Transformer is preloaded successfully
        and outputs dense numerical tensors with 384 components.
        """
        load_embedding_model()
        test_phrases = [
            "how do i cancel my subscription and get a refund?",
            "i want to track my order number and check shipping progress.",
            "my package arrived but it is damaged and broken."
        ]
        
        vectors = embed_messages(test_phrases)
        self.assertEqual(vectors.shape, (3, 384))
        self.assertTrue(isinstance(vectors, np.ndarray))

    def test_clustering_and_fallbacks(self):
        """
        Verifies that UMAP reduces vectors and HDBSCAN segments clusters.
        We feed a larger replicated list to meet HDBSCAN's density metrics.
        """
        load_embedding_model()
        base_phrases = [
            "how do i cancel my subscription and get a refund?",
            "i want to track my order number and check shipping progress.",
            "my package arrived but it is damaged and broken.",
            "how do i apply a promo code to my existing order?",
            "the mobile app keeps crashing whenever i try to log in.",
            "the payment failed at checkout, can i try another card?",
            "i want to delete my account and erase my personal data.",
            "i got double charged for my monthly subscription fees.",
            "is there a phone number i can call for live customer support?",
            "i need to change my delivery address for my recent order."
        ]
        
        # Duplicate to meet minimum clustering requirement of 10
        phrases = base_phrases * 2
        vectors = embed_messages(phrases)
        
        clusters = cluster_messages(vectors, phrases)
        self.assertTrue(isinstance(clusters, dict))
        
        # Check that all indices were categorized (or marked noise -1)
        total_in_clusters = sum(len(msgs) for msgs in clusters.values())
        self.assertEqual(total_in_clusters, len(phrases))

    def test_sentiment_classification_thresholds(self):
        """
        Verifies DistilBERT classification accuracy and cluster aggregate scoring.
        - >60% positive -> "positive"
        - >60% negative -> "negative"
        - else -> "mixed"
        """
        load_sentiment_pipeline()
        
        positive_cluster = [
            "This app is absolutely amazing, thank you so much for the brilliant features!",
            "I love the service, it works beautifully and fast.",
            "Great job team, this checkout was incredibly easy and helpful."
        ]
        
        negative_cluster = [
            "This is horrible, the system keeps failing and crashing.",
            "I want a refund immediately, your support is terrible.",
            "Extremely disappointed with this broken item and delayed delivery."
        ]
        
        mixed_cluster = [
            "I love the service, it works beautifully and fast.", # positive
            "This is horrible, the system keeps failing and crashing." # negative
        ]
        
        pos_res = analyze_cluster_sentiment(positive_cluster)
        neg_res = analyze_cluster_sentiment(negative_cluster)
        mixed_res = analyze_cluster_sentiment(mixed_cluster)
        
        self.assertEqual(pos_res["overall"], "positive")
        self.assertEqual(neg_res["overall"], "negative")
        self.assertEqual(mixed_res["overall"], "mixed")

if __name__ == "__main__":
    unittest.main()
