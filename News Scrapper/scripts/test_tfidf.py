from src.features.tfidf_vectorizer import TFIDFFeatureExtractor
import logging

logging.basicConfig(level=logging.INFO)
sample_docs = [
    "artificial intelligence machine learning",
    "stock market investment economy",
    "football world cup tournament",
]

logging.info("Testing TF-IDF")

extractor = TFIDFFeatureExtractor(max_features=50)
X = extractor.fit_transform(sample_docs)

logging.info(f"Shape: {X.shape}")
