import logging as logger
from database.operations import get_processed_articles
from src.features.tfidf_vectorizer import TFIDFFeatureExtractor
logger.basicConfig(level=logger.INFO)
from pathlib import Path
def main():
    logger.info("Fetching processed articles...")

    articles = get_processed_articles(limit=50_000)

    if not articles:
        raise RuntimeError("No processed articles found")

    documents = [a.processed_text for a in articles]

    logger.info(f"Loaded {len(documents)} documents")

    extractor = TFIDFFeatureExtractor(
        max_features=50_000,
        ngram_range=(1, 2)
    )

    X = extractor.fit_transform(documents)

    logger.info(f"TF-IDF shape: {X.shape}")
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    save_dir = PROJECT_ROOT / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Include filename for pickle
    save_file = save_dir / "tfidf_vectorizer.pkl"

    path = extractor.save(save_file)
    logger.info(f"Vectorizer saved at: {path}")


if __name__ == "__main__":
    main()
