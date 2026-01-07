"""
Evaluate a saved model on new data.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model-path data/models/logistic_regression_model.pkl
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH
from src.models.classifier import NewsArticleClassifier
from src.features.tfidf_vectorizer import TFIDFFeatureExtractor
from src.models.evaluator import ModelEvaluator
from scripts.train_model import load_training_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved model')
    
    parser.add_argument(
        '--model-path',
        type=Path,
        default=MODEL_SAVE_PATH,
        help='Path to saved model'
    )
    
    parser.add_argument(
        '--vectorizer-path',
        type=Path,
        default=VECTORIZER_SAVE_PATH,
        help='Path to saved vectorizer'
    )
    
    args = parser.parse_args()
    
    logger.info("Loading saved model and vectorizer...")
    
    # Load model and vectorizer
    try:
        classifier = NewsArticleClassifier.load(args.model_path)
        vectorizer = TFIDFFeatureExtractor.load(args.vectorizer_path)
        logger.success("Model and vectorizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load test data
    logger.info("Loading test data...")
    documents, labels, _ = load_training_data(min_samples_per_class=10)
    
    # Transform features
    logger.info("Extracting features...")
    X_test = vectorizer.transform(documents)
    
    # Evaluate
    logger.info("Evaluating model...")
    results = classifier.evaluate(X_test, labels)
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1 Score:  {results['f1_score']:.4f}")
    
    # Generate visualizations
    evaluator = ModelEvaluator()
    y_pred = classifier.predict(X_test)
    
    evaluator.plot_confusion_matrix(
        labels, y_pred,
        classifier.class_names,
        model_name=f"{classifier.classifier_type}_evaluation"
    )
    
    plt.show()
    logger.success("Evaluation complete!")


if __name__ == '__main__':
    main()