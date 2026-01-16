"""
Training script for sentiment classifier.

This script:
1. Loads processed tweets from database
2. Extracts TF-IDF features
3. Trains and evaluates multiple classifiers
4. Selects best model based on performance
5. Saves trained model and vectorizer
6. Generates comprehensive evaluation reports

Usage:
    python scripts/train_model.py --classifier logistic_regression --tune-hyperparams
    python scripts/train_model.py --compare-all
    python scripts/train_model.py --min-samples 100 --test-size 0.2
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MODEL_SAVE_PATH,
    VECTORIZER_SAVE_PATH,
    DEFAULT_CLASSIFIER,
    RANDOM_STATE,
    MODELS_DIR,
    DATA_DIR
)
from database.connect import DatabaseConnection
from src.features.tfidf_vectorizer import TfidfVectorizer,extract_top_features
from src.models.classifier import SentimentAnalyzer,compare_classifiers,train_classifier

from src.models.evaluator import ModelEvaluator

db=DatabaseConnection()


def setup_logging():
    """Configure logging for training script."""
    log_level = "DEBUG" if verbose else "INFO"
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File handler
    log_file = DATA_DIR / 'logs' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    
    logger.success(f"Logging initialized. Log file: {log_file}")

def load_training_data( min_samples_per_class: int = 12,
    max_samples: Optional[int] = None,
    balance_classes: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    

    try:
        results=db.find_many("processed_tweet",batch_size=500,limit=max_samples)
        documents=[]
        sentiments=[]
        tweet_ids=[]
        for doc in results:
            if doc and  doc['sentiment']:
                documents.append(doc['processed_text'])
                sentiments.append(doc['sentiment'])
                tweet_ids.append(doc['source_id'])
        return documents,sentiments,tweet_ids
    except Exception as e:
        logger.error(f"failed to load training date:{e}")
        raise



def _balance_classes(
    tweet_ids: List[int],
    documents: List[str],
    labels: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Balance class distribution by undersampling majority classes.
    
    Args:
        tweet_ids: List of tweet IDs
        documents: List of documents
        labels: List of labels
        
    Returns:
        Balanced (tweet_ids, documents, labels)
    """
    from collections import defaultdict

    grouped=defaultdict(list)

    for tid,doc,label in zip(tweet_ids,documents,labels):
        grouped[label].append(tid,doc,label)

    min_size=min([len(sample) for sample in grouped.values()])

    logger.info(f"Balancing classes to {min_size} samples each")

    # Sample from each class
    balanced = []
    for label, samples in grouped.items():
        sampled = np.random.choice(len(samples), min_size, replace=False)
        balanced.extend([samples[i] for i in sampled])
    
    # Shuffle
    np.random.shuffle(balanced)
    
    # Unpack
    tweet_ids, documents, labels = zip(*balanced)
    return list(tweet_ids), list(documents), list(labels)

def train_single_model(
    X_train, y_train,
    X_test, y_test,
    classifier_type: str,
    tune_hyperparams: bool = False,
    save_model: bool = True
) -> Tuple[SentimentAnalyzer, Dict]:
    """
    Train a single classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classifier_type: Type of classifier
        tune_hyperparams: Whether to tune hyperparameters
        save_model: Whether to save trained model
        
    Returns:
        Tuple of (trained classifier, evaluation results)
    """
    logger.info(f"{'='*70}")
    logger.info(f"Training {classifier_type}")
    logger.info(f"{'='*70}")
    
    # Train
    classifier, results = train_classifier(
        X_train, y_train,
        X_test, y_test,
        classifier_type=classifier_type,
        tune_hyperparams=tune_hyperparams
    )
    
    # Log results
    logger.info(f"\nResults for {classifier_type}:")
    logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1 Score:  {results['f1_score']:.4f}")
    
    # Generate visualizations
    evaluator = ModelEvaluator()
    
    # Predictions for visualization
    y_pred = classifier.predict(X_test)
    
    # Confusion matrix
    try:
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            classifier.class_names,
            model_name=classifier_type,
            normalize=True
        )
        logger.info(f"  ✓ Confusion matrix saved")
    except Exception as e:
        logger.warning(f"  ✗ Could not generate confusion matrix: {e}")
    
    # Precision/Recall by class
    try:
        evaluator.plot_precision_recall_by_class(
            y_test, y_pred,
            classifier.class_names,
            model_name=classifier_type
        )
        logger.info(f"  ✓ Precision/Recall plot saved")
    except Exception as e:
        logger.warning(f"  ✗ Could not generate precision/recall plot: {e}")
    
    # ROC curves (if probabilities available)
    try:
        y_proba = classifier.predict_proba(X_test)
        evaluator.plot_roc_curves(
            y_test, y_proba,
            classifier.class_names,
            model_name=classifier_type
        )
        logger.info(f"  ✓ ROC curves saved")
    except Exception as e:
        logger.warning(f"  ✗ Could not generate ROC curves: {e}")
    
    # Full evaluation report
    try:
        y_proba = classifier.predict_proba(X_test)
        report = evaluator.generate_full_report(
            y_test, y_pred, y_proba,
            classifier.class_names,
            model_name=classifier_type,
            save=True
        )
        logger.info(f"  ✓ Evaluation report saved")
    except Exception as e:
        logger.warning(f"  ✗ Could not generate evaluation report: {e}")
    
    # Save model
    if save_model:
        try:
            model_path = MODELS_DIR / f"{classifier_type}_model.pkl"
            classifier.save(model_path)
            logger.success(f"  ✓ Model saved to {model_path}")
        except Exception as e:
            logger.error(f"  ✗ Failed to save model: {e}")
    
    return classifier, results
