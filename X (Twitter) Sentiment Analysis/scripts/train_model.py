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



