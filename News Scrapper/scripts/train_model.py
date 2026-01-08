"""
Training script for news article classifier.

This script:
1. Loads processed articles from database
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
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DEFAULT_CLASSIFIER,
    TEST_SIZE,
    RANDOM_STATE,
    MIN_CONFIDENCE_THRESHOLD,
    TOPIC_CATEGORIES,
    MODELS_DIR,
    DATA_DIR
)
from database.connection import DatabaseConnection
from database.models import ProcessedArticle, SourceArticle
from src.features.tfidf_vectorizer import TFIDFFeatureExtractor, extract_top_features
from src.models.classifier import (
    NewsArticleClassifier,
    train_classifier,
    compare_classifiers
)
from src.models.evaluator import ModelEvaluator
db=DatabaseConnection()

def setup_logging(verbose: bool = False):
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
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_training_data(
    min_samples_per_class: int = 12,
    max_samples: Optional[int] = None,
    balance_classes: bool = False
) -> Tuple[List[str], List[str], List[int]]:
    """
    Load processed articles from database for training.
    
    Args:
        min_samples_per_class: Minimum samples required per class
        max_samples: Maximum total samples to load (None = all)
        balance_classes: Whether to balance class distribution
        
    Returns:
        Tuple of (documents, labels, article_ids)
    """
    logger.info("Loading training data from database...")
    
    with db.get_db() as session:
        # Query processed articles with their raw article info
        query = session.query(
            ProcessedArticle.id,
            ProcessedArticle.processed_text,
            SourceArticle.source,
            SourceArticle.url,
            ProcessedArticle.category
        ).join(
            SourceArticle,
            ProcessedArticle.source_article_id == SourceArticle.id
        ).filter(
            ProcessedArticle.is_duplicate == False,  # Exclude duplicates
            ProcessedArticle.token_count > 50  # Minimum token threshold
        )
        
        # Fetch all
        results = query.all()
    
    if not results:
        raise ValueError("No processed articles found in database")
    
    logger.info(f"Found {len(results)} processed articles")
    
    # Extract data
    article_ids = []
    documents = []
    labels = []
    
   
    
    for article_id, text, source, url,category in results:
        # Infer category from URL or source
        
        
        if category:
            article_ids.append(article_id)
            documents.append(text)
            labels.append(category.split("|")[0])
    
    logger.info(f"Mapped {len(documents)} articles to categories")
    
    # Filter classes with insufficient samples
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("Class distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count}")
    
    # Remove underrepresented classes
    valid_labels = {label for label, count in label_counts.items() 
                    if count >= min_samples_per_class}
    
    if not valid_labels:
        raise ValueError(
            f"No classes have at least {min_samples_per_class} samples"
        )
    
    filtered_data = [
        (aid, doc, label) 
        for aid, doc, label in zip(article_ids, documents, labels)
        if label in valid_labels
    ]
    
    article_ids, documents, labels = zip(*filtered_data)
    article_ids = list(article_ids)
    documents = list(documents)
    labels = list(labels)
    
    logger.info(f"After filtering: {len(documents)} articles in {len(valid_labels)} classes")
    
    # Balance classes if requested
    if balance_classes:
        article_ids, documents, labels = _balance_classes(
            article_ids, documents, labels
        )
        logger.info(f"After balancing: {len(documents)} articles")
    
    # Limit total samples if specified
    if max_samples and len(documents) > max_samples:
        indices = np.random.choice(len(documents), max_samples, replace=False)
        article_ids = [article_ids[i] for i in indices]
        documents = [documents[i] for i in indices]
        labels = [labels[i] for i in indices]
        logger.info(f"Limited to {max_samples} samples")
    
    return documents, labels, article_ids


def _map_sources_to_categories() -> Dict[str, str]:
    """
    Map news sources to topic categories.
    
    Returns:
        Dictionary mapping source domains to categories
    """
    return {
        'techcrunch.com': 'technology',
        'theverge.com': 'technology',
        'arstechnica.com': 'technology',
        'wired.com': 'technology',
        
        'reuters.com/markets': 'business',
        'bloomberg.com': 'business',
        'wsj.com': 'business',
        'ft.com': 'business',
        
        'nature.com': 'science',
        'sciencedaily.com': 'science',
        'newscientist.com': 'science',
        
        'espn.com': 'sports',
        'bleacherreport.com': 'sports',
        
        'politico.com': 'politics',
        'thehill.com': 'politics',
        
        'bbc.com/news/world': 'world',
        'theguardian.com/world': 'world',
        'aljazeera.com': 'world',
    }


def _infer_category(url: str, source: str, mapping: Dict[str, str]) -> Optional[str]:
    """
    Infer article category from URL and source.
    
    Args:
        url: Article URL
        source: Source domain
        mapping: Source to category mapping
        
    Returns:
        Inferred category or None
    """
    url_lower = url.lower()
    
    # Check URL patterns
    for pattern, category in [
        ('/technology/', 'technology'),
        ('/tech/', 'technology'),
        ('/science/', 'science'),
        ('/business/', 'business'),
        ('/sports/', 'sports'),
        ('/politics/', 'politics'),
        ('/world/', 'world'),
        ('/health/', 'health'),
        ('/entertainment/', 'entertainment'),
    ]:
        if pattern in url_lower:
            return category
    
    # Check source mapping
    for source_pattern, category in mapping.items():
        if source_pattern in source.lower() or source_pattern in url_lower:
            return category
    
    return None


def _balance_classes(
    article_ids: List[int],
    documents: List[str],
    labels: List[str]
) -> Tuple[List[int], List[str], List[str]]:
    """
    Balance class distribution by undersampling majority classes.
    
    Args:
        article_ids: List of article IDs
        documents: List of documents
        labels: List of labels
        
    Returns:
        Balanced (article_ids, documents, labels)
    """
    from collections import defaultdict
    
    # Group by label
    grouped = defaultdict(list)
    for aid, doc, label in zip(article_ids, documents, labels):
        grouped[label].append((aid, doc, label))
    
    # Find minimum class size
    min_size = min(len(samples) for samples in grouped.values())
    
    logger.info(f"Balancing classes to {min_size} samples each")
    
    # Sample from each class
    balanced = []
    for label, samples in grouped.items():
        sampled = np.random.choice(len(samples), min_size, replace=False)
        balanced.extend([samples[i] for i in sampled])
    
    # Shuffle
    np.random.shuffle(balanced)
    
    # Unpack
    article_ids, documents, labels = zip(*balanced)
    return list(article_ids), list(documents), list(labels)


def train_single_model(
    X_train, y_train,
    X_test, y_test,
    classifier_type: str,
    tune_hyperparams: bool = False,
    save_model: bool = True
) -> Tuple[NewsArticleClassifier, Dict]:
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


def train_and_compare_all(
    X_train, y_train,
    X_test, y_test
) -> Dict[str, Dict]:
    """
    Train and compare all available classifiers.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of results for all classifiers
    """
    logger.info("Training and comparing all classifiers...")
    
    # Compare all models
    results = compare_classifiers(
        X_train, y_train,
        X_test, y_test
    )
    
    # Generate comparison plot
    evaluator = ModelEvaluator()
    evaluator.compare_models(results, save=True)
    logger.info("Model comparison plot saved")
    
    # Save best model
    best_model = None
    best_f1 = 0.0
    best_name = None
    
    for name, result in results.items():
        if 'metrics' in result:
            f1 = result['metrics']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_model = result['classifier']
                best_name = name
    
    if best_model:
        logger.success(f"\nBest model: {best_name} (F1: {best_f1:.4f})")
        
        # Save as default model
        try:
            best_model.save()
            logger.success(f"Best model saved as default")
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
    
    return results


def save_training_metadata(
    vectorizer: TFIDFFeatureExtractor,
    classifier: NewsArticleClassifier,
    results: Dict,
    train_size: int,
    test_size: int
):
    """
    Save training metadata for reproducibility.
    
    Args:
        vectorizer: Fitted vectorizer
        classifier: Trained classifier
        results: Evaluation results
        train_size: Number of training samples
        test_size: Number of test samples
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'classifier_type': classifier.classifier_type,
        'model_version': classifier.config.get('model_version', 'unknown'),
        'training_samples': train_size,
        'test_samples': test_size,
        'num_classes': classifier.num_classes,
        'class_names': classifier.class_names,
        'vocabulary_size': vectorizer.vocabulary_size,
        'tfidf_config': {
            'max_features': vectorizer.max_features,
            'min_df': vectorizer.min_df,
            'max_df': vectorizer.max_df,
            'ngram_range': vectorizer.ngram_range,
        },
        'performance': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        },
        'random_state': RANDOM_STATE
    }
    
    # Save metadata
    metadata_file = MODELS_DIR / 'training_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.success(f"Training metadata saved to {metadata_file}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train news article classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train default classifier (Logistic Regression)
  python scripts/train_model.py
  
  # Train specific classifier with hyperparameter tuning
  python scripts/train_model.py --classifier naive_bayes --tune-hyperparams
  
  # Compare all classifiers
  python scripts/train_model.py --compare-all
  
  # Train with custom test split
  python scripts/train_model.py --test-size 0.25 --min-samples 50
        """
    )
    
    parser.add_argument(
        '--classifier',
        type=str,
        default=DEFAULT_CLASSIFIER,
        choices=['logistic_regression', 'naive_bayes', 'svm', 'random_forest'],
        help='Classifier type to train'
    )
    
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Train and compare all classifiers'
    )
    
    parser.add_argument(
        '--tune-hyperparams',
        action='store_true',
        help='Perform hyperparameter tuning (slower)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=TEST_SIZE,
        help='Test set proportion (default: 0.2)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum samples per class (default: 10)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum total samples to use (default: all)'
    )
    
    parser.add_argument(
        '--balance-classes',
        action='store_true',
        help='Balance class distribution by undersampling'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logger.info("="*70)
    logger.info("NEWS ARTICLE CLASSIFIER - TRAINING PIPELINE")
    logger.info("="*70)
    
    try:
        # Step 1: Load data
        logger.info("\n[Step 1/5] Loading training data...")
        documents, labels, article_ids = load_training_data(
            min_samples_per_class=args.min_samples,
            max_samples=args.max_samples,
            balance_classes=args.balance_classes
        )
        
        logger.info(f"Loaded {len(documents)} articles")
        logger.info(f"Classes: {sorted(set(labels))}")
        
        # Step 2: Split data
        logger.info("\n[Step 2/5] Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            documents, labels,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            stratify=labels
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Step 3: Extract features
        logger.info("\n[Step 3/5] Extracting TF-IDF features...")
        vectorizer = TFIDFFeatureExtractor()
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        logger.info(f"Feature matrix shape: {X_train_tfidf.shape}")
        logger.info(f"Vocabulary size: {vectorizer.vocabulary_size}")
        
        # Save vectorizer
        vectorizer.save()
        logger.success("Vectorizer saved")
        
        # Analyze top features per class
        logger.info("\nTop features per class:")
        class_features = extract_top_features(X_train, y_train, top_n=5)
        for label, features in sorted(class_features.items()):
            top_terms = [term for term, _ in features]
            logger.info(f"  {label}: {', '.join(top_terms)}")
        
        # Step 4: Train model(s)
        logger.info("\n[Step 4/5] Training classifier(s)...")
        
        if args.compare_all:
            # Compare all classifiers
            results = train_and_compare_all(
                X_train_tfidf, y_train,
                X_test_tfidf, y_test
            )
            
            # Get best classifier for metadata
            best_result = max(
                [r for r in results.values() if 'metrics' in r],
                key=lambda x: x['metrics']['f1_score']
            )
            classifier = best_result['classifier']
            eval_results = best_result['metrics']
            
        else:
            # Train single classifier
            classifier, eval_results = train_single_model(
                X_train_tfidf, y_train,
                X_test_tfidf, y_test,
                classifier_type=args.classifier,
                tune_hyperparams=args.tune_hyperparams,
                save_model=not args.no_save
            )
        
        # Step 5: Save metadata
        if not args.no_save:
            logger.info("\n[Step 5/5] Saving training metadata...")
            save_training_metadata(
                vectorizer,
                classifier,
                eval_results,
                len(X_train),
                len(X_test)
            )
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Final Model: {classifier.classifier_type}")
        logger.info(f"Accuracy:    {eval_results['accuracy']:.4f}")
        logger.info(f"F1 Score:    {eval_results['f1_score']:.4f}")
        logger.info(f"Classes:     {classifier.num_classes}")
        logger.success("\nModel training completed successfully!")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()