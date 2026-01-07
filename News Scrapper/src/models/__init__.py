"""
Machine learning models for news article classification.

This module provides classifiers, evaluation utilities, and model management
for the news topic classification task.
"""

from src.models.classifier import (
    NewsArticleClassifier,
    train_classifier,
    evaluate_classifier,
    compare_classifiers
)

from src.models.evaluator import (
    ModelEvaluator,
    generate_classification_report,
    plot_confusion_matrix,
    plot_learning_curves
)

__all__ = [
    'NewsArticleClassifier',
    'train_classifier',
    'evaluate_classifier',
    'compare_classifiers',
    'ModelEvaluator',
    'generate_classification_report',
    'plot_confusion_matrix',
    'plot_learning_curves'
]

__version__ = '1.0.0'