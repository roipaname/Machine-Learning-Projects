"""
Model evaluation utilities for news article classification.

Provides comprehensive evaluation metrics, visualizations, and analysis tools:
- Classification reports
- Confusion matrices
- Learning curves
- ROC curves
- Error analysis
- Model comparison
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

from config.settings import DATA_DIR
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    
    Provides methods for:
    - Performance metrics calculation
    - Visualization generation
    - Error analysis
    - Model comparison
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation outputs
        """
        self.output_dir = output_dir or (DATA_DIR / 'evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        logger.info(f"Evaluator initialized. Output dir: {self.output_dir}")
    
    def generate_full_report(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "Classifier",
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            class_names: List of class names
            model_name: Name of the model
            save: Whether to save report to disk
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Generating evaluation report for {model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
        
        # Compile report
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'num_samples': len(y_true)
            },
            'per_class_metrics': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': class_names
        }
        
        # Add ROC AUC if probabilities provided
        if y_proba is not None and class_names is not None:
            try:
                roc_auc = self._calculate_roc_auc(y_true, y_proba, class_names)
                report['roc_auc'] = roc_auc
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Error analysis
        report['error_analysis'] = self._analyze_errors(
            y_true, y_pred, class_names
        )
        
        # Log summary
        logger.info(f"Evaluation Summary for {model_name}:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        
        # Save report
        if save:
            self._save_report(report, model_name)
        
        return report
    
    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        class_names: List[str],
        model_name: str = "Classifier",
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix with annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of the model
            normalize: Whether to normalize values
            figsize: Figure size
            save: Whether to save plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Plotting confusion matrix for {model_name}...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save if requested
        if save:
            filename = f"confusion_matrix_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_learning_curves(
        self,
        estimator,
        X: csr_matrix,
        y: np.ndarray,
        model_name: str = "Classifier",
        cv: int = 5,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot learning curves to diagnose bias/variance.
        
        Args:
            estimator: Sklearn estimator
            X: Feature matrix
            y: Target labels (encoded)
            model_name: Name of the model
            cv: Cross-validation folds
            train_sizes: Training set sizes to evaluate
            figsize: Figure size
            save: Whether to save plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Generating learning curves for {model_name}...")
        
        # Calculate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring='f1_weighted',
            n_jobs=1,
            verbose=0
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot learning curves
        ax.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color='r'
        )
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Validation score')
        ax.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color='g'
        )
        
        # Formatting
        ax.set_title(f'Learning Curves - {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Examples', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save:
            filename = f"learning_curves_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: List[str],
        y_proba: np.ndarray,
        class_names: List[str],
        model_name: str = "Classifier",
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: List of class names
            model_name: Name of the model
            figsize: Figure size
            save: Whether to save plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Plotting ROC curves for {model_name}...")
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=class_names)
        n_classes = len(class_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr,
                tpr,
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {model_name}', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save:
            filename = f"roc_curves_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_by_class(
        self,
        y_true: List[str],
        y_pred: List[str],
        class_names: List[str],
        model_name: str = "Classifier",
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot precision and recall for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of the model
            figsize: Figure size
            save: Whether to save plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Plotting precision/recall by class for {model_name}...")
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, zero_division=0
        )
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot precision
        x_pos = np.arange(len(class_names))
        ax1.bar(x_pos, precision, color='skyblue', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision by Class', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(precision):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot recall
        ax2.bar(x_pos, recall, color='lightcoral', alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_title('Recall by Class', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(recall):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        fig.suptitle(f'{model_name} - Per-Class Metrics', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save if requested
        if save:
            filename = f"precision_recall_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision/recall plot saved to {save_path}")
        
        return fig
    
    def compare_models(
        self,
        results: Dict[str, Dict],
        figsize: Tuple[int, int] = (14, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Compare multiple models side by side.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            figsize: Figure size
            save: Whether to save plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Comparing {len(results)} models...")
        
        # Extract metrics
        model_names = list(results.keys())
        metrics_dict = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': []
        }
        
        for model_name in model_names:
            if 'metrics' in results[model_name]:
                m = results[model_name]['metrics']
                metrics_dict['Accuracy'].append(m['accuracy'])
                metrics_dict['Precision'].append(m['precision'])
                metrics_dict['Recall'].append(m['recall'])
                metrics_dict['F1 Score'].append(m['f1_score'])
            else:
                # Model failed
                for key in metrics_dict:
                    metrics_dict[key].append(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set width of bars
        x = np.arange(len(model_names))
        width = 0.2
        
        # Plot bars
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        # Formatting
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save:
            filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        return fig
    
    def _calculate_roc_auc(
        self,
        y_true: List[str],
        y_proba: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROC AUC scores for each class.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: List of class names
            
        Returns:
            Dictionary mapping class names to AUC scores
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=class_names)
        
        # Calculate AUC for each class
        roc_auc = {}
        for i, class_name in enumerate(class_names):
            try:
                auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                roc_auc[class_name] = float(auc_score)
            except ValueError:
                roc_auc[class_name] = 0.0
        
        # Calculate macro average
        roc_auc['macro_avg'] = float(np.mean(list(roc_auc.values())))
        
        return roc_auc
    
    def _analyze_errors(
        self,
        y_true: List[str],
        y_pred: List[str],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze misclassification patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Dictionary containing error analysis
        """
        # Find misclassified samples
        errors = [(i, true, pred) for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
        
        # Count errors by true class
        error_by_class = {}
        for _, true, pred in errors:
            if true not in error_by_class:
                error_by_class[true] = {'total': 0, 'misclassified_as': {}}
            error_by_class[true]['total'] += 1
            
            if pred not in error_by_class[true]['misclassified_as']:
                error_by_class[true]['misclassified_as'][pred] = 0
            error_by_class[true]['misclassified_as'][pred] += 1
        
        # Calculate error rate
        total_errors = len(errors)
        error_rate = total_errors / len(y_true) if len(y_true) > 0 else 0
        
        analysis = {
            'total_errors': total_errors,
            'error_rate': float(error_rate),
            'errors_by_class': error_by_class,
            'most_confused_pairs': self._find_confused_pairs(y_true, y_pred, top_n=5)
        }
        
        return analysis
    
    def _find_confused_pairs(
        self,
        y_true: List[str],
        y_pred: List[str],
        top_n: int = 5
    ) -> List[Tuple[str, str, int]]:
        """
        Find most commonly confused class pairs.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            top_n: Number of top pairs to return
            
        Returns:
            List of (true_class, predicted_class, count) tuples
        """
        confusion_counts = {}
        
        for true, pred in zip(y_true, y_pred):
            if true != pred:
                pair = (true, pred)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        
        # Sort by count
        sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [(true, pred, count) for (true, pred), count in sorted_pairs[:top_n]]
    
    def _save_report(self, report: Dict, model_name: str):
        """
        Save evaluation report to JSON file.
        
        Args:
            report: Evaluation report dictionary
            model_name: Name of the model
        """
        filename = f"evaluation_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_path = self.output_dir / filename
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation report saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_classification_report(
    y_true: List[str],
    y_pred: List[str],
    class_names: Optional[List[str]] = None,
    output_file: Optional[Path] = None
) -> str:
    """
    Generate and optionally save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_file: Path to save report
        
    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {output_file}")
    
    return report


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Quick confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    evaluator = ModelEvaluator()
    fig = evaluator.plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=normalize,
        save=save_path is not None
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curves(
    estimator,
    X: csr_matrix,
    y: np.ndarray,
    cv: int = 5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Quick learning curves plot.
    
    Args:
        estimator: Sklearn estimator
        X: Feature matrix
        y: Target labels
        cv: Cross-validation folds
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    evaluator = ModelEvaluator()
    fig = evaluator.plot_learning_curves(
        estimator, X, y,
        cv=cv,
        save=save_path is not None
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    """
    Example usage and testing of evaluator.
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    logger.info("Testing ModelEvaluator...")
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=4,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Class names
    class_names = [f'Class_{i}' for i in range(4)]
    y_test_labels = [class_names[i] for i in y_test]
    y_pred_labels = [class_names[i] for i in y_pred]
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate full report
    report = evaluator.generate_full_report(
        y_test_labels,
        y_pred_labels,
        y_proba,
        class_names,
        model_name="LogisticRegression"
    )
    
    logger.info(f"Overall Accuracy: {report['overall_metrics']['accuracy']:.4f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        y_test_labels,
        y_pred_labels,
        class_names,
        model_name="LogisticRegression"
    )
    
    # Plot learning curves
    evaluator.plot_learning_curves(
        clf,
        X_train,
        y_train,
        model_name="LogisticRegression"
    )
    
    # Plot ROC curves
    evaluator.plot_roc_curves(
        y_test_labels,
        y_proba,
        class_names,
        model_name="LogisticRegression"
    )
    
    # Plot precision/recall by class
    evaluator.plot_precision_recall_by_class(
        y_test_labels,
        y_pred_labels,
        class_names,
        model_name="LogisticRegression"
    )
    
    logger.info("Evaluator testing complete!")
    plt.show()