from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support,confusion_matrix
import numpy as np
from loguru import logger
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict,List,Optional,Tuple,Any

from config.settings import (
    DEFAULT_CLASSIFIER,
    MODEL_SAVE_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS
)

class SentimentAnalyzer:
    """
    Wrapper for scikit-learn classifiers with project-specific functionality.
    
    Provides:
    - Multiple classifier options
    - Consistent training/prediction interface
    - Model persistence
    - Performance metrics
    - Cross-validation
    """
    CLASSIFIERS={
        "logistic_regression":{
            "class":LogisticRegression,
            "params":{
                "max_iter":1000,
                "n_jobs":1,
                "random_state":RANDOM_STATE,
                'solver': 'saga',  # Fast for large datasets
                'multi_class': 'multinomial',
            },
            'grid_params': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        },
        "naives_bayes":{
            "class":MultinomialNB,
            "params":{
               "alpha":1.0
            },
            'grid_params': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        },
        'svm': {
            'class': LinearSVC,
            'params': {
                'max_iter': 2000,
                'random_state': RANDOM_STATE,
                'dual': False  # Faster when n_samples > n_features
            },
            'grid_params': {
                'C': [0.1, 1.0, 10.0],
                'loss': ['hinge', 'squared_hinge']
            }
        },
         'random_forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'random_state': RANDOM_STATE,
                'n_jobs': 1,
                'max_depth': 50
            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [30, 50, None]
            }
        },
        'gradien_boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'random_state': RANDOM_STATE,
                'n_jobs': 1,
                'max_depth': 5,
                'learning_rate': 0.1,

            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [30, 50, None]
            }
        }
    }

    def __init__(self,classifier_type:str="gradient_boosting",custom_params:Optional[Dict]=None):
        """
        Initialize classifier.
        
        Args:
            classifier_type: Type of classifier ('logistic_regression', 'naive_bayes', etc.)
            custom_params: Optional parameters to override defaults
        """
        self.classifier_type=classifier_type

        if classifier_type not in self.CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier: {classifier_type}. "
                f"Available: {list(self.CLASSIFIERS.keys())}"
            )
        self.config=self.CLASSIFIERS[classifier_type].copy()

        if custom_params:
            self.config["params"].update(custom_params)
        classifier_class=self.config["class"]
        self.classifier=classifier_class(**self.config["params"])
        self.label_encoder=LabelEncoder()

        self.is_trained=False
        self.training_date=None
        self.training_samples=0
        self.num_classes=0
        self.class_names=[]
        self.feature_dim=0
        logger.success(f"Initialized {classifier_type} classifier")

    def fit(self,X:csr_matrix,y:List[str],validate:bool=True)->'SentimentAnalyzer':
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (string labels)
            validate: Whether to perform cross-validation
            
        Returns:
            self (for method chaining)
        """
        if X.shape[0]!=len(y):
            raise ValueError("X and y must have same number of samples")
        logger.info(f"Training {self.classifier_type} on {X.shape[0]} samples...")

        y_encoded=self.label_encoder.fit_transform(y)
        self.training_samples=X.shape[0]
        self.num_classes=len(self.label_encoder.classes_)
        self.class_names=self.label_encoder.classes_.tolist()
        self.feature_dim=X.shape[1]

        logger.info(f"  - Features: {self.feature_dim}")
        logger.info(f"  - Classes: {self.num_classes}")
        logger.info(f"  - Class distribution: {np.bincount(y_encoded)}")

        if validate:
            cv_scores=self._cross_validate(X,y_encoded)
            logger.info(f"  - CV Score: {cv_scores['mean']:.4f} ± {cv_scores['std']:.4f}")
        try:
            self.classifier.fit(X,y_encoded)
            self.is_trained=True
            self.training_date=datetime.utcnow()

            logger.info(f"{self.classifier_type} training complete")
            return self
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    def predict(self,X:csr_matrix)->List[str]:
        """
        Prediction.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            self (for method chaining)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if X.shape[1]!=self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.feature_dim}, got {X.shape[1]}"
            )
        try:
             y_pred= self.classifier.predict(X)
             y_pred_word=self.label_encoder.inverse_transform(y_pred)
             return y_pred_word
        except Exception as e:
            logger.error(f"failed to predict : {e}")
            raise
    def predict_proba(self,X:csr_matrix)->np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if not hasattr(self.classifier,"predict_proba"):
            if hasattr(self.classifier,"decision_function"):
                decision=self.classifier.decision_function(X)
                # Convert to pseudo-probabilities using softmax
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                probas = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                return probas
            else:
                raise AttributeError(
                    f"{self.classifier_type} does not support probability prediction"
                )
        return self.classifier.predict_proba(X)
    def predict_with_confidence(
        self,
        X: csr_matrix,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Predict with confidence scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            threshold: Minimum confidence for prediction (else return 'uncertain')
            
        Returns:
            List of (predicted_label, confidence) tuples
        """

        probas=self.predict_proba(X)
        max_probas=probas.max(axis=1)
        predictions=self.predict(X)

        results=[]
        for pred,conf in zip(predictions,probas):
            if conf>=threshold:
                results.append((pred,float(conf)))
            else:
                results.append(("uncertain",float(conf)))

        return results
    
    def evaluate(self,X:csr_matrix,y_true:List[str])->Dict[str,Any]:
        """
        Evaluate model performance on test set.
        
        Args:
            X: Feature matrix
            y_true: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        y_pred=self.predict(X)
        y_true_encoded=self.label_encoder.transform(y_true)
        y_pred_encoded=self.label_encoder.transform(y_pred)

        accuracy=accuracy_score(y_true_encoded,y_pred_encoded)

        precision,recall,f1,support=precision_recall_fscore_support(y_true_encoded,y_pred_encoded,average='weighted')
        # Per-class metrics
        class_report = classification_report(
            y_true_encoded,
            y_pred_encoded,
            target_names=self.class_names,
            output_dict=True
        )
        conf_matrix=confusion_matrix(y_true_encoded,y_pred_encoded)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': self.class_names,
            'num_samples': len(y_true)
        }
        
        logger.success(
            f"Evaluation complete - Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
        )
        
        return results
    
    def _cross_validate( self,
        X: csr_matrix,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Encoded labels
            
        Returns:
            Dictionary with mean and std of CV scores
        """

        logger.debug(f"Running {CV_FOLDS}-fold cross-validation...")
        try:
            scores=cross_val_score(self.classifier,X,y,cv=CV_FOLDS,scoring='f1_weighted',n_jobs=1)
            return {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'scores': scores.tolist()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {'mean': 0.0, 'std': 0.0, 'scores': []}
        
    def hyperparameter_tuning(
        self,
        X: csr_matrix,
        y: List[str],
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and scores
        """
        logger.info(f"Starting hyperparameter tuning for {self.classifier_type}...")
        y_encoded=self.label_encoder.fit_transform(y)
        param_grid=self.config['grid_params']

        grid_search=GridSearchCV(
            estimator=self.config['class'](**self.config['params']),
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1

        )
        grid_search.fit(X,y_encoded)
        self.classifier=grid_search.best_estimator_
        self.is_trained=True

        results = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(
            f"Best parameters: {grid_search.best_params_} "
            f"(score: {grid_search.best_score_:.4f})"
        )
        return results
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[float]]:
        """
        Get feature importance scores (model-specific).
        
        Args:
            top_n: Number of top features per class
            
        Returns:
            Dictionary mapping class names to feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Only certain models have interpretable coefficients
        if not hasattr(self.classifier, 'coef_'):
            logger.warning(f"{self.classifier_type} does not have interpretable coefficients")
            return {}
        
        coef = self.classifier.coef_
        
        # Get top features per class
        importance = {}
        for idx, class_name in enumerate(self.class_names):
            class_coef = coef[idx] if len(coef.shape) > 1 else coef
            top_indices = np.argsort(np.abs(class_coef))[-top_n:][::-1]
            importance[class_name] = top_indices.tolist()
        
        return importance
    def save(self,filepath:Optional[Path]=None)->Path:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (uses default if None)
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        save_path=filepath or MODEL_SAVE_PATH
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save model and metadata
            model_data = {
                'model': self.classifier,
                'label_encoder': self.label_encoder,
                'classifier_type': self.classifier_type,
                'config': self.config,
                'metadata': {
                    'training_date': self.training_date.isoformat(),
                    'training_samples': self.training_samples,
                    'num_classes': self.num_classes,
                    'class_names': self.class_names,
                    'feature_dim': self.feature_dim,
                    'model_version': MODEL_VERSION
                }
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Model saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'SentimentAnalyzer':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model (uses default if None)
            
        Returns:
            Loaded NewsArticleClassifier instance
        """
        load_path = filepath or MODEL_SAVE_PATH
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Reconstruct classifier
            classifier = cls(classifier_type=model_data['classifier_type'])
            classifier.classifier = model_data['model']
            classifier.label_encoder = model_data['label_encoder']
            classifier.config = model_data['config']
            
            # Restore metadata
            metadata = model_data['metadata']
            classifier.is_trained = True
            classifier.training_date = datetime.fromisoformat(metadata['training_date'])
            classifier.training_samples = metadata['training_samples']
            classifier.num_classes = metadata['num_classes']
            classifier.class_names = metadata['class_names']
            classifier.feature_dim = metadata['feature_dim']
            
            logger.sucess(
                f"Model loaded from {load_path}. "
                f"Trained on {metadata['training_date'][:10]}"
            )
            
            return classifier
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of classifier."""
        status = "trained" if self.is_trained else "not trained"
        return (
            f"NewsArticleClassifier("
            f"type={self.classifier_type}, "
            f"status={status}, "
            f"classes={self.num_classes})"
        )
    
def train_classifier(
    X_train: csr_matrix,
    y_train: List[str],
    X_test: csr_matrix,
    y_test: List[str],
    classifier_type: str = DEFAULT_CLASSIFIER,
    tune_hyperparams: bool = False
) -> Tuple[SentimentAnalyzer, Dict]:
    """
    Train a classifier and evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classifier_type: Type of classifier to use
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Tuple of (trained classifier, evaluation results)
    """
    logger.info(f"Training {classifier_type} classifier...")
    
    # Initialize classifier
    classifier = SentimentAnalyzer(classifier_type=classifier_type)
    
    # Hyperparameter tuning if requested
    if tune_hyperparams:
        tuning_results = classifier.hyperparameter_tuning(X_train, y_train)
        logger.info(f"Best parameters: {tuning_results['best_params']}")
    else:
        # Standard training
        classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    results = classifier.evaluate(X_test, y_test)
    
    return classifier, results


def evaluate_classifier(
    classifier: SentimentAnalyzer,
    X_test: csr_matrix,
    y_test: List[str]
) -> Dict:
    """
    Evaluate a trained classifier.
    
    Args:
        classifier: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    return classifier.evaluate(X_test, y_test)


def compare_classifiers(
    X_train: csr_matrix,
    y_train: List[str],
    X_test: csr_matrix,
    y_test: List[str],
    classifiers: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Train and compare multiple classifiers.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classifiers: List of classifier types to compare (None = all)
        
    Returns:
        Dictionary mapping classifier names to evaluation results
    """
    if classifiers is None:
        classifiers = list(SentimentAnalyzer.CLASSIFIERS.keys())
    
    logger.info(f"Comparing {len(classifiers)} classifiers...")
    
    results = {}
    
    for clf_type in classifiers:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {clf_type}")
            logger.info(f"{'='*60}")
            
            clf, eval_results = train_classifier(
                X_train, y_train,
                X_test, y_test,
                classifier_type=clf_type
            )
            
            results[clf_type] = {
                'classifier': clf,
                'metrics': eval_results
            }
            
            logger.info(
                f"{clf_type} - Accuracy: {eval_results['accuracy']:.4f}, "
                f"F1: {eval_results['f1_score']:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to train {clf_type}: {e}")
            results[clf_type] = {'error': str(e)}
    
    # Print comparison summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    
    for clf_type, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            logger.info(
                f"{clf_type:20s} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1_score']:.4f} | "
                f"Prec: {metrics['precision']:.4f} | "
                f"Rec: {metrics['recall']:.4f}"
            )
    
    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    """
    Example usage and testing of classifier.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from src.features.tfidf_vectorizer import TfidfVectorizer
    
    logger.info("Loading sample data...")
    
    # Load subset of 20newsgroups dataset
    categories = ['sci.med', 'sci.space', 'rec.sport.baseball', 'comp.graphics']
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    
    X = newsgroups.data
    y = [categories[i] for i in newsgroups.target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Extract features
    logger.info("Extracting TF-IDF features...")
    extractor = TfidfVectorizer(max_features=1000)
    X_train_tfidf = extractor.fit_transform(X_train)
    X_test_tfidf = extractor.transform(X_test)
    
    # Train single classifier
    logger.info("\nTraining single classifier...")
    clf, results = train_classifier(
        X_train_tfidf, y_train,
        X_test_tfidf, y_test,
        classifier_type='logistic_regression'
    )
    
    logger.info(f"\nAccuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['f1_score']:.4f}")
    
    # Compare classifiers
    logger.info("\n\nComparing multiple classifiers...")
    comparison = compare_classifiers(
        X_train_tfidf, y_train,
        X_test_tfidf, y_test,
        classifiers=['logistic_regression', 'naive_bayes']
    )
    
    logger.info("Testing complete!")

     



        


