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


        


