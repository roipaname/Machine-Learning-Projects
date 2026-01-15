from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support
import numpy as np
from loguru import logger
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict,List,Optional

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
            self.model.fit(X,y_encoded)
            self.is_trained=True
            self.training_date=datetime.utcnow()

            logger.info(f"{self.classifier_type} training complete")
            return self
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

