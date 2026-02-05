from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,precision_recall_curve
from sklearn.preprocessing import StandardScaler,LabelEncoder
from xgboost import XGBClassifier
import pandas as pd
from config.settings import RANDOM_STATE
from src.features.training_data import build_training_dataset
import joblib
import shap
from loguru import logger
import numpy as np
from typing import Dict,Optional
from datetime import datetime

class ChurnPredictor:
    CLASSIFIERS={
        'random_forest':{
            'class':RandomForestClassifier,
            'params':{
                'n_estimators':100,
                'max_depth':40,
                'random_state':RANDOM_STATE
            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [30, 50, None]
            }
        },
        'logistic_regression':{
            'class':LogisticRegression,
            'params':{
                'solver':'liblinear',
                'random_state':RANDOM_STATE
            },
            'grid_params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        },
        'gradient_boosting':{
            'class':GradientBoostingClassifier,
            'params':{
                'n_estimators':100,
                'learning_rate':0.1,
                'max_depth':30,
                'random_state':RANDOM_STATE
            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'xgboost':{
            'class':XGBClassifier,
            'params':{
                'n_estimators':100,
                'learning_rate':0.1,
                'max_depth':30,
                'random_state':RANDOM_STATE,
                'use_label_encoder':False,
                'eval_metric':'logloss'
            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }

    def __init__(self,classifier_name:str='random_forest',custom_params:Optional[Dict]=None):
        if classifier_name not in self.CLASSIFIERS:
            logger.error(f"Classifier {classifier_name} not recognized. Availble:{list(self.CLASSIFIERS.keys())}")
            raise ValueError("Invalid classifier name")
        self.config=self.CLASSIFIERS[classifier_name]
        self.params=self.config['paramss']
        if custom_params:
            self.params.update(custom_params)
        self.model=self.config['class'](**self.params)
        self.is_trained=False
        self.scaler=StandardScaler()
        self.label_encoder=LabelEncoder()
        self.training_date=None
        self.training_date=None
        self.training_samples=0
        self.num_classes=0
        self.class_names=[]
        self.feature_dim=0
        logger.success(f"Initialized {classifier_name} classifier")
    def fit(self,X:pd.DataFrame,y:pd.Series,validate:bool=False):
        if X.shape[0]!=len(y):
            raise ValueError("X and y must have same number of samples")

        X_Scaled=self.scaler.fit_transform(X)
        y_encoded=self.label_encoder.fit_transform(y)
        if validate:
            cv_scores=self._cross_validate(X_Scaled,y_encoded)
            logger.info(f"Cross-validation scores: {cv_scores}")

        try:
            self.model.fit(X_Scaled,y_encoded)
            self.is_trained=True
            self.training_samples=len(X)
            self.num_classes=len(np.unique(y_encoded))
            self.class_names=self.label_encoder.classes_.tolist()
            self.feature_dim=X.shape[1]
            self.training_date=datetime.utcnow()
            logger.info(f"  - Features: {self.feature_dim}")
            logger.info(f"  - Classes: {self.num_classes}")
            logger.info(f"  - Class distribution: {np.bincount(y_encoded)}")
            logger.success(f"Model trained on {self.training_samples} samples with {self.feature_dim} features")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e
        
    def predict(self,X:pd.DataFrame)->np.ndarray:
        if not self.is_trained:
            logger.error("Model is not yet trained.Call fit() before predict")
            raise ValueError("Model not trained")
        if X.shape[1]!=self.feature_dim:
            logger.error(f"Feature dimension mismatch. Expected {self.feature_dim} but got {X.shape[1]}")
            raise ValueError("Feature dimension mismatch")
        X_scaled=self.scaler.transform(X)
        try:
            predictions=self.model.predict(X_scaled)
            return self.label_encoder.inverse_transform(predictions)
        except Exception as e:
            logger.error(f"Error during prediction {e}")
            raise e
    def predict_proba(self,X:pd.DataFrame)->np.ndarray:
        if not self.is_trained:
            logger.error("Model is not yet trained.Call fit() before predict_proba")
            raise ValueError("Model not Trained")
        if X.shape[1]!=self.feature_dim:
            logger.error(f"Feature dimension mismatch. Expected {self.feature_dim} but got {X.shape[1]}")
            raise ValueError("Feature dimension mismatch")
        X_scaled=self.scaler.transform(X)
        try:
            if not hasattr(self.model,'predict_proba'):
                if hasattr(self.model,"decision_function"):
                    decision_scores=self.model.decision_function(X_scaled)
                    exp_scores=np.exp(decision_scores- np.max(decision_scores,axis=1,keepdims=True))
                    probas=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
                    return probas
                else:
                    logger.error("Model does not support probability predictions")
                    raise ValueError("No probability prediction method")
            probas=self.model.predict_proba(X_scaled)
            return probas
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise e
        

    
