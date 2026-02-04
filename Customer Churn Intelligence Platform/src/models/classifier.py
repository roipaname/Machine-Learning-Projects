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
        self.sclaer=StandardScaler()
        self.label_encoder=LabelEncoder()
        self.training_date=None