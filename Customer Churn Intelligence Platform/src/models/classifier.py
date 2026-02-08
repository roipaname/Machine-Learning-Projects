from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,precision_recall_curve,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV,cross_val_score
from xgboost import XGBClassifier
import pandas as pd
from config.settings import RANDOM_STATE,CV_FOLDS,MODELS_DIR,MODEL_VERSION
from src.features.training_data import build_training_dataset
import joblib
import shap
from loguru import logger
import numpy as np
from typing import Dict,Optional,List,Tuple,Any
from datetime import datetime
from pathlib import Path

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
        self.params=self.config['params']
        if custom_params:
            self.params.update(custom_params)
        self.model=self.config['class'](**self.params)
        self.classifier_type=classifier_name
        self.is_trained=False
        self.scaler=StandardScaler()
        self.label_encoder=LabelEncoder()
        self.training_date=None
        self.training_date=None
        self.training_samples=0
        self.num_classes=0
        self.class_names=[]
        self.feature_dim=0
        self.label_encoders = {}

        logger.success(f"Initialized {classifier_name} classifier")
    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare features for ML training
        """
        # Separate features and target
        X = df.drop(['customer_id', 'churned', 'churn_date'], axis=1, errors='ignore')
        y = df['churned'].astype(int)
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y
    
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
    def predict_with_confidence(self,X:pd.DataFrame,threshold:float=0.5)->List[Tuple[str,float]]:
        """
        Predict with confidence scores.
        
        Args:
            X: FDataFrame (n_samples, n_features)
            threshold: Minimum confidence for prediction (else return 'uncertain')
            
        Returns:
            List of (predicted_label, confidence) tuples
        """
        probas=self.predict_proba(X)
        max_probas=probas.max(axis=1)
        predictions=self.predict(X)
        results=[]

        for conf,label in zip(max_probas,predictions):
            if conf>=threshold:
                results.append((label,conf))
            else:
                results.append(('uncertain',conf))
        return results
    
    def evaluate(self,X:pd.DataFrame,y:pd.Series)->Dict:
        if not self.is_trained:
            logger.error("Model is not yet trained.Call fit() before evaluate")
            raise ValueError("Model not trained")
        y_pred=self.predict(X)
        y_proba=self.predict_proba(X)[:,1] if self.num_classes==2 else None
        report=classification_report(y,y_pred,output_dict=True)
        conf_matrix=confusion_matrix(y,y_pred).tolist()
        auc_score=roc_auc_score(y,self.predict_proba(X)[:,1]) if self.num_classes==2 else None
        precision,recall,_=precision_recall_curve(y,y_proba) if self.num_classes==2 else (None,None,None)
        accuracy_scores=accuracy_score(y,y_pred)

        evaluation_results={
            'classification_report':report,
            'confusion_matrix':conf_matrix,
            'roc_auc_score':auc_score,
            'precision_recall_curve':{
                'precision':precision.tolist() if precision is not None else None,
                'recall':recall.tolist() if recall is not None else None
            },
            'accuracy_score':accuracy_scores
        }
        logger.success(
            f"Evaluation complete - Accuracy: {accuracy_scores:.4f}"
        )
        
        return evaluation_results
    def _cross_validate(self,X:np.ndarray,y:np.ndarray,cv:int=CV_FOLDS)-> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Encoded labels
            
        Returns:
            Dictionary with mean and std of CV scores
        """
        try:
            cv_score=cross_val_score(self.model,X,y,cv=cv,scoring='accuracy')
            return{
                'mean_cv_score':cv_score.mean(),
                'std_cv_score':cv_score.std(),
                'scores': cv_score.tolist()   
            }
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            raise e
        

    def hyperparameter_tuning(self,X:pd.DataFrame,y:pd.Series,cv:int=CV_FOLDS)->Dict[str,Any]:
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
        X_scaled=self.scaler.fit_transform(X)
        grid_params=self.config.get('grid_params',{})
        if not grid_params:
            logger.warning("No grid parameters defined for this classifier. Skipping tuning.")
            return {}
        
        try:
            grid_search=GridSearchCV(estimator=self.config['class'](**self.params),param_grid=grid_params,cv=cv,scoring='accuracy',verbose=1,n_jobs=-1)
            grid_search.fit(X_scaled,y_encoded)
            self.model=grid_search.best_estimator_
            self.is_trained=True
            self.training_date=datetime.utcnow()
            results={
                'best_params':grid_search.best_params_,
                'best_score':grid_search.best_score_,
                'cv_results':grid_search.cv_results_
            }
            logger.success(f"Hyperparameter tuning complete. Best Score: {results['best_score']:.4f}")
            return results
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise e
    def get_feature_importance_basic(self, top_n: int = 20) -> Dict[str, List[float]]:
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
        if not hasattr(self.model, 'coef_'):
            logger.warning(f"{self.classifier_type} does not have interpretable coefficients")
            return {}
        
        coef = self.model.coef_
        
        # Get top features per class
        importance = {}
        for idx, class_name in enumerate(self.class_names):
            class_coef = coef[idx] if len(coef.shape) > 1 else coef
            top_indices = np.argsort(np.abs(class_coef))[-top_n:][::-1]
            top_values = class_coef[top_indices]  # <-- actual coefficients
            importance[class_name] = top_values.tolist()  # store as list of floats
        
        return importance
    def get_feature_importance_shap(self,X:pd.DataFrame, top_n: int = 20) -> Dict[str, List[float]]:
        """
        Get feature importance scores using SHAP values.
        
        Args:
            X: Feature matrix (unscaled)
            top_n: Number of top features per class
            
        Returns:
            Dictionary mapping class names to SHAP importance scores
        """
        if not  self.is_trained:
            raise ValueError("Model must be trained first")
        X_scaled = self.scaler.transform(X)
        if self.classifier_type in ['random_forest', 'gradient_boosting', 'xgboost']:
           explainer = shap.TreeExplainer(self.model)
           shap_values = explainer.shap_values(X_scaled)
        else:
           explainer = shap.LinearExplainer(self.model, X_scaled)
           shap_values = explainer.shap_values(X_scaled)

        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        logger.success("SHAP analysis complete")
        return shap_values, feature_importance
    
    def calculate_feature_importance(self, X, y):
        """
        Calculate SHAP values for best model
        """
        logger.info("Calculating SHAP values...")
        
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        logger.success("SHAP analysis complete")
        return shap_values, feature_importance

    def save_model(self,path:Optional[Path]=None):
        if not self.is_trained:
            logger.error("Model is not trained.Train Model before saving")
            raise ValueError("Model not trained")
        model_path=path or MODELS_DIR/f"{self.classifier_type}_model.joblib"
        model_path.parent.mkdir(parents=True,exist_ok=True)
        try:
            model_data={
                "version":MODEL_VERSION,
                "training_date":self.training_date.isoformat(),
                "classifier_type":self.classifier_type, 
                "model_state":self.model,
                "scaler":self.scaler,
                "label_encoder":self.label_encoder,
                "feature_names":self.feature_names,
                "training_samples":self.training_samples,
                "num_classes":self.num_classes,
                "class_names":self.class_names
            }

            with open(model_path,'wb') as f:
                joblib.dump(model_data,f)
            logger.success(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e
    def train_models(self, X_train, y_train, X_val, y_val):
        for name,config in self.CLASSIFIERS.items():
            logger.info(f"Training {name}...")
            results={}
            try:
                model=config['class'](**config['params'])
                model.fit(X_train,y_train)
                y_pred=model.predict(X_val)
                y_pred_proba=model.predict_proba(X_val)[:,1] if hasattr(model,'predict_proba') else None
                roc_auc_score=roc_auc_score(y_val,y_pred_proba) if y_pred_proba is not None else None
                accuracy=accuracy_score(y_val,y_pred)
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc_score:.4f}" if roc_auc_score is not None else f"{name} - Accuracy: {accuracy:.4f}")
                results[name]={
                    'model':model,
                    'accuracy':accuracy,
                    'roc_auc_score':roc_auc_score
                }
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        best_name=max(results,key=lambda x:results[x]['roc_auc_score'] if results[x]['roc_auc_score'] is not None else results[x]['accuracy'])
        self.best_model=results[best_name]['model']
        self.models = {k: v['model'] for k, v in results.items()}
        
        logger.success(f"Best model: {best_name} (AUC: {results[best_name]['roc_auc_score']:.4f})")
        
        return results
    

        
    @classmethod
    def load_model(classifier_type:str="random_forest",path:Path=None)->'ChurnPredictor':
        model_path=path or MODELS_DIR/f"{classifier_type.tolower()}_model.joblib"

        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"No model found at {model_path}")
        try:

            with open(model_path,'rb') as f:
                model_data=joblib.load(f)
            predictor=ChurnPredictor(classifier_type=model_data['classifier_type'])
            predictor.model=model_data['model_state']
            predictor.scaler=model_data['scaler']
            predictor.label_encoder=model_data['label_encoder']
            predictor.feature_names=model_data['feature_names']
            predictor.training_samples=model_data['training_samples']
            predictor.num_classes=model_data['num_classes']
            predictor.class_names=model_data['class_names']
            predictor.training_date=datetime.fromisoformat(model_data['training_date'])
            predictor.is_trained=True
            logger.success(f"Model loaded from {model_path}")
            return predictor
        except Exception as e:
            logger.error(f"Error loading model:,please check if model exist {e}")
            raise e