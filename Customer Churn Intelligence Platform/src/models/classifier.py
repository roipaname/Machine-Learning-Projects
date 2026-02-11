from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, accuracy_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import joblib
from loguru import logger
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from pathlib import Path
from config.settings import RANDOM_STATE, CV_FOLDS, MODELS_DIR, MODEL_VERSION

class ChurnPredictor:
    CLASSIFIERS = {
        'random_forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 40,
                'class_weight': 'balanced',  # Handle class imbalance
                'random_state': RANDOM_STATE
            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [30, 50, None]
            }
        },
        'logistic_regression': {
            'class': LogisticRegression,
            'params': {
                'solver': 'liblinear',
                'class_weight': 'balanced',  # Handle class imbalance
                'random_state': RANDOM_STATE,
                'max_iter': 1000
            },
            'grid_params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': RANDOM_STATE
            },
            'grid_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }

    def __init__(self, classifier_name: str = 'random_forest', custom_params: Optional[Dict] = None):
        if classifier_name not in self.CLASSIFIERS:
            raise ValueError(f"Invalid classifier. Choose from: {list(self.CLASSIFIERS.keys())}")
        
        self.classifier_type = classifier_name
        self.config = self.CLASSIFIERS[classifier_name]
        
        # Merge params
        self.params = self.config['params'].copy()
        if custom_params:
            self.params.update(custom_params)
        
        # Initialize model
        self.model = self.config['class'](**self.params)
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}  # For categorical features only
        self.feature_names = []
        
        # Training metadata
        self.is_trained = False
        self.training_date = None
        self.training_samples = 0
        self.feature_dim = 0
        
        logger.success(f"Initialized {classifier_name} classifier")

    def prepare_data(self, df: pd.DataFrame, fit_encoders: bool = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features from raw DataFrame.
        
        Args:
            df: Raw DataFrame with customer data
            fit_encoders: 
                - True: Always fit new encoders (training mode)
                - False: Always use existing encoders (inference mode, will error if none exist)
                - None (default): Auto-detect - fit if no encoders exist, use existing otherwise
        
        Returns:
            X_scaled: Scaled feature matrix (DataFrame)
            y: Binary target (0/1) or None if 'churned' not in df
        """
        # Separate features and target
        X = df.drop(['customer_id', 'churned', 'churn_date'], axis=1, errors='ignore')
        y = df['churned'].astype(int) if 'churned' in df.columns else None
        
        # Store feature names on first call
        if not self.feature_names:
            self.feature_names = X.columns.tolist()
        
        # Auto-detect mode if not specified
        if fit_encoders is None:
            fit_encoders = len(self.label_encoders) == 0
            logger.info(f"Auto-detect mode: fit_encoders={fit_encoders}")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if fit_encoders:
            # TRAINING MODE: Fit new encoders
            logger.info("Fitting new encoders and scaler...")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"  Encoded '{col}': {list(le.classes_)}")
        else:
            # INFERENCE MODE: Use existing encoders
            logger.info("Using existing encoders and scaler...")
            for col in categorical_cols:
                if col not in self.label_encoders:
                    raise ValueError(
                        f"No encoder found for column '{col}'. "
                        f"Available encoders: {list(self.label_encoders.keys())}. "
                        f"Set fit_encoders=True to create new encoders."
                    )
                
                # Handle unseen categories gracefully
                known_classes = set(self.label_encoders[col].classes_)
                unknown_mask = ~X[col].astype(str).isin(known_classes)
                
                if unknown_mask.any():
                    unknown_values = X.loc[unknown_mask, col].unique()
                    logger.warning(
                        f"Column '{col}' has {unknown_mask.sum()} unseen values: {unknown_values[:5]}. "
                        f"Mapping to 'UNKNOWN'."
                    )
                    
                    # Replace unseen with UNKNOWN
                    X.loc[unknown_mask, col] = 'UNKNOWN'
                    
                    # Add UNKNOWN to encoder classes if not present
                    if 'UNKNOWN' not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(
                            self.label_encoders[col].classes_, 'UNKNOWN'
                        )
                
                # Transform using saved encoder
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        if fit_encoders:
            # TRAINING MODE: Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            self.feature_dim = X.shape[1]
            logger.info(f"Fitted scaler on {self.feature_dim} features")
        else:
            # INFERENCE MODE: Use existing scaler
            if self.feature_dim == 0:
                raise ValueError(
                    "Scaler not fitted. Set fit_encoders=True or train the model first."
                )
            X_scaled = self.scaler.transform(X)
            logger.debug(f"Transformed using existing scaler")
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        logger.success(
            f"Data prepared: {len(X_scaled)} samples, {self.feature_dim} features "
            f"(mode: {'TRAINING' if fit_encoders else 'INFERENCE'})"
        )
        return X_scaled, y

    def fit(self, X: pd.DataFrame, y: pd.Series, validate: bool = False):
        """
        Train the model.
        
        Args:
            X: Scaled feature matrix (from prepare_data)
            y: Binary target (0/1)
            validate: Whether to run cross-validation
        """
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        
        if X.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {X.shape[1]}")
        
        # Cross-validation before training
        if validate:
            cv_scores = self._cross_validate(X.values, y.values)
            logger.info(f"Cross-validation: {cv_scores['mean_cv_score']:.4f} ± {cv_scores['std_cv_score']:.4f}")
        
        # Train model
        try:
            self.model.fit(X, y)  # y is already 0/1, no encoding needed
            
            # Update metadata
            self.is_trained = True
            self.training_samples = len(X)
            self.training_date = datetime.utcnow()
            
            logger.info(f"Training complete:")
            logger.info(f"  - Samples: {self.training_samples}")
            logger.info(f"  - Features: {self.feature_dim}")
            logger.info(f"  - Class distribution: {np.bincount(y)}")
            logger.success("Model trained successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (0/1)"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if X.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {X.shape[1]}")
        
        try:
            predictions = self.model.predict(X)
            return predictions  # Returns 0/1 directly
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if X.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {X.shape[1]}")
        
        try:
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)
            elif hasattr(self.model, 'decision_function'):
                # Convert decision scores to probabilities
                decision_scores = self.model.decision_function(X)
                probas = 1 / (1 + np.exp(-decision_scores))  # Sigmoid
                probas = np.vstack([1 - probas, probas]).T
            else:
                raise ValueError("Model doesn't support probability predictions")
            
            return probas
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise


    def predict_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convenience method: Prepare and predict on new data in one call.
        Uses existing encoders/scaler (inference mode).
        
        Args:
            df: Raw DataFrame with same structure as training data
            
        Returns:
            predictions: Binary predictions (0/1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() or load_model() first.")
        
        X, _ = self.prepare_data(df, fit_encoders=False)
        return self.predict(X)
    
    def predict_proba_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convenience method: Prepare and predict probabilities on new data.
        Uses existing encoders/scaler (inference mode).
        
        Args:
            df: Raw DataFrame with same structure as training data
            
        Returns:
            probabilities: Class probabilities shape (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() or load_model() first.")
        
        X, _ = self.prepare_data(df, fit_encoders=False)
        return self.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred).tolist()
        auc = roc_auc_score(y, y_proba)
        acc = accuracy_score(y, y_pred)
        
        precision, recall, _ = precision_recall_curve(y, y_proba)
        
        results = {
            'accuracy': acc,
            'roc_auc': auc,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        }
        
        logger.success(f"Evaluation: Accuracy={acc:.4f}, AUC={auc:.4f}")
        return results

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS) -> Dict:
        """Perform cross-validation"""
        try:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
            return {
                'mean_cv_score': scores.mean(),
                'std_cv_score': scores.std(),
                'scores': scores.tolist()
            }
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            raise

    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, cv: int = CV_FOLDS) -> Dict:
        """Hyperparameter tuning with GridSearchCV"""
        logger.info(f"Starting hyperparameter tuning for {self.classifier_type}...")
        
        grid_params = self.config.get('grid_params', {})
        if not grid_params:
            logger.warning("No grid parameters defined. Skipping tuning.")
            return {}
        
        try:
            grid_search = GridSearchCV(
                estimator=self.config['class'](**self.params),
                param_grid=grid_params,
                cv=cv,
                scoring='roc_auc',  # Use AUC for churn
                verbose=1,
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best estimator
            self.model = grid_search.best_estimator_
            self.is_trained = True
            self.training_date = datetime.utcnow()
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.success(f"Tuning complete. Best AUC: {results['best_score']:.4f}")
            logger.info(f"Best params: {results['best_params']}")
            
            return results
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            raise

    def calculate_feature_importance(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        method: str = 'auto'
    ) -> pd.DataFrame:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            X: Feature matrix (scaled)
            y: Target variable (required for permutation importance)
            method: 'auto', 'builtin', 'permutation', or 'both'
                - 'auto': Use built-in for tree models, permutation for linear
                - 'builtin': Use model's feature_importances_ or coef_
                - 'permutation': Use permutation importance (requires y)
                - 'both': Calculate both methods
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        logger.info(f"Calculating feature importance (method={method})...")
        
        results = {}
        
        # Determine which methods to use
        if method == 'auto':
            if self.classifier_type in ['random_forest', 'gradient_boosting']:
                methods = ['builtin']
            else:
                methods = ['permutation'] if y is not None else ['builtin']
        elif method == 'both':
            methods = ['builtin', 'permutation']
        else:
            methods = [method]
        
        # Built-in importance (feature_importances_ or coefficients)
        if 'builtin' in methods:
            try:
                if hasattr(self.model, 'feature_importances_'):
                    # Tree-based models
                    results['builtin_importance'] = self.model.feature_importances_
                    logger.info("Using tree-based feature_importances_")
                    
                elif hasattr(self.model, 'coef_'):
                    # Linear models
                    coef = self.model.coef_
                    if len(coef.shape) > 1:
                        coef = coef[0]  # Binary classification
                    results['builtin_importance'] = np.abs(coef)
                    logger.info("Using linear model coefficients")
                    
                else:
                    logger.warning("Model doesn't have built-in importance scores")
                    
            except Exception as e:
                logger.warning(f"Built-in importance failed: {e}")
        
        # Permutation importance
        if 'permutation' in methods:
            if y is None:
                logger.warning("Permutation importance requires y target. Skipping.")
            else:
                try:
                    perm_importance = permutation_importance(
                        self.model, 
                        X, 
                        y, 
                        n_repeats=10,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                    results['permutation_importance'] = perm_importance.importances_mean
                    results['permutation_std'] = perm_importance.importances_std
                    logger.info("Calculated permutation importance")
                    
                except Exception as e:
                    logger.warning(f"Permutation importance failed: {e}")
        
        # Create DataFrame
        importance_df = pd.DataFrame({'feature': self.feature_names})
        
        for key, values in results.items():
            importance_df[key] = values
        
        # Add combined score if both methods available
        if 'builtin_importance' in results and 'permutation_importance' in results:
            # Normalize both to 0-1 range
            builtin_norm = results['builtin_importance'] / results['builtin_importance'].sum()
            perm_norm = results['permutation_importance'] / results['permutation_importance'].sum()
            importance_df['combined_importance'] = (builtin_norm + perm_norm) / 2
        
        # Sort by most relevant column
        sort_col = 'combined_importance' if 'combined_importance' in importance_df.columns \
                   else 'builtin_importance' if 'builtin_importance' in importance_df.columns \
                   else 'permutation_importance'
        
        importance_df = importance_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        logger.success(f"Feature importance calculated. Top 5: {importance_df['feature'].head().tolist()}")
        return importance_df

    def save_model(self, path: Optional[Path] = None):
        """Save trained model and artifacts"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        model_path = path or MODELS_DIR / f"{self.classifier_type}_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            model_data = {
                'version': MODEL_VERSION,
                'classifier_type': self.classifier_type,
                'training_date': self.training_date.isoformat(),
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'training_samples': self.training_samples,
                'feature_dim': self.feature_dim
            }
            
            joblib.dump(model_data, model_path)
            logger.success(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise

    @classmethod
    def load_model(cls, classifier_type: str = 'random_forest', path: Optional[Path] = None) -> 'ChurnPredictor':
        """Load saved model"""
        model_path = path or MODELS_DIR / f"{classifier_type}_model.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            model_data = joblib.load(model_path)
            
            predictor = cls(classifier_type=model_data['classifier_type'])
            predictor.model = model_data['model']
            predictor.scaler = model_data['scaler']
            predictor.label_encoders = model_data['label_encoders']
            predictor.feature_names = model_data['feature_names']
            predictor.training_samples = model_data['training_samples']
            predictor.feature_dim = model_data['feature_dim']
            predictor.training_date = datetime.fromisoformat(model_data['training_date'])
            predictor.is_trained = True
            
            logger.success(f"Model loaded from {model_path}")
            return predictor
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise