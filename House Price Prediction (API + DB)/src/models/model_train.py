import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime
import os
import sys

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import DatabaseConnection

class ModelTrainer:
    """Train and evaluate house price prediction models"""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()
        self.models = {}
        self.results = []
    
    def load_training_data(self):
        """Load features and target from database"""
        print("Loading training data from database...")
        
        query = """
        SELECT 
            f.*,
            p.price as target_price,
            p.city,
            p.state
        FROM processed.features f
        JOIN raw.properties p ON f.property_id = p.property_id
        WHERE p.price IS NOT NULL
        """
        
        df = self.db.read_sql(query)
        print(f"  Loaded {len(df)} records")
        
        # Separate features and target
        exclude_cols = ['property_id', 'target_price', 'created_at', 
                       'price_category', 'city', 'state','house_age']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target_price']
        
        # Don't fill NaN here - let the pipeline handle it
        print(f"  Features: {len(feature_cols)}")
        print(f"  Feature names: {', '.join(feature_cols)}")
        # Drop columns that are entirely null

        all_null_cols = X.columns[X.isna().all()].tolist()
        if all_null_cols:
            print(f"  Dropping all-null features: {all_null_cols}")
            X = X.drop(columns=all_null_cols)
            feature_cols = [c for c in feature_cols if c not in all_null_cols]

        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, name, pipeline, X_train, X_test, y_train, y_test, features_used):
        """Train a single model pipeline and evaluate"""
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Train
        print("  Fitting model pipeline...")
        pipeline.fit(X_train, y_train)
        
        # Predictions
        print("  Making predictions...")
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Cross-validation
        print("  Running cross-validation...")
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())
        
        # Display results
        print(f"\n  Results:")
        print(f"    Train RMSE: ${train_rmse:,.2f}")
        print(f"    Test RMSE:  ${test_rmse:,.2f}")
        print(f"    CV RMSE:    ${cv_rmse:,.2f} (+/- ${cv_std:,.2f})")
        print(f"    Train R²:   {train_r2:.4f}")
        print(f"    Test R²:    {test_r2:.4f}")
        print(f"    Train MAE:  ${train_mae:,.2f}")
        print(f"    Test MAE:   ${test_mae:,.2f}")
        
        # Feature importance (for tree-based models)
        # Access the actual model from the pipeline
        model = pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features_used,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Top 5 Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Save pipeline (includes imputer + model)
        model_filename = f"{name.lower().replace(' ', '_')}_v1.pkl"
        model_path = os.path.join('models', model_filename)
        os.makedirs('models', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"\n  ✓ Pipeline saved: {model_path}")
        
        # Store results
        result = {
            'model_name': name,
            'model_version': 'v1',
            'algorithm': type(model).__name__,
            'hyperparameters': json.dumps(model.get_params()),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'features_used': features_used,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'trained_at': datetime.now()
        }
        
        self.results.append(result)
        self.models[name] = pipeline
        
        return pipeline, result
    
    def train_all_models(self):
        """Train multiple models with pipelines and compare"""
        print(f"\n{'='*60}")
        print(f"Starting Model Training Pipeline")
        print(f"{'='*60}\n")
        
        # Load data
        X, y, features_used = self.load_training_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Define model pipelines with imputation
        models_config = {
            'Random Forest': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'Gradient Boosting': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ))
            ]),
            'Ridge Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', Ridge(alpha=0.1))
            ]),
            'Lasso Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', Lasso(alpha=0.1, max_iter=1000))
            ])
        }
        
        # Train each model pipeline
        for name, pipeline in models_config.items():
            
            self.train_model(
                name, pipeline, X_train, X_test, y_train, y_test, features_used
               )
        
        # Save results to database
        self.save_results_to_db()
        
        # Display comparison
        self.display_model_comparison()
        
        print(f"\n{'='*60}")
        print(f"Model Training Complete!")
        print(f"{'='*60}\n")
    
    def save_results_to_db(self):
        """Save model metadata to database"""
        print("\nSaving model metadata to database...")
        
        try:
            results_df = pd.DataFrame(self.results)
            results_df.to_sql(
                'model_metadata',
                self.db.engine,
                schema='models',
                if_exists='append',
                index=False
            )
            print(f"✓ Saved metadata for {len(self.results)} models")
        except Exception as e:
            print(f"✗ Failed to save metadata: {e}")
    
    def display_model_comparison(self):
        """Display comparison table of all models"""
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}\n")
        
        comparison_df = pd.DataFrame(self.results)[
            ['model_name', 'test_rmse', 'test_r2', 'test_mae']
        ].sort_values('test_rmse')
        
        # Format for better display
        comparison_df['test_rmse'] = comparison_df['test_rmse'].apply(lambda x: f"${x:,.2f}")
        comparison_df['test_r2'] = comparison_df['test_r2'].apply(lambda x: f"{x:.4f}")
        comparison_df['test_mae'] = comparison_df['test_mae'].apply(lambda x: f"${x:,.2f}")
        
        print(comparison_df.to_string(index=False))
        
        best_model = comparison_df.iloc[0]['model_name']
        print(f"\n🏆 Best Model: {best_model}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()