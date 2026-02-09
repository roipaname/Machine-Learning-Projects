import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.classifier import ChurnPredictor
from config.settings import DATA_PROCESSED_DIR
from loguru import logger
data_parquet_path=DATA_PROCESSED_DIR /'training_features.parquet'

def train_model(data_path):
    #load dataset
    df=pd.read_parquet(data_path)

    logger.success(f"Dataset loaded successfully from {data_path}:{len(df)} rows")
    logger.info(f"Dataset: {len(df)} samples, {df['churned'].sum()} churned ({df['churned'].mean():.2%})")

    predictor=ChurnPredictor()
    X,y=predictor.prepare_data(df)
     #Time-based split (important for churn!)
    # Use first 70% for training, next 15% for validation, last 15% for test
    train_idx = int(len(X) * 0.7)
    val_idx = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    results=predictor.train_models(X_train, y_train, X_val, y_val)
    shap_values,feature_importance=predictor.calculate_feature_importance(X_val, y_val)
    print("\nTop 10 Churn Drivers:")
    print(feature_importance.head(10))
    print("="*50)
    print(shap_values[:5])  # Show SHAP values for first 5 samples
    
    # Save
    predictor.save_model()
    feature_importance.to_csv('models/artifacts/feature_importance.csv', index=False)


if __name__=="__main__":
    train_model()