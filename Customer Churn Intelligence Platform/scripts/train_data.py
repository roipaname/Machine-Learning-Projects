import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.classifier import ChurnPredictor
from config.settings import DATA_PROCESSED_DIR, MODELS_DIR
from loguru import logger

data_parquet_path = DATA_PROCESSED_DIR / 'training_features.parquet'

def train_model(data_path):
    """Train and compare multiple churn prediction models."""
    
    # Load dataset
    df = pd.read_parquet(data_path)
    logger.success(f"Dataset loaded successfully from {data_path}: {len(df)} rows")
    logger.info(f"Dataset: {len(df)} samples, {df['churned'].sum()} churned ({df['churned'].mean():.2%})")
    
    # Initialize predictor and prepare data
    predictor = ChurnPredictor()
    X, y = predictor.prepare_data(df, fit_encoders=True)
    
    # Time-based split (important for churn!)
    # Use first 70% for training, next 15% for validation, last 15% for test
    train_idx = int(len(X) * 0.7)
    val_idx = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    logger.info(f"Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Churn rates - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Train multiple models and compare
    logger.info("\n" + "="*60)
    logger.info("TRAINING MULTIPLE MODELS")
    logger.info("="*60)
    
    results = predictor.train_models(
        X_train, y_train, 
        X_val, y_val,
        models_to_train=None,  # Train all available models
        tune_hyperparameters=False  # Set to True for hyperparameter tuning
    )
    
    # The best model is now loaded in predictor
    best_model_name = results['summary']['best_model']
    logger.success(f"\nBest model selected: {best_model_name}")
    
    # Evaluate best model on test set
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)
    
    test_metrics = predictor.evaluate(X_test, y_test)
    logger.info(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"\nClassification Report:")
    
    # Print classification report nicely
    report = test_metrics['classification_report']
    for label in ['0', '1']:
        if label in report:
            logger.info(f"  Class {label}: Precision={report[label]['precision']:.3f}, "
                       f"Recall={report[label]['recall']:.3f}, F1={report[label]['f1-score']:.3f}")
    
    # Feature importance for best model
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE (Best Model)")
    logger.info("="*60)
    
    feature_importance = predictor.calculate_feature_importance(X_val, y_val, method='auto')
    
    print("\nTop 10 Churn Drivers:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save best model
    logger.info("\n" + "="*60)
    logger.info("SAVING MODEL AND RESULTS")
    logger.info("="*60)
    
    predictor.save_model()
    
    # Save feature importance
    importance_path = MODELS_DIR / 'feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    logger.success(f"Feature importance saved to {importance_path}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(results['summary']['comparison'])
    comparison_path = MODELS_DIR / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.success(f"Model comparison saved to {comparison_path}")
    
    # Save test results
    test_results = {
        'best_model': best_model_name,
        'test_auc': test_metrics['roc_auc'],
        'test_accuracy': test_metrics['accuracy'],
        'confusion_matrix': test_metrics['confusion_matrix'],
        'classification_report': test_metrics['classification_report']
    }
    
    import json
    results_path = MODELS_DIR / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.success(f"Test results saved to {results_path}")
    
    logger.success("\n✅ Training pipeline completed successfully!")
    
    return predictor, results, test_metrics


if __name__ == "__main__":
    train_model(data_parquet_path)