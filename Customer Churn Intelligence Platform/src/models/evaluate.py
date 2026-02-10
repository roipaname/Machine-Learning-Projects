import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from pathlib import Path
from typing import Optional, Dict
from src.models.train_churn_model import ChurnPredictor
from config.settings import MODELS_DIR

def evaluate_churn_model(
    classifier_type: str = 'random_forest',
    model_path: Optional[Path] = None,
    data_path: str = 'data/training_features.parquet',
    save_results: bool = True
) -> pd.DataFrame:
    """
    Comprehensive model evaluation with business metrics
    
    Args:
        classifier_type: Type of classifier to load
        model_path: Optional custom path to model file
        data_path: Path to training features parquet
        save_results: Whether to save predictions to CSV
        
    Returns:
        DataFrame with test predictions and risk tiers
    """
    logger.info(f"Starting evaluation for {classifier_type} model...")
    
    # === LOAD MODEL ===
    try:
        predictor = ChurnPredictor.load_model(
            classifier_type=classifier_type,
            path=model_path
        )
        logger.success(f"Loaded model trained on {predictor.training_date}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # === LOAD TEST DATA ===
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded dataset: {len(df)} samples")
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        raise
    
    # === PREPARE TEST SET ===
    # Prepare all data using the model's preprocessing
    X, y = predictor.prepare_data(df)
    
    # Use last 15% as test set (time-based split)
    val_idx = int(len(X) * 0.85)
    df_test = df[val_idx:].copy()
    X_test = X[val_idx:]
    y_test = y[val_idx:]
    
    logger.info(f"Test set: {len(X_test)} samples ({len(y_test[y_test==1])} churned)")
    
    # === PREDICTIONS ===
    try:
        y_pred = predictor.predict(X_test)
        y_pred_proba = predictor.predict_proba(X_test)[:, 1]
        logger.success("Predictions generated")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
    
    # === CLASSIFICATION METRICS ===
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':15} Predicted: No | Predicted: Yes")
    print(f"Actual: No      {cm[0,0]:8d}      {cm[0,1]:8d}")
    print(f"Actual: Yes     {cm[1,0]:8d}      {cm[1,1]:8d}")
    
    # Calculate key metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nTrue Positives (Caught churners): {tp}")
    print(f"False Negatives (Missed churners): {fn}")
    print(f"False Positives (False alarms): {fp}")
    print(f"True Negatives (Correct non-churn): {tn}")
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n{'='*60}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*60)
    
    # === BUSINESS METRICS ===
    print("\n" + "="*60)
    print("BUSINESS IMPACT METRICS")
    print("="*60)
    
    # Add predictions to test dataframe
    df_test['churn_probability'] = y_pred_proba
    df_test['predicted_churn'] = y_pred
    
    # Risk tiers
    df_test['risk_tier'] = pd.cut(
        df_test['churn_probability'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Add MRR for business metrics
    df_test['mrr'] = df_test['monthly_fee']
    
    # Revenue at risk calculations
    total_revenue_at_risk = df_test[df_test['churned'] == 1]['mrr'].sum() * 12  # Annual
    captured_revenue_at_risk = df_test[
        (df_test['churned'] == 1) & (df_test['predicted_churn'] == 1)
    ]['mrr'].sum() * 12
    
    revenue_recall = captured_revenue_at_risk / total_revenue_at_risk if total_revenue_at_risk > 0 else 0
    
    print(f"\nRevenue Analysis:")
    print(f"  Total Revenue at Risk: ${total_revenue_at_risk:,.2f}")
    print(f"  Captured by Model: ${captured_revenue_at_risk:,.2f}")
    print(f"  Revenue Recall: {revenue_recall:.2%}")
    
    # False positives cost (unnecessary interventions)
    false_positives = df_test[
        (df_test['churned'] == 0) & (df_test['predicted_churn'] == 1)
    ]
    intervention_cost_per_customer = 100  # $100 per intervention
    total_intervention_cost = len(false_positives) * intervention_cost_per_customer
    
    print(f"\nIntervention Costs:")
    print(f"  False Positive Interventions: {len(false_positives)}")
    print(f"  Wasted Intervention Cost: ${total_intervention_cost:,.2f}")
    print(f"  Cost per False Positive: ${intervention_cost_per_customer}")
    
    # Net value calculation
    avg_customer_value = df_test['mrr'].mean() * 12
    true_positives_value = tp * avg_customer_value * 0.6  # Assume 60% save rate
    net_value = true_positives_value - total_intervention_cost
    
    print(f"\nNet Business Value:")
    print(f"  Potential Saves (60% success): ${true_positives_value:,.2f}")
    print(f"  Intervention Costs: ${total_intervention_cost:,.2f}")
    print(f"  Net Value: ${net_value:,.2f}")
    
    # === RISK TIER DISTRIBUTION ===
    print("\n" + "="*60)
    print("RISK TIER DISTRIBUTION")
    print("="*60)
    
    risk_summary = df_test.groupby('risk_tier', observed=True).agg({
        'customer_id': 'count',
        'churned': 'mean',
        'mrr': 'sum',
        'churn_probability': 'mean'
    }).rename(columns={
        'customer_id': 'count',
        'churned': 'actual_churn_rate',
        'mrr': 'total_mrr',
        'churn_probability': 'avg_predicted_prob'
    })
    
    # Add revenue at risk by tier
    risk_summary['annual_revenue_at_risk'] = risk_summary['total_mrr'] * 12
    
    print(risk_summary.round(4))
    
    # === THRESHOLD ANALYSIS ===
    print("\n" + "="*60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*60)
    
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8]
    threshold_results = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        cm_thresh = confusion_matrix(y_test, y_pred_thresh)
        tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
        
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        
        threshold_results.append({
            'threshold': thresh,
            'precision': prec_t,
            'recall': rec_t,
            'true_positives': tp_t,
            'false_positives': fp_t,
            'false_negatives': fn_t
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    print(threshold_df.round(4))
    
    # === SAVE RESULTS ===
    if save_results:
        output_dir = MODELS_DIR / 'evaluation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions_path = output_dir / f'{classifier_type}_test_predictions.csv'
        df_test.to_csv(predictions_path, index=False)
        logger.success(f"Predictions saved to {predictions_path}")
        
        # Save metrics
        metrics = {
            'model': classifier_type,
            'test_samples': len(y_test),
            'churned_samples': int(y_test.sum()),
            'auc_roc': float(auc),
            'accuracy': float((y_pred == y_test).mean()),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'total_revenue_at_risk': float(total_revenue_at_risk),
            'captured_revenue': float(captured_revenue_at_risk),
            'revenue_recall': float(revenue_recall),
            'false_positive_cost': float(total_intervention_cost),
            'net_business_value': float(net_value)
        }
        
        metrics_path = output_dir / f'{classifier_type}_metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.success(f"Metrics saved to {metrics_path}")
        
        # Save risk summary
        risk_summary_path = output_dir / f'{classifier_type}_risk_summary.csv'
        risk_summary.to_csv(risk_summary_path)
        logger.success(f"Risk summary saved to {risk_summary_path}")
        
        # Save threshold analysis
        threshold_path = output_dir / f'{classifier_type}_threshold_analysis.csv'
        threshold_df.to_csv(threshold_path, index=False)
        logger.success(f"Threshold analysis saved to {threshold_path}")
    
    logger.success("Evaluation complete!")
    return df_test


def plot_evaluation_charts(
    df_test: pd.DataFrame,
    classifier_type: str = 'random_forest',
    save_plots: bool = True
):
    """
    Generate evaluation visualizations
    
    Args:
        df_test: DataFrame with predictions from evaluate_churn_model
        classifier_type: Name for saving plots
        save_plots: Whether to save plots to disk
    """
    output_dir = MODELS_DIR / 'evaluation' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(df_test['churned'], df_test['churn_probability'])
    auc = roc_auc_score(df_test['churned'], df_test['churn_probability'])
    
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_plots:
        plt.savefig(output_dir / f'{classifier_type}_roc_curve.png', dpi=300, bbox_inches='tight')
        logger.info("ROC curve saved")
    plt.close()
    
    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(df_test['churned'], df_test['churn_probability'])
    
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    if save_plots:
        plt.savefig(output_dir / f'{classifier_type}_precision_recall.png', dpi=300, bbox_inches='tight')
        logger.info("Precision-Recall curve saved")
    plt.close()
    
    # 3. Risk Tier Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count by risk tier
    risk_counts = df_test['risk_tier'].value_counts()
    axes[0].bar(risk_counts.index.astype(str), risk_counts.values, color=['green', 'yellow', 'orange', 'red'])
    axes[0].set_xlabel('Risk Tier')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Customer Distribution by Risk Tier')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Actual churn rate by risk tier
    churn_rate = df_test.groupby('risk_tier', observed=True)['churned'].mean()
    axes[1].bar(churn_rate.index.astype(str), churn_rate.values, color=['green', 'yellow', 'orange', 'red'])
    axes[1].set_xlabel('Risk Tier')
    axes[1].set_ylabel('Actual Churn Rate')
    axes[1].set_title('Actual Churn Rate by Predicted Risk Tier')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_dir / f'{classifier_type}_risk_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Risk distribution plots saved")
    plt.close()
    
    # 4. Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    churned = df_test[df_test['churned'] == 1]['churn_probability']
    not_churned = df_test[df_test['churned'] == 0]['churn_probability']
    
    ax.hist(not_churned, bins=30, alpha=0.6, label='Not Churned', color='green')
    ax.hist(churned, bins=30, alpha=0.6, label='Churned', color='red')
    ax.set_xlabel('Predicted Churn Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Probabilities by Actual Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_plots:
        plt.savefig(output_dir / f'{classifier_type}_probability_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Probability distribution saved")
    plt.close()
    
    logger.success("All plots generated!")


if __name__ == "__main__":
    # Evaluate model
    df_results = evaluate_churn_model(
        classifier_type='random_forest',
        save_results=True
    )
    
    # Generate plots
    plot_evaluation_charts(
        df_test=df_results,
        classifier_type='random_forest',
        save_plots=True
    )
    
    logger.success("Evaluation and visualization complete!")