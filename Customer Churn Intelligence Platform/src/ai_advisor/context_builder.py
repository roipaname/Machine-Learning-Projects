import pandas as pd
from typing import List,Optional,Dict
from database.operations import get_customers_by_account,get_priority_tickets
from src.features.feature_eng import extract_customer_features
import joblib
from loguru import logger
from config.settings import MODELS_DIR,COLUMNS_TO_EXCLUDE
from src.models.classifier import ChurnPredictor
class CustomerContextBuilder:
    def __init__(self, model_type:str="random_forest"):
        self.model_type=model_type
        self.model_path=MODELS_DIR / f"{model_type}_model.joblib"
        if not self.model_path.exists():
            logger.warning(f"Model file {self.model_path} not found. Context builder will not be able to generate predictions.")
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        self.model=ChurnPredictor.load_model(path=self.model_path)
        self.feature_importance=pd.read_csv(MODELS_DIR / f"feature_importance.csv")
    def build_context(self,customer_id:str)->Dict:
        """
        Build concise customer context for LLM
        """
        # Extract features
        features=extract_customer_features(customer_id)
        if features is None:
            logger.warning(f"No features found for customer {customer_id}. Context will be limited.")
            return {"customer_id":customer_id,"context":"No data available to build context"}
        
        #Get Churn prediction
        X=pd.DataFrame([features]).drop(columns=COLUMNS_TO_EXCLUDE,errors='ignore')
        X_scaled, _ = self.model.prepare_data(X, fit_encoders=False)
        churn_proba=self.model.predict_proba(X_scaled)[0,1]

        #Determine Risk Tier:

        if churn_proba>0.8:
            risk_tier="Critical"
        elif churn_proba>0.6:
            risk_tier="High"
        elif churn_proba>0.3:
            risk_tier="Medium"
        else:
            risk_tier="Low"

        top_features=self.feature_importance.head(10)['feature'].tolist()
        top_drivers=[{'feature':feat,'value':features[feat]} for feat in top_features if feat in features ]

        context={
            'customer_id':customer_id,
            'churn_probability': round(churn_proba,4),
            'risk_tier': risk_tier,
            'account_info':{
                'account_tier': features.get('account_tier'),
                'contract_type': features.get('contract_type'),
                'customer_segment': features.get('customer_segment'),
                'company_size': features.get('company_size'),
                'acquisition_channel': features.get('acquisition_channel'),
                'monthly_fee': features.get('monthly_fee'),
                'days_until_renewal': features.get('days_until_renewal')
            },
            'engagement_signals':{
                'usage_count_30d': features.get('usage_count_30d'),
                'usage_count_60d': features.get('usage_count_60d'),
                'usage_decline_30d_vs_60d':features.get('usage_decline_30d_vs_60d'),
                'days_since_last_activity': features.get('days_since_last_activity'),
            },
            'support_signals':{
                'tickets_30d':features.get('support_tickets_30d'),
                'high_priority_tickets_90d':features.get('high_priority_tickets_90d'),
            'avg_satisfaction_score': features.get('avg_satisfaction_score')
            },
            'billing_signals': {
                'unpaid_invoices': features.get('unpaid_invoices'),
                'avg_days_late': features.get('avg_days_late')
            },
            'top_churn_drivers': top_drivers[:5]  # Top 5 only

        }

        logger.success(f"Context built for customer {customer_id} with churn probability {churn_proba:.4f} and risk tier {risk_tier}")
        return context
if __name__ == "__main__":
    builder = CustomerContextBuilder()
    # Replace with actual customer_id
    context = builder.build_context("cb6761fd-f237-4a50-84e4-a38285b77648")
    print(context)