from database.connection import DatabaseConnection
from database.schemas import Customers,Subscriptions,UsageEvents,SupportTickets,Accounts
from datetime import datetime,timedelta
import pandas as pd
from loguru import logger
from typing import Dict,List
from sqlalchemy import func, case


db = DatabaseConnection()

def extract_customer_features(customer_id: str, as_of_date: datetime = None) -> Dict:
    """
    Extract all churn-predictive features for a single customer
    as of a specific date (for point-in-time correctness)
    """
    if as_of_date is None:
        as_of_date=datetime.utcnow()

    with db.get_db() as session:
        # Calculate time windows
        date_30d = as_of_date - timedelta(days=30)
        date_60d = as_of_date - timedelta(days=60)
        date_90d = as_of_date - timedelta(days=90)
        
        features = {}
        
        # Customer basic info
        customer = session.query(Customers).filter_by(
            customer_id=customer_id
        ).first()
        
        if not customer:
            logger.warning(f"Customer {customer_id} not found")
            return None
            
        # Account info
        account = session.query(Accounts).filter_by(
            account_id=customer.account_id
        ).first()
        
        # Subscription info
        subscription = session.query(Subscriptions).filter_by(
            account_id=customer.account_id,
            status='active'
        ).first()
        
        # === DEMOGRAPHIC FEATURES ===
        features['customer_id'] = str(customer_id)
        features['account_tier'] = account.account_tier.value if account else None
        features['contract_type'] = account.contract_type.value if account else None
        features['customer_segment'] = customer.customer_segment.value if customer.customer_segment else None
        features['company_size'] = account.company_size if account else None
        features['acquisition_channel'] = customer.acquisition_channel.value if customer.acquisition_channel else None
        
        # === TENURE FEATURES ===
        features['days_since_signup'] = (as_of_date - customer.signup_date).days
        features['account_age_days'] = (as_of_date - account.created_at).days if account else None
        
        # === SUBSCRIPTION FEATURES ===
        if subscription:
            features['plan_name'] = subscription.plan_name.value
            features['monthly_fee'] = subscription.monthlyfee
            features['days_until_renewal'] = (subscription.end_date - as_of_date).days
            features['contract_length_days'] = (subscription.end_date - subscription.start_date).days
        else:
            features['plan_name'] = None
            features['monthly_fee'] = 0
            features['days_until_renewal'] = None
            features['contract_length_days'] = None

        # === USAGE FEATURES ===
        # Total usage events in different windows
        usage_count_30d=session.query(func.count(UsageEvents.event_id).filter(
            UsageEvents.customer_id==customer_id,
            UsageEvents.timestamp>=date_30d
        )).scalar() or 0

        usage_count_60d=session.query(func.count(UsageEvents.event_id).filter(
            UsageEvents.customer_id==customer_id,
            UsageEvents.timestamp>=date_60d
        )).scalar() or 0

        usage_count_90d=session.query(func.count(UsageEvents.event_id).filter(
            UsageEvents.customer_id==customer_id,
            UsageEvents.timestamp>=date_90d
        )).scalar() or 0

        features['usage_count_30d'] = usage_count_30d
        features['usage_count_60d'] = usage_count_60d
        features['usage_count_90d'] = usage_count_90d

