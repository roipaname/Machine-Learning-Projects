from database.connection import DatabaseConnection
from database.schemas import Customers,Subscriptions,UsageEvents,SupportTickets,Accounts,BillingInvoices,Priority
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
            features['days_until_renewal'] = max((subscription.end_date - as_of_date).days, 0)
            features['contract_length_days'] = (subscription.end_date - subscription.start_date).days
        else:
            features['plan_name'] = None
            features['monthly_fee'] = 0
            features['days_until_renewal'] = None
            features['contract_length_days'] = None

        # === USAGE FEATURES ===
        # Total usage events in different windows
        usage_count_30d=session.query(func.count(UsageEvents.event_id)).filter(
            UsageEvents.customer_id==customer_id,
            UsageEvents.timestamp>=date_30d
        ).scalar() or 0

        usage_count_60d=session.query(func.count(UsageEvents.event_id)).filter(
            UsageEvents.customer_id==customer_id,
            UsageEvents.timestamp>=date_60d
        ).scalar() or 0

        usage_count_90d=session.query(func.count(UsageEvents.event_id)).filter(
            UsageEvents.customer_id==customer_id,
            UsageEvents.timestamp>=date_90d
        ).scalar() or 0

        features['usage_count_30d'] = usage_count_30d
        features['usage_count_60d'] = usage_count_60d
        features['usage_count_90d'] = usage_count_90d

        # Usage decline (key churn signal!)
        if usage_count_60d > 0:
            features['usage_decline_30d_vs_60d'] = (usage_count_60d - usage_count_30d) / usage_count_60d
        else:
            features['usage_decline_30d_vs_60d'] = 0

        last_event=session.query(func.max(UsageEvents.timestamp)).filter(
            UsageEvents.customer_id==customer_id
        ).scalar()


        features['days_since_last_activity'] = (as_of_date - last_event).days if last_event else 999
        # === SUPPORT FEATURES ===
        tickets_30d = session.query(func.count(SupportTickets.ticket_id)).filter(
            SupportTickets.customer_id == customer_id,
            SupportTickets.created_at >= date_30d
        ).scalar() or 0

        

        high_priority_tickets = session.query(func.count(SupportTickets.ticket_id)).filter(
            SupportTickets.customer_id == customer_id,
            SupportTickets.priority == Priority.high,
            SupportTickets.created_at >= date_90d
        ).scalar() or 0
        
        avg_resolution_time = session.query(func.avg(SupportTickets.resolution_time_hours)).filter(
            SupportTickets.customer_id == customer_id,
            SupportTickets.created_at >= date_90d
        ).scalar() or 0
        
        avg_satisfaction = session.query(func.avg(SupportTickets.satisfaction_score)).filter(
            SupportTickets.customer_id == customer_id,
            SupportTickets.created_at >= date_90d
        ).scalar() or 0
        
        features['support_tickets_30d'] = tickets_30d
        features['high_priority_tickets_90d'] = high_priority_tickets
        features['avg_resolution_time_hours'] = float(avg_resolution_time) if avg_resolution_time else 0
        features['avg_satisfaction_score'] = float(avg_satisfaction) if avg_satisfaction else 0
        
        # === BILLING FEATURES ===
        unpaid_invoices = session.query(func.count(BillingInvoices.invoice_id)).filter(
            BillingInvoices.account_id == customer.account_id,
            BillingInvoices.paid == False
        ).scalar() or 0
        
        avg_days_late = session.query(func.avg(BillingInvoices.days_late)).filter(
            BillingInvoices.account_id == customer.account_id,
            BillingInvoices.days_late > 0
        ).scalar() or 0
        
        features['unpaid_invoices'] = unpaid_invoices
        features['avg_days_late'] = float(avg_days_late) if avg_days_late else 0
        
        return features

def extract_all_customers_features(as_of_date=None) -> pd.DataFrame:
    if as_of_date is None:
        as_of_date = datetime.utcnow()

    with db.get_db() as session:
        customers = session.query(Customers.customer_id).all()

        feature_list = []
        for (custid,) in customers:
            features = extract_customer_features(custid, as_of_date)
            if features:
                feature_list.append(features)

    df = pd.DataFrame(feature_list)
    logger.success(f"Extracted features for {len(df)} customers")
    return df


