from loguru import logger
from database.connection import DatabaseConnection
from database.schemas import Accounts,Customers,Subscriptions,UsageEvents,BillingInvoices,SupportTickets,SubscriptionStatus,Churn_Labels
from typing import List,Dict,Optional
from sqlalchemy import func, and_, not_
from sqlalchemy.orm import Session
from datetime import datetime
db=DatabaseConnection()

def insert_new_account(account_data:Dict):
    account=Accounts(**account_data)

    try:
        with db.get_db() as session:
            if check_if_account_exist(account_data['account_id']):
                logger.info(f"Account with ID:{account_data['account_id']} already exist")
                return
            account=Accounts(**account_data)
            session.add(account)
            session.flush()
            logger.success(f"Account  Account with ID:{account_data['account_id']} created  Successfully") 
            return account.account_id
    except Exception as e:
        logger.error(f"Failed to create account:{e}")
        raise           


def check_if_account_exist(account_id:str):

    try:
        with db.get_db() as session:
            result=session.query(Accounts).filter_by(account_id=account_id).first()
            if not result:
                return False
            return True
        
    except Exception as e:
        logger.error(f"failed to check if account with Id {account_id} exist")
        raise

def get_accounts_by_company_name(company_name:str)->List[Accounts]:
    try:
        with db.get_db() as session:

            if company_name:
                results=session.query(Accounts).filter_by(
                    company_name=company_name
                ).all()
                return results
    except Exception as e:
        logger.error(f"failed to check if account withs with company name {company_name} exist")
        raise
    
def update_account(account_id: str, updates: Dict) -> bool:
    try:
        with db.get_db() as session:
            account = session.query(Accounts).filter_by(account_id=account_id).first()
            if not account:
                return False

            for key, value in updates.items():
                setattr(account, key, value)

            session.flush()
            logger.success(f"Account {account_id} updated successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to update account {account_id}: {e}")
        raise
def delete_account(account_id: str) -> bool:
    try:
        with db.get_db() as session:
            account = session.query(Accounts).filter_by(account_id=account_id).first()
            if not account:
                return False

            session.delete(account)
            logger.success(f"Account {account_id} deleted successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to delete account {account_id}: {e}")
        raise
def get_all_accounts(limit: int = 100, offset: int = 0) -> List[Accounts]:
    try:
        with db.get_db() as session:
            return (
                session.query(Accounts)
                .order_by(Accounts.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
    except Exception as e:
        logger.error("Failed to fetch accounts")
        raise
def insert_customer(customer_data: Dict):
    try:
        with db.get_db() as session:
            customer = Customers(**customer_data)
            session.add(customer)
            session.flush()
            logger.success(f"Customer {customer.customer_id} created")
            return customer.customer_id
    except Exception as e:
        logger.error(f"Failed to create customer: {e}")
        raise

def get_customers_by_account(account_id: str) -> List[Customers]:
    try:
        with db.get_db() as session:
            return (
                session.query(Customers)
                .filter_by(account_id=account_id)
                .all()
            )
    except Exception as e:
        logger.error(f"Failed to fetch customers for account {account_id}")
        raise
def check_customer_email_exists(email: str) -> bool:
    try:
        with db.get_db() as session:
            return (
                session.query(Customers)
                .filter_by(email=email)
                .first()
                is not None
            )
    except Exception as e:
        logger.error("Failed to check customer email")
        raise
def create_subscription(subscription_data: Dict):
    try:
        with db.get_db() as session:
            subscription = Subscriptions(**subscription_data)
            session.add(subscription)
            session.flush()
            logger.success(f"Subscription {subscription.subscription_id} created")
            return subscription.subscription_id
    except Exception as e:
        logger.error(f"Failed to create subscription: {e}")
        raise
def get_active_subscription(account_id: str) -> Optional[Subscriptions]:
    try:
        with db.get_db() as session:
            return (
                session.query(Subscriptions)
                .filter_by(account_id=account_id, status="active")
                .order_by(Subscriptions.start_date.desc())
                .first()
            )
    except Exception as e:
        logger.error(f"Failed to fetch subscription for account {account_id}")
        raise

def bulk_insert_usage_events(events: List[Dict], batch_size: int = 500):
    try:
        with db.get_db() as session:
            for i in range(0, len(events), batch_size):
                batch = [UsageEvents(**e) for e in events[i:i + batch_size]]
                session.bulk_save_objects(batch)
            logger.success(f"Inserted {len(events)} usage events")
    except Exception as e:
        logger.error("Failed to bulk insert usage events")
        raise

def get_usage_events(customer_id: str, limit: int = 1000):
    try:
        with db.get_db() as session:
            return (
                session.query(UsageEvents)
                .filter_by(customer_id=customer_id)
                .limit(limit)
                .all()
            )
    except Exception as e:
        logger.error("Failed to fetch usage events")
        raise
def insert_invoice(invoice_data: Dict):
    try:
        with db.get_db() as session:
            invoice = BillingInvoices(**invoice_data)
            session.add(invoice)
            session.flush()
            logger.success(f"Invoice {invoice.invoice_id} created")
            return invoice.invoice_id
    except Exception as e:
        logger.error("Failed to create invoice")
        raise

def get_unpaid_invoices(account_id: str) -> List[BillingInvoices]:
    try:
        with db.get_db() as session:
            return (
                session.query(BillingInvoices)
                .filter_by(account_id=account_id, paid=False)
                .all()
            )
    except Exception as e:
        logger.error("Failed to fetch unpaid invoices")
        raise

def get_paid_invoices(account_id: str) -> List[BillingInvoices]:
    try:
        with db.get_db() as session:
            return (
                session.query(BillingInvoices)
                .filter_by(account_id=account_id, paid=True)
                .all()
            )
    except Exception as e:
        logger.error("Failed to fetch paid invoices")
        raise
def create_support_ticket(ticket_data: Dict):
    try:
        with db.get_db() as session:
            ticket = SupportTickets(**ticket_data)
            session.add(ticket)
            session.flush()
            logger.success(f"Support ticket {ticket.ticket_id} created")
            return ticket.ticket_id
    except Exception as e:
        logger.error("Failed to create support ticket")
        raise


def get_priority_tickets(customer_id: str,priority:str="high"):
    try:
        with db.get_db() as session:
            return (
                session.query(SupportTickets)
                .filter_by(customer_id=customer_id, priority=priority)
                .all()
            )
    except Exception as e:
        logger.error(f"Failed to fetch {priority} priority tickets")
        raise


def count_usage_events(customer_id: str) -> int:
    with db.get_db() as session:
        return (
            session.query(UsageEvents)
            .filter_by(customer_id=customer_id)
            .count()
        )

def avg_resolution_time(account_id: str) -> float:
    with db.get_db() as session:
        return (
            session.query(SupportTickets.resolution_time_hours)
            .join(Customers)
            .filter(Customers.account_id == account_id)
            .scalar()
            or 0.0
        )
if __name__ == "__main__":
    import uuid
    from datetime import datetime, timedelta
    from loguru import logger

    logger.info("Starting manual DB function tests...")

    # =========================
    # TEST DATA
    # =========================
    account_id = uuid.uuid4()
    customer_id = uuid.uuid4()

    account_data = {
        "account_id": account_id,
        "company_name": "Paname AI",
        "industry": "SaaS",
        "company_size": 12,
        "contract_type": "monthly",
        "account_tier": "gold",
    }

    customer_data = {
        "customer_id": customer_id,
        "account_id": account_id,
        "first_name": "Clarence",
        "last_name": "Ebebe",
        "email": f"clarence.{uuid.uuid4()}@paname.ai",
        "country": "ZA",
        "acquisition_channel": "organic",
        "customer_segment": "SMB",
    }

    subscription_data = {
        "account_id": account_id,
        "plan_name": "pro",
        "monthlyfee": 499.99,
        "start_date": datetime.utcnow(),
        "end_date": datetime.utcnow() + timedelta(days=365),
        "status": "active",
    }

    invoice_data = {
        "account_id": account_id,
        "invoice_date": datetime.utcnow(),
        "paid": False,
        "days_late": 5,
    }

    ticket_data = {
        "customer_id": customer_id,
        "issue_type": "billing",
        "priority": "high",
        "resolution_time_hours": 48.5,
        "satisfaction_score": 2.0,
    }

    usage_events = [
        {
            "customer_id": customer_id,
            "event_type": "apicall",
            "device_type": "web",
        }
        for _ in range(20)
    ]

    try:
        # =========================
        # ACCOUNT
        # =========================
        logger.info("Testing account creation...")
        insert_new_account(account_data)

        exists = check_if_account_exist(account_id)
        logger.info(f"Account exists: {exists}")

        accounts = get_accounts_by_company_name("Paname AI")
        logger.info(f"Accounts fetched by name: {len(accounts)}")

        # =========================
        # CUSTOMER
        # =========================
        logger.info("Testing customer creation...")
        insert_customer(customer_data)

        customers = get_customers_by_account(account_id)
        logger.info(f"Customers under account: {len(customers)}")

        email_exists = check_customer_email_exists(customer_data["email"])
        logger.info(f"Customer email exists: {email_exists}")

        # =========================
        # SUBSCRIPTION
        # =========================
        logger.info("Testing subscription creation...")
        create_subscription(subscription_data)

        active_sub = get_active_subscription(account_id)
        logger.info(
            f"Active subscription plan: {active_sub.plan_name if active_sub else 'None'}"
        )

        # =========================
        # USAGE EVENTS
        # =========================
        logger.info("Testing bulk usage events insert...")
        bulk_insert_usage_events(usage_events)

        usage_count = count_usage_events(customer_id)
        logger.info(f"Total usage events: {usage_count}")

        # =========================
        # BILLING
        # =========================
        logger.info("Testing billing invoice...")
        insert_invoice(invoice_data)

        unpaid = get_unpaid_invoices(account_id)
        logger.info(f"Unpaid invoices: {len(unpaid)}")

        # =========================
        # SUPPORT TICKETS
        # =========================
        logger.info("Testing support ticket...")
        create_support_ticket(ticket_data)

        high_priority = get_priority_tickets(customer_id)
        logger.info(f"High priority tickets: {len(high_priority)}")

        logger.success("ALL MANUAL DB TESTS COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.exception("MANUAL DB TEST FAILED")


def get_churned_accounts():
    """
    An account is churned if it has NO active subscription.
    """

    with db.get_db() as session:
        active_subq=(session.query(Subscriptions).filter(Subscriptions.status==SubscriptionStatus.active,Subscriptions.end_date>= datetime.utcnow()).subquery())
        churned_accounts=(
            session.query(Subscriptions.account_id,func.max(
                Subscriptions.end_date).label("churn_date")).filter(
                    not_(Subscriptions.account_id.in_(active_subq))
                ).group_by(Subscriptions.account_id).all())

        return churned_accounts


def generate_customer_churn_labels():

    with db.get_db() as session:
        churned_accounts=get_churned_accounts()
        if not churned_accounts:
            logger.info("No Churned Accounst found.")
            return
        churned_account_map={
            acc.account_id:acc.churn_date for acc in churned_accounts
        }
        customers=(
            session.query(Customers).filter(Customers.account_id.in_(churned_account_map.keys())).all

        )

        for customer in customers:
            churn_date=churned_account_map[customer.account_id]
            exists=(
                session.query(
                    Churn_Labels
                ).filter(Churn_Labels.customer_id==customer.customer_id).first()
            )
            if exists:
                continue

            label=Churn_Labels(
                customer_id=customer.customer_id,
                churned=True,
                churn_date=churn_date,
                churn_reason='Account subscription ended'
            )

            session.add(label)
            session.flush()
            inserted+=1
        session.commit()
        logger.info(f"Inserted {inserted} churn labels.")
