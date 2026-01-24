from loguru import logger
from database.connection import DatabaseConnection
from database.schemas import Accounts,Customers,Subscriptions,UsageEvents,BillingInvoices,SupportTickets
from typing import List,Dict,Optional
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
