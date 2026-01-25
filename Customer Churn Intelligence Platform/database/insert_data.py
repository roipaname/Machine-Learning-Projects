import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config.settings import DATA_PROCESSED_DIR
from loguru import logger
from database.operations import insert_new_account,insert_customer,create_subscription,create_support_ticket,bulk_insert_usage_events,insert_invoice

DATA_PROCESSED_CSV=DATA_PROCESSED_DIR /'csv'
accounts_path=DATA_PROCESSED_CSV/ 'accounts.csv'
customer_path=DATA_PROCESSED_CSV/ 'customers.csv'
subscription_path=DATA_PROCESSED_CSV / 'subscriptions.csv'
support_path=DATA_PROCESSED_CSV / 'support_tickets.csv'
usage_path=DATA_PROCESSED_CSV / 'usage_events.csv'
invoice_path=DATA_PROCESSED_CSV / 'billing_invoices.csv'

def read_and_store_accounts(acc_path:Path):
    if not acc_path or not acc_path.exists():
        logger.error("Path not passed or found")
        raise

    df=pd.read_csv(acc_path)
    print(df.info())
    logger.info(f"inserting {len(df)} accouns into database")

    for idx,row in df.iterrows():
        try:
            insert_new_account(row.to_dict())
        except Exception as e:
            logger.error(f"Failed to insert account {row['account_id']}")
    logger.success("Done inserting Accounts")

def read_and_store_cusomers(customer_path:Path):
    if not customer_path or not customer_path.exists():
        logger.error("Path not passed or found")
        raise

    df=pd.read_csv(customer_path)
    print(df.info())
    logger.info(f"inserting {len(df)} customers into database")

    for idx,row in df.iterrows():
        try:
            insert_customer(row.to_dict())
        except Exception as e:
            logger.error(f"Failed to insert customer {row['customer_id']}")
    logger.success("Done inserting Accounts")

def read_and_store_subs(subpath:Path):
    if not subpath or not subpath.exists():
        logger.error("Path not passed or found")
        raise
    df= pd.read_csv(subpath)
    print(df.info())
    logger.info(f"inserting {len(df)} subs into database")

    for idx ,row in df.iterrows():
        try:
            create_subscription(row.to_dict())
        except Exception as e:
            logger.error(f"failed to insert sub id {row['subscription_id']}")
            raise
        logger.success("Done inserting Subscriptions")
    
def read_and_store_support(support_path:Path):
    if not support_path or not support_path.exists():
        logger.error("Path not passed or found")
        raise
    df= pd.read_csv(support_path)
    print(df.info())
    logger.info(f"inserting {len(df)} support into database")

    for idx ,row in df.iterrows():
        try:
            create_support_ticket(row.to_dict())
        except Exception as e:
            logger.error(f"failed to insert support id {row['ticket_id']}")
            raise
    logger.success("Done inserting support")
    
def read_and_store_usages(usage_path:Path):
    if not usage_path or not usage_path.exists():
        logger.error("Path not passed or found")
        raise
    df= pd.read_csv(usage_path)
    print(df.info())
    usages=df.to_dict(orient='records')
    logger.info(f"inserting {len(df)} usages into database")
    try:
       bulk_insert_usage_events(usages)
    except Exception as e:
        logger.error("failed to insert usages")
        raise
    logger.success("Done inserting usages")

def read_and_store_invoices(invoice_path:Path):
    if not invoice_path or not invoice_path.exists():
        logger.error("Path not passed or found")
        raise
    df= pd.read_csv(invoice_path)
    print(df.info())
    logger.info(f"inserting {len(df)} invoices into database")

    for idx ,row in df.iterrows():
        try:
            insert_invoice(row.to_dict())
        except Exception as e:
            logger.error(f"failed to insert invoice id {row['invoice_id']}")
            raise
    logger.success("Done inserting invoices") 

#read_and_store_accounts(accounts_path)

#read_and_store_cusomers(customer_path)

#read_and_store_subs(subscription_path)
#read_and_store_support(support_path)
#read_and_store_usages(usage_path)
#read_and_store_invoices(invoice_path)