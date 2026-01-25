import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config.settings import DATA_PROCESSED_DIR
from loguru import logger
from database.operations import insert_new_account,insert_customer,create_subscription,create_support_ticket

DATA_PROCESSED_CSV=DATA_PROCESSED_DIR /'csv'
accounts_path=DATA_PROCESSED_CSV/ 'accounts.csv'
customer_path=DATA_PROCESSED_CSV/ 'customers.csv'
subscription_path=DATA_PROCESSED_CSV / 'subscriptions.csv'

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
    


#read_and_store_accounts(accounts_path)

#read_and_store_cusomers(customer_path)

read_and_store_subs(subscription_path)