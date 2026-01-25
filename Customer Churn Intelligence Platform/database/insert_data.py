import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config.settings import DATA_PROCESSED_DIR
from loguru import logger
from database.operations import insert_new_account

DATA_PROCESSED_CSV=DATA_PROCESSED_DIR /'csv'
accounts_path=DATA_PROCESSED_CSV/ 'accounts.csv'

def read_and_store_accounts(acc_path:Path):
    if not acc_path or not acc_path.exists():
        logger.error("Path not passed or found")
        raise

    df=pd.read_csv(acc_path)
    print(df.info())
    logger.info(f"inserting {len(df)} accouns into database")

    for idx,row in df.iterrows():
        insert_new_account(row.to_dict())
    logger.success("Done inserting Accounts")


read_and_store_accounts(accounts_path)