import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config.settings import DATA_PROCESSED_DIR
from loguru import logger

accounts_path=DATA_PROCESSED_DIR / 'accounts.csv'

def read_and_store_accounts(acc_path:Path):
    if not acc_path or not acc_path.exists():
        logger.error("Path not passed or found")
        raise

    df=pd.read_csv(acc_path)
    print(df.info())


read_and_store_accounts(accounts_path)