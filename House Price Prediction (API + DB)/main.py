import os
from dotenv import load_dotenv
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
df=pd.read_csv(os.getenv("PATH_TO_DATASET"))

logging.info(f"Dataframe shape: {df.shape}")
logging.info(f"Dataframe columns: {df.columns.tolist()}")