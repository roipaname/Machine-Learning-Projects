import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict
from loguru import logger

#load environment variables
load_dotenv()

# Project root directory and Paths
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR /'data'
DATA_PROCESSED_DIR=DATA_DIR/'processed'
DATA_RAW_DIR=DATA_DIR/'raw'
LOGS_DIR=BASE_DIR /'logs'
MODELS_DIR=BASE_DIR/'models'
SCRIPTS_DIR=BASE_DIR/'scripts'
CONFIG_DIR=BASE_DIR/'config'
SCRAPER_DIR=BASE_DIR/'scraper'

for directory in [DATA_DIR,DATA_PROCESSED_DIR,DATA_RAW_DIR,LOGS_DIR,MODELS_DIR,SCRIPTS_DIR,CONFIG_DIR,SCRAPER_DIR]:
    directory.mkdir(parents=True,exist_ok=True)

#Hugging Face Models
HF_TOKEN=os.getenv("HF_API_TOKEN")
HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"


DB_NAME=os.getenv("DB_CHURN_NAME")

if not DB_NAME:
    logger.error("DB_NAME not set")
    raise
DB_USER=os.getenv("DB_USER")
DB_PASSWORD=os.getenv("DB_PASSWORD")
DB_HOST=os.getenv("DB_HOST","localhost")
DB_PORT=os.getenv("DB_PORT","5432")
DB_TYPE=os.getenv("DB_TYPE","postgres")

DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '20'))
DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
DB_ECHO = os.getenv('DB_ECHO', 'False').lower() == 'true'
