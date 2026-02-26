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

for directory in [DATA_DIR,DATA_PROCESSED_DIR,DATA_RAW_DIR,LOGS_DIR,MODELS_DIR,SCRIPTS_DIR,CONFIG_DIR]:
    directory.mkdir(parents=True,exist_ok=True)

#Hugging Face Models
HF_TOKEN=os.getenv("HF_API_TOKEN")
HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
