from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
import os
from typing import Dict,List
# Load environment variables from .env file
load_dotenv()





# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR/ 'data'
MODELS_DIR=BASE_DIR / 'models'
LOGS_DIR=BASE_DIR  /' logs'
VECTORIZER_SAVE_PATH=BASE_DIR /'vectorizer'

KAGGLE_TRAIN_DATASET=DATA_DIR / 'raw/twitter_training.csv'
KAGGLE_TEST_DATASET=DATA_DIR / 'raw/twitter_training.csv'


