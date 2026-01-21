import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from typing import Dict,List
# Load environment variables from .env file
load_dotenv()
# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR / 'data'
MODELS_DIR=BASE_DIR / 'models'
LOGS_DIR=BASE_DIR  /' logs'
CONFIG_DIR=BASE_DIR / 'config'
DATA_RAW_DIR=DATA_DIR /'raw'
DATA_PROCESSED_DIR=DATA_DIR /'processed'
DATA_FEATURE_DIR=DATA_DIR /'feature store'
KAGGLE_TRAIN_DATASET=DATA_DIR / 'raw/training.csv'
KAGGLE_TEST_DATASET=DATA_DIR / 'raw/test.csv'

SRC_DIR = BASE_DIR / "src"
MODEL_DIR = SRC_DIR / "ml_models"
PROMPT_DIR = SRC_DIR / "prompt_engineering"
RAG_DIR = SRC_DIR / "rag"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TESTS_DIR = BASE_DIR / "tests"


for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR,DATA_RAW_DIR,DATA_PROCESSED_DIR,DATA_FEATURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE SET UP
# ============================================================================


DB_NAME=os.getenv("customerchurn")

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


# =========================================================
# FEATURE STORE CONFIG
# =========================================================

FEATURE_STORE_CONFIG = {
    "backend": "filesystem",  # later: feast, redis, s3
    "path": DATA_FEATURE_DIR,
    "versioning": True,
    "entity_key": "customer_id",
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model types available
AVAILABLE_CLASSIFIERS = [
    'logistic_regression',
    'naive_bayes',
    'svm',
    'random_forest'
]
# Model versioning
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')

# Default classifier
DEFAULT_CLASSIFIER = os.getenv('DEFAULT_CLASSIFIER', 'xgboost')

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
CV_FOLDS = int(os.getenv('CV_FOLDS', '5'))