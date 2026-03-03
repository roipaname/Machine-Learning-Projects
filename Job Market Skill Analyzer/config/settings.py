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

"""
Classifier and AI advisor details

"""


#Hugging Face Models
HF_TOKEN=os.getenv("HF_API_TOKEN")
HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"


AVAILABLE_CLASSIFIERS=[ 'logistic_regression',
    'naive_bayes',
    'svm',
    'random_forest']


# Model versioning
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')

# Default classifier
DEFAULT_CLASSIFIER = os.getenv('DEFAULT_CLASSIFIERS', 'random_forest')

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
CV_FOLDS = int(os.getenv('CV_FOLDS', '5'))


# Minimum confidence threshold for classification
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.5'))


# Similarity threshold for near-duplicate detection
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))

# Hash algorithm for content hashing
HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')


# =========================================================
# RAG (RETRIEVAL AUGMENTED GENERATION)
# =========================================================

RAG_CONFIG = {
    "enabled": True,
    "embedding_model": "text-embedding-3-large",
    "vector_store": "faiss",  # faiss | chroma | pinecone
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 5,
}


# =========================================================
# OUTPUT REGULARIZATION
# =========================================================

OUTPUT_RULES = {
    "allow_probabilities": True,
    "confidence_threshold": 0.65,
    "disallowed_phrases": [
        "guaranteed",
        "100% sure",
        "no risk"
    ],
    "explanation_style": "business_friendly",  # technical | exec | business
}

# =========================================================
# DECISION INTELLIGENCE
# =========================================================

DECISION_POLICY = {
    "high_risk_threshold": 0.8,
    "medium_risk_threshold": 0.5,
    "actions": {
        "high": "immediate_retention_offer",
        "medium": "engagement_campaign",
        "low": "monitor_only"
    }
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Log file paths
LOG_FILE = LOGS_DIR / 'app.log'
ERROR_LOG_FILE = LOGS_DIR / 'error.log'

# Log rotation
LOG_ROTATION = os.getenv('LOG_ROTATION', '10 MB')
LOG_RETENTION = os.getenv('LOG_RETENTION', '30 days')

# Log format
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


# =========================================================
# API SETTINGS
# =========================================================

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
}

def get_model_config() -> Dict:
    """Get model configuration as dictionary."""
    return {
        'classifier': DEFAULT_CLASSIFIER,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'cv_folds': CV_FOLDS,
        'model_version': MODEL_VERSION,
        'min_confidence': MIN_CONFIDENCE_THRESHOLD
    }


__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'get_model_config'
    'KAGGLE_TRAIN_DATASET'
]

from loguru import logger

logger.remove()
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format=LOG_FORMAT
)