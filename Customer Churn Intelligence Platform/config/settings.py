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




# Minimum confidence threshold for classification
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.5'))


# Similarity threshold for near-duplicate detection
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))

# Hash algorithm for content hashing
HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')

# =========================================================
# PROMPT ENGINEERING
# =========================================================

PROMPT_CONFIG = {
    "prompt_version": "v1.0",
    "templates_path": PROMPT_DIR / "templates",
    "default_system_prompt": (
        "You are a churn analysis assistant. "
        "Explain predictions clearly and avoid speculation."
    ),
    "max_prompt_tokens": 2048,
}



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

# =========================================================
# GOVERNANCE & MONITORING
# =========================================================

GOVERNANCE = {
    "log_predictions": True,
    "log_prompts": True,
    "store_explanations": True,
    "audit_path": BASE_DIR / "docs" / "audit_logs",
    "pii_columns": ["email", "phone_number"],
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