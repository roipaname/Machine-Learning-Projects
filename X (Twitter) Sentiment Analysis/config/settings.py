from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
import os
import sys
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
CONFIG_DIR=BASE_DIR / 'config'
VECTORIZER_SAVE_PATH=BASE_DIR /'vectorizer'

KAGGLE_TRAIN_DATASET=DATA_DIR / 'raw/twitter_training.csv'
KAGGLE_TEST_DATASET=DATA_DIR / 'raw/twitter_training.csv'

for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR,VECTORIZER_SAVE_PATH,]:
    directory.mkdir(parents=True, exist_ok=True)

MONGODBURL=os.getenv("MONGODBURL")

if not MONGODBURL:
    logger.error("MONGODB URL not available")
    raise
# Database connection pool settings
DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '20'))
DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
DB_ECHO = os.getenv('DB_ECHO', 'False').lower() == 'true'
DB_BNAME=os.getenv("DB_BNAME")



# Request headers
DEFAULT_HEADERS = {
    
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Rate limiting
DEFAULT_RATE_LIMIT = float(os.getenv('DEFAULT_RATE_LIMIT', '2.0'))  # seconds between requests
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '10'))  # seconds
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))


# Scraping limits
MAX_TWEET_PER_SOURCE = int(os.getenv('MAX_ARTICLES_PER_SOURCE', '100'))
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))

# Content validation
MIN_ARTICLE_LENGTH = int(os.getenv('MIN_ARTICLE_LENGTH', '100'))  # characters
MAX_ARTICLE_LENGTH = int(os.getenv('MAX_ARTICLE_LENGTH', '50000'))  # characters


# Token filtering
MIN_TOKEN_LENGTH = int(os.getenv('MIN_TOKEN_LENGTH', '3'))
MAX_TOKEN_LENGTH = int(os.getenv('MAX_TOKEN_LENGTH', '20'))

# Preprocessing options
REMOVE_NUMBERS = os.getenv('REMOVE_NUMBERS', 'True').lower() == 'true'
REMOVE_URLS = os.getenv('REMOVE_URLS', 'True').lower() == 'true'
AGGRESSIVE_CLEANING = os.getenv('AGGRESSIVE_CLEANING', 'True').lower() == 'true'

# Custom stopwords for news domain
CUSTOM_STOPWORDS = [
    'said', 'told', 'says', 'according', 'report', 'reports',
    'article', 'news', 'source', 'sources', 'journalist',
    'correspondent', 'editor', 'published', 'story', 'coverage'
]

# Language settings
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
SUPPORTED_LANGUAGES = ['en', 'sp', 'fr', 'ge']
# ============================================================================
# FEATURE ENGINEERING (TF-IDF) CONFIGURATION
# ============================================================================

# TF-IDF parameters
TFIDF_MAX_FEATURES = int(os.getenv('TFIDF_MAX_FEATURES', '5000'))
TFIDF_MIN_DF = int(os.getenv('TFIDF_MIN_DF', '2'))  # Minimum document frequency
TFIDF_MAX_DF = float(os.getenv('TFIDF_MAX_DF', '0.85'))  # Maximum document frequency
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# Feature selection
USE_IDF = os.getenv('USE_IDF', 'True').lower() == 'true'
SUBLINEAR_TF = os.getenv('SUBLINEAR_TF', 'True').lower() == 'true'


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
DEFAULT_CLASSIFIER = os.getenv('DEFAULT_CLASSIFIER', 'logistic_regression')

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
CV_FOLDS = int(os.getenv('CV_FOLDS', '5'))

# Model persistence
MODEL_SAVE_PATH = MODELS_DIR / 'classifier_model.pkl'
VECTORIZER_SAVE_PATH = MODELS_DIR / 'tfidf_vectorizer.pkl'
LABEL_ENCODER_SAVE_PATH = MODELS_DIR / 'label_encoder.pkl'

# Model versioning


# Minimum confidence threshold for classification
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.5'))


# Similarity threshold for near-duplicate detection
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))

# Hash algorithm for content hashing
HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')



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



# ============================================================================
# API/SCHEDULER CONFIGURATION (
# ============================================================================

# Scheduler settings
ENABLE_SCHEDULER = os.getenv('ENABLE_SCHEDULER', 'False').lower() == 'true'
SCRAPER_SCHEDULE = os.getenv('SCRAPER_SCHEDULE', '0 */6 * * *')  # Every 6 hours
PREPROCESSING_SCHEDULE = os.getenv('PREPROCESSING_SCHEDULE', '30 */6 * * *')

# API settings (if building REST API)
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'



def validate_settings():
    """Validate critical settings on startup."""
    errors = []
    
    # Check database URL
    if 'mongodb://' not in MONGODBURL:
        errors.append("DATABASE_URL must be a valid PostgreSQL connection string")
    
    # Check model directory
    if not MODELS_DIR.exists():
        try:
            MODELS_DIR.mkdir(parents=True)
        except Exception as e:
            errors.append(f"Cannot create models directory: {e}")
    
    
    
    # Validate TF-IDF parameters
    if TFIDF_MAX_DF <= 0 or TFIDF_MAX_DF > 1:
        errors.append("TFIDF_MAX_DF must be between 0 and 1")
    
    if TFIDF_MIN_DF < 1:
        errors.append("TFIDF_MIN_DF must be at least 1")
    
    # Validate test size
    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        errors.append("TEST_SIZE must be between 0 and 1")
    
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration validation failed with {len(errors)} error(s)")
    
    logger.info("Configuration validated successfully")


def get_db_config() -> Dict:
    """Get database configuration as dictionary."""
    return {
        'url': MONGODBURL,
        'pool_size': DB_POOL_SIZE,
        'max_overflow': DB_MAX_OVERFLOW,
        'pool_timeout': DB_POOL_TIMEOUT,
        'echo': DB_ECHO
    }


def get_preprocessing_config() -> Dict:
    """Get preprocessing configuration as dictionary."""
    return {
        'language': DEFAULT_LANGUAGE,
        'min_token_length': MIN_TOKEN_LENGTH,
        'max_token_length': MAX_TOKEN_LENGTH,
        'remove_numbers': REMOVE_NUMBERS,
        'remove_urls': REMOVE_URLS,
        'aggressive': AGGRESSIVE_CLEANING,
        'custom_stopwords': CUSTOM_STOPWORDS
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

# ============================================================================
# INITIALIZATION
# ============================================================================

# Run validation on import (can be disabled if needed)
if __name__ != '__main__':
    try:
        validate_settings()
    except Exception as e:
        logger.warning(f"Settings validation skipped: {e}")


# Export commonly used settings
__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'VECTORIZER_SAVE_PATH',
    'MONGODBURL',
    'DEFAULT_HEADERS',
    'load_news_sources',
    'get_db_config',
    
    'get_preprocessing_config',
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
