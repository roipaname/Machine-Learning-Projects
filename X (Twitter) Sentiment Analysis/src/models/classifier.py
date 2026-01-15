from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support
import numpy as np
from loguru import logger
import pickle
from pathlib import Path
from datetime import datetime

from config.settings import (
    DEFAULT_CLASSIFIER,
    MODEL_SAVE_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS
)