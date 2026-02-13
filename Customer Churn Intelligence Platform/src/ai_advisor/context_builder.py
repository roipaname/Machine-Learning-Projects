import pandas as pd
from typing import List,Optional,Dict
from database.operations import get_customers_by_account,get_priority_tickets
from src.features.feature_eng import extract_customer_features
import joblib
from loguru import logger
from config.settings import MODELS_DIR
from src.models.classifier import ChurnPredictor
class CustomerContextBuilder:
    def __init__(self, model_type:str="ranndom_forest"):
        self.model_type=model_type
        self.model_path=MODELS_DIR / f"{model_type}_model.joblib"
        if not self.model_path.exists():
            logger.warning(f"Model file {self.model_path} not found. Context builder will not be able to generate predictions.")
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        self.model=ChurnPredictor.load_model(self.model_path)
        self.feature_importance=pd.read_csv(MODELS_DIR / f"feature_importance.csv")
        