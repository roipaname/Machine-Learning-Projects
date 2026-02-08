import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.classifier import ChurnPredictor
from config.settings import DATA_PROCESSED_DIR

data_parquet_path=DATA_PROCESSED_DIR /'training_features.parquet'

def train_model(data_path):
    #load dataset
    df=pd.read_parquet(data_path)
