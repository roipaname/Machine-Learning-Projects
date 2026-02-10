from pathlib import Path
import joblib
from config.settings import MODEL_DIR



def load_model(model_name:str):
    model_path=MODEL_DIR / f"model_{model_name}.joblib"
    model_data=joblib.load(model_path)
    