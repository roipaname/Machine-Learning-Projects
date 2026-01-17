import pickle
from config.settings import MODELS_DIR,VECTORIZER_SAVE_PATH,MODEL_SAVE_PATH
from src.models.classifier import SentimentAnalyzer
from src.features.tfidf_vectorizer import TFIDFVectorizer
from pathlib import Path
from loguru import logger
from typing import Optional,Dict,Tuple
from contextlib import contextmanager

from functools import lru_cache
class ModelService:
    def __init__(self):
        self._vectorizer: Optional[TFIDFVectorizer] = None
        self._models: Dict[str, SentimentAnalyzer] = {}
        self._active_model: Optional[str] = None

    # -------------------------
    # Context manager
    # -------------------------
    @contextmanager
    def use_model(self, classifier_type: Optional[str] = None):
        """
        Context manager to load & activate a model safely.
        """
        model = self._get_or_load_model(classifier_type)
        self._active_model = model.classifier_type
        try:
            yield model
        finally:
            self._active_model = None

    # -------------------------
    # Load vectorizer once
    # -------------------------
    def _load_vectorizer(self):
        if self._vectorizer is None:
            logger.info("Loading TF-IDF vectorizer...")
            self._vectorizer = TFIDFVectorizer.load(VECTORIZER_SAVE_PATH)
            logger.success("Vectorizer loaded")

    # -------------------------
    # Load or reuse model
    # -------------------------
    def _get_or_load_model(self, classifier_type: Optional[str]) -> SentimentAnalyzer:
        self._load_vectorizer()

        # Determine model key + path
        if classifier_type:
            model_key = classifier_type
            model_path = Path(MODELS_DIR) / f"{classifier_type}.pkl"
        else:
            model_path = Path(MODEL_SAVE_PATH)
            model_key = model_path.stem

        # Return cached model if available
        if model_key in self._models:
            logger.debug(f"Using cached model: {model_key}")
            return self._models[model_key]

        logger.info(f"Loading model from disk: {model_path}")
        model = SentimentAnalyzer.load(model_path)

        self._models[model_key] = model
        logger.success(f"Model '{model_key}' cached in memory")

        return model
    @lru_cache(2048)
    def _cached_predict(
        self,
        texts: Tuple[str, ...],
        classifier_type: Optional[str]
    ):
        
        with self.use_model(classifier_type) as model:
            X=self._vectorizer.transform(list(texts))
            preds = tuple(model.predict(X))

        probs = None
        if hasattr(model, "predict_proba"):
             probs = tuple(model.predict_proba(X).max(axis=1))
        return preds, probs




    # -------------------------
    # Prediction (cached-ready)
    # -------------------------
    def predict(self, texts: list[str], classifier_type: Optional[str] = None):
        texts_key = tuple(texts)
        preds, probs = self._cached_predict(texts_key, classifier_type)

        return list(preds), list(probs) if probs else None



model_service = ModelService()