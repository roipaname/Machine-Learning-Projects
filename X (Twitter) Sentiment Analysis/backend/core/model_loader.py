from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Tuple, List
from contextlib import contextmanager
from functools import lru_cache

from config.settings import MODELS_DIR, VECTORIZER_SAVE_PATH, MODEL_SAVE_PATH
from src.models.classifier import SentimentAnalyzer
from src.features.tfidf_vectorizer import TFIDFVectorizer


class ModelService:
    def __init__(self):
        self._vectorizer: Optional[TFIDFVectorizer] = None
        self._models: Dict[str, SentimentAnalyzer] = {}

    # -------------------------
    # Context manager
    # -------------------------
    @contextmanager
    def use_model(self, classifier_type: Optional[str] = None):
        model = self._get_or_load_model(classifier_type)
        try:
            yield model
        finally:
            pass  # placeholder for future cleanup

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

        if classifier_type:
            model_key = classifier_type
            model_path = Path(MODELS_DIR) / f"{classifier_type}_model.pkl"
        else:
            model_path = Path(MODEL_SAVE_PATH)
            model_key = model_path.stem

        if model_key in self._models:
            logger.debug(f"Using cached model: {model_key}")
            return self._models[model_key]

        logger.info(f"Loading model from disk: {model_path}")
        model = SentimentAnalyzer.load(model_path)

        self._models[model_key] = model
        logger.success(f"Model '{model_key}' cached in memory")

        return model

    # -------------------------
    # Cached prediction core
    # -------------------------
    @staticmethod
    @lru_cache(maxsize=2048)
    def _predict_cached(
        texts: Tuple[str, ...],
        classifier_type: Optional[str],
        vectorizer_id: int,
        service_ref: "ModelService"
    ):
        with service_ref.use_model(classifier_type) as model:
            X = service_ref._vectorizer.transform(list(texts))
            preds = tuple(model.predict(X))

            probs = None
            if hasattr(model, "predict_proba"):
                probs = tuple(model.predict_proba(X).max(axis=1))

        return model.classifier_type, preds, probs

    # -------------------------
    # Public prediction API
    # -------------------------
    def predict(
        self,
        texts: List[str],
        classifier_type: Optional[str] = None
    ) -> Tuple[str, List[str], Optional[List[float]]]:

        if not texts:
            raise ValueError("Input texts cannot be empty")

        self._load_vectorizer()

        model_name, preds, probs = self._predict_cached(
            texts=tuple(texts),
            classifier_type=classifier_type,
            vectorizer_id=id(self._vectorizer),
            service_ref=self
        )

        return model_name, list(preds), list(probs) if probs else None


# -------------------------
# FastAPI singleton
# -------------------------
@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    logger.info("Initializing ModelService singleton")
    return ModelService()
