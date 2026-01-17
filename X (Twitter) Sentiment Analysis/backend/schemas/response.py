from pydantic import BaseModel
from typing import List, Optional


class SentimentPrediction(BaseModel):
    text: str
    label: str
    confidence: Optional[float]


class SentimentResponse(BaseModel):
    model: str
    predictions: List[SentimentPrediction]
