from fastapi import FastAPI,HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from schemas.request import SentimentRequest
from schemas.response import SentimentResponse,SentimentPrediction
from core.dependencies import get_model_service

from core.model_loader import ModelService

from loguru import logger
app=FastAPI(
    title="Twitter Sentiment Ananlyzer",
    description="API for uploading tweets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
     allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/predict", response_model=SentimentResponse, tags=["Inference"])
def predict(
    request: SentimentRequest,
    service: ModelService = Depends(get_model_service)
):
    try:
        model_name, labels, confidences = service.predict(
            texts=request.texts,
            classifier_type=request.classifier_type
        )
    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    predictions = [
        SentimentPrediction(
            text=text,
            label=labels[i],
            confidence=confidences[i] if confidences else None
        )
        for i, text in enumerate(request.texts)
    ]

    return SentimentResponse(
        model=model_name,
        predictions=predictions
    )
@app.get("/health",tags=["Health"])
def health_check():
    return {"status_code":200,"message":"Working Well"}

