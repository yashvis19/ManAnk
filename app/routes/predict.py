from fastapi import APIRouter, HTTPException

from app.prediction_service import predict_single
from app.schemas import PredictionResponse, StudentData

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(data: StudentData):
    """Predict a single student's mental health score, with top contributing factors."""
    try:
        return predict_single(data)
    except FileNotFoundError as e:
        # Model/metadata missing - a deployment problem, not a bad request.
        raise HTTPException(status_code=503, detail=str(e))
