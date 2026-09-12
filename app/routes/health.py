from fastapi import APIRouter

from app.model_loader import get_metadata, get_model
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/", tags=["health"])
def root():
    return {"message": "Mental Health Score Prediction API", "docs": "/docs"}


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """Readiness check: confirms the model is loaded and reports its accuracy."""
    get_model()  # raises if the model can't load
    metadata = get_metadata()
    test_metrics = metadata.get("test_metrics", {})
    return HealthResponse(
        status="ok",
        model_version=metadata.get("model_version", "unknown"),
        model_type=metadata.get("model_type", "unknown"),
        test_r2=test_metrics.get("r2"),
        test_mae=test_metrics.get("mae"),
        test_rmse=test_metrics.get("rmse"),
    )
