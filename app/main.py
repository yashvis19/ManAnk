"""FastAPI app entrypoint. Kept intentionally small: it just configures the
app and plugs in the route modules. All actual logic lives in
prediction_service.py, explainability.py, and the routes/ package.

Run locally with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import batch_predict, health, predict

app = FastAPI(
    title="ManAnk API",
    description=(
        "Predicts a student's mental health score (0-10) from academic, "
        "lifestyle, and social-media-usage factors, using a cross-validated "
        "Random Forest model. Includes per-prediction explainability and "
        "batch (CSV) prediction. Not a clinical or diagnostic tool."
    ),
    version="2.0.0",
)
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch_predict.router)
