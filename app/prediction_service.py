"""Core prediction logic, kept separate from the FastAPI route layer so it
can be unit-tested directly without spinning up a server.
"""

from app.explainability import explain_prediction
from app.model_loader import get_metadata, get_model
from app.preprocessing import to_model_input
from app.schemas import PredictionResponse, StudentData, FeatureContribution


def predict_single(data: StudentData) -> PredictionResponse:
    model = get_model()
    metadata = get_metadata()

    input_row = to_model_input(data)
    raw_prediction = float(model.predict(input_row)[0])
    # Scores are conceptually 0-10; clip defensively in case of extreme inputs.
    score = round(max(0.0, min(10.0, raw_prediction)), 2)

    top_factors = explain_prediction(model, input_row, top_n=5)

    return PredictionResponse(
        predicted_mental_health_score=score,
        top_factors=[FeatureContribution(**f) for f in top_factors],
        model_version=metadata.get("model_version", "unknown"),
    )
