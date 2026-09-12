"""Tests for model loading and the prediction service, independent of FastAPI."""

from app.model_loader import get_metadata, get_model
from app.preprocessing import to_model_input
from app.prediction_service import predict_single
from app.schemas import StudentData
from ml_core.preprocessing import FEATURE_COLUMNS


def test_model_loads():
    model = get_model()
    assert model is not None
    assert hasattr(model, "predict")


def test_metadata_has_expected_keys():
    metadata = get_metadata()
    assert metadata["model_type"] == "RandomForestRegressor"
    assert "test_metrics" in metadata
    assert set(metadata["feature_columns"]) == set(FEATURE_COLUMNS)


def test_to_model_input_has_correct_columns(valid_payload):
    student = StudentData(**valid_payload)
    row = to_model_input(student)
    assert list(row.columns) == FEATURE_COLUMNS
    assert len(row) == 1


def test_predict_single_returns_score_in_range(valid_payload):
    student = StudentData(**valid_payload)
    result = predict_single(student)
    assert 0 <= result.predicted_mental_health_score <= 10
    assert len(result.top_factors) == 5


def test_grouped_country_applied_for_unknown_country(valid_payload):
    payload = dict(valid_payload)
    payload["country"] = "Atlantis"
    student = StudentData(**payload)
    row = to_model_input(student)
    assert row.iloc[0]["Grouped_country"] == "Other"
