"""Pydantic models: the shapes of data going in and out of the API.

Keeping these separate from routes means the exact same validation rules
can be reused by both the single-prediction and batch-prediction endpoints.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ml_core.preprocessing import TOP_COUNTRIES

PLATFORMS = Literal[
    "Facebook", "LinkedIn", "Instagram", "Snapchat",
    "Twitter", "YouTube", "TikTok", "LINE",
    "KakaoTalk", "VKontakte", "WhatsApp", "WeChat",
]

STRESS_LEVELS = Literal["Low", "Medium", "High", "Very High"]
ACADEMIC_LEVELS = Literal["Undergraduate", "Graduate", "High School"]
PURPOSES = Literal["Networking", "Education", "Entertainment", "News"]

# Any country is accepted — unrecognized ones are grouped into "Other" by the
# preprocessing step. This constant is exposed so the frontend can show
# which countries get their own category vs get grouped.
KNOWN_COUNTRIES = TOP_COUNTRIES


class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100, description="Student age in years")
    gender: Literal["Male", "Female"]
    country: str = Field(..., min_length=1, max_length=60)
    academic_level: ACADEMIC_LEVELS
    most_used_platform: PLATFORMS
    purpose_of_use: PURPOSES
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0, le=500)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: STRESS_LEVELS

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "age": 21,
                "gender": "Female",
                "country": "India",
                "academic_level": "Undergraduate",
                "most_used_platform": "Instagram",
                "purpose_of_use": "Entertainment",
                "avg_daily_usage_hours": 4.5,
                "daily_unlocks": 60,
                "study_hours": 3.0,
                "physical_activity_hours": 1.0,
                "sleep_hours_per_night": 6.5,
                "stress_level": "High",
            }
    })


class FeatureContribution(BaseModel):
    feature: str
    value: str
    impact: float = Field(..., description="Positive pushes score up, negative pushes it down")
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
    score_range: str = "0 (lowest) to 10 (highest)"
    top_factors: list[FeatureContribution]
    model_version: str
    disclaimer: str = (
        "This is a statistical estimate based on a research dataset, not a "
        "clinical or diagnostic assessment. If you're struggling, please talk "
        "to a mental health professional or a trusted person in your life."
    )


class BatchPredictionRow(BaseModel):
    row_index: int
    predicted_mental_health_score: Optional[float] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_rows: int
    successful: int
    failed: int
    results: list[BatchPredictionRow]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    test_r2: Optional[float] = None
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
