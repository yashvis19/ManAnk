import sys
from pathlib import Path

# Make the project root importable so `import app...` / `import ml_core...`
# work no matter where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
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
