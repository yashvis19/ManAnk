"""Tests for the API endpoints: health, single prediction, batch prediction."""

import io


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "RandomForestRegressor"
    # Test-set metrics should be present since evaluate.py has been run.
    assert body["test_r2"] is not None
    assert 0 <= body["test_r2"] <= 1


def test_predict_valid_payload(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_mental_health_score"] <= 10
    assert len(body["top_factors"]) == 5
    assert "disclaimer" in body


def test_predict_top_factors_have_direction(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    body = r.json()
    for factor in body["top_factors"]:
        assert factor["direction"] in ("increases", "decreases")
        assert "feature" in factor
        assert "value" in factor


def test_batch_predict_valid_csv(client):
    csv_content = (
        "age,gender,country,academic_level,most_used_platform,purpose_of_use,"
        "avg_daily_usage_hours,daily_unlocks,study_hours,physical_activity_hours,"
        "sleep_hours_per_night,stress_level\n"
        "21,Male,India,Undergraduate,Instagram,Entertainment,4.0,100,3.0,1.5,7.0,Medium\n"
        "22,Female,USA,Graduate,LinkedIn,Education,2.0,50,5.0,2.0,8.0,Low\n"
    )
    file = io.BytesIO(csv_content.encode())
    r = client.post("/predict/batch", files={"file": ("students.csv", file, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == 2
    assert body["successful"] == 2
    assert body["failed"] == 0


def test_batch_predict_rejects_non_csv(client):
    file = io.BytesIO(b"not a csv")
    r = client.post("/predict/batch", files={"file": ("students.txt", file, "text/plain")})
    assert r.status_code == 400


def test_batch_predict_missing_columns(client):
    csv_content = "age,gender\n21,Male\n"
    file = io.BytesIO(csv_content.encode())
    r = client.post("/predict/batch", files={"file": ("students.csv", file, "text/csv")})
    assert r.status_code == 400
    assert "missing" in r.json()["detail"].lower()


def test_batch_predict_reports_row_level_errors(client):
    """One good row, one row with an invalid stress level - batch should not fail entirely."""
    csv_content = (
        "age,gender,country,academic_level,most_used_platform,purpose_of_use,"
        "avg_daily_usage_hours,daily_unlocks,study_hours,physical_activity_hours,"
        "sleep_hours_per_night,stress_level\n"
        "21,Male,India,Undergraduate,Instagram,Entertainment,4.0,100,3.0,1.5,7.0,Medium\n"
        "22,Female,USA,Graduate,LinkedIn,Education,2.0,50,5.0,2.0,8.0,Extreme\n"
    )
    file = io.BytesIO(csv_content.encode())
    r = client.post("/predict/batch", files={"file": ("students.csv", file, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["successful"] == 1
    assert body["failed"] == 1
