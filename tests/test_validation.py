"""Tests that bad input is rejected with a 422 before it ever reaches the model."""

import pytest


@pytest.mark.parametrize("field,bad_value", [
    ("age", -5),
    ("age", 200),
    ("avg_daily_usage_hours", -1),
    ("avg_daily_usage_hours", 30),
    ("daily_unlocks", -10),
    ("study_hours", 25),
    ("physical_activity_hours", -2),
    ("sleep_hours_per_night", 25),
])
def test_out_of_range_numeric_fields_rejected(client, valid_payload, field, bad_value):
    payload = dict(valid_payload)
    payload[field] = bad_value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


@pytest.mark.parametrize("field,bad_value", [
    ("gender", "Other"),  # only Male/Female accepted by this dataset's schema
    ("stress_level", "Extreme"),
    ("academic_level", "PhD"),
    ("most_used_platform", "MySpace"),
    ("purpose_of_use", "Shopping"),
])
def test_invalid_categorical_values_rejected(client, valid_payload, field, bad_value):
    payload = dict(valid_payload)
    payload[field] = bad_value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_missing_required_field_rejected(client, valid_payload):
    payload = dict(valid_payload)
    del payload["stress_level"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_unknown_country_still_accepted(client, valid_payload):
    """Countries outside the top-10 list are valid - they just get grouped as 'Other'."""
    payload = dict(valid_payload)
    payload["country"] = "Narnia"
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
