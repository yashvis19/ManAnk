"""Turns a validated StudentData request into the exact DataFrame row the
trained pipeline expects. Uses the same FEATURE_COLUMNS / group_country
logic as training, imported from ml_core, so the API can never quietly
drift out of sync with what the model was trained on.
"""

import pandas as pd

from ml_core.preprocessing import FEATURE_COLUMNS, group_country
from app.schemas import StudentData


def to_model_input(data: StudentData) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact column order the model expects."""
    country_group = group_country(data.country)

    row = {
        "Study_Hours": data.study_hours,
        "Age": data.age,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
        "Gender": data.gender,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Grouped_country": country_group,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
