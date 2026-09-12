"""
Explainable AI for individual predictions.

Method: leave-one-feature-out ablation against a "typical student" baseline.
For each feature, we take the input row, swap JUST that one feature for its
baseline value (the training-set mean for numbers, the most common value for
categories), and re-run the model. The difference between the original
prediction and this "what if this feature were typical" prediction tells us
how much that feature is pulling the score up or down relative to a typical
student.

This is model-agnostic, has no extra dependencies, and is easy to explain in
a README (unlike SHAP's Shapley-value math). It answers a very concrete
question: "how much did THIS feature, at THIS value, move the score away
from what a typical student would get?"
"""

import functools
import json
from pathlib import Path

import pandas as pd

from ml_core.preprocessing import FEATURE_COLUMNS

BASELINES_PATH = Path(__file__).parent.parent / "models" / "feature_baselines.json"

# Human-readable labels for the API response.
FEATURE_LABELS = {
    "Study_Hours": "Study hours",
    "Age": "Age",
    "Avg_Daily_Usage_Hours": "Social media usage hours",
    "Daily_Unlocks": "Daily phone unlocks",
    "Physical_Activity_Hours": "Physical activity hours",
    "Sleep_Hours_Per_Night": "Sleep hours",
    "Stress_Level": "Stress level",
    "Gender": "Gender",
    "Academic_Level": "Academic level",
    "Most_Used_Platform": "Most used platform",
    "Purpose_Of_Use": "Purpose of social media use",
    "Grouped_country": "Country group",
}


@functools.lru_cache(maxsize=1)
def get_baselines() -> dict:
    if not BASELINES_PATH.exists():
        raise FileNotFoundError(
            f"Baselines not found at {BASELINES_PATH}. Run `python training/train.py` first."
        )
    return json.loads(BASELINES_PATH.read_text())


def explain_prediction(model, input_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Return the top_n features that most moved this prediction, with direction.

    `input_row` must be a single-row DataFrame already in the model's expected
    feature order/format (see app/preprocessing.py::to_model_input).
    """
    baselines = get_baselines()
    original_prediction = float(model.predict(input_row)[0])

    original_values = input_row.iloc[0].to_dict()

    contributions = []
    for feature in FEATURE_COLUMNS:
        ablated_values = dict(original_values)
        ablated_values[feature] = baselines[feature]
        # Rebuild as a fresh DataFrame (rather than mutating a column in
        # place) so pandas doesn't try to force a float baseline into a
        # column whose dtype was inferred as int from the original value.
        ablated_row = pd.DataFrame([ablated_values], columns=FEATURE_COLUMNS)
        ablated_prediction = float(model.predict(ablated_row)[0])

        # If swapping this feature for "typical" changes the score, this
        # feature's actual value was responsible for that change.
        impact = original_prediction - ablated_prediction

        contributions.append({
            "feature": FEATURE_LABELS.get(feature, feature),
            "value": str(input_row.iloc[0][feature]),
            "impact": round(impact, 4),
            "direction": "increases" if impact >= 0 else "decreases",
        })

    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions[:top_n]
