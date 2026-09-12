"""
Shared preprocessing definitions.

This is the SINGLE SOURCE OF TRUTH for how raw student data is turned into
model-ready features. Both training (train.py) and the live API
(app/preprocessing.py) import from here so the two can never drift apart.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
import numpy as np

# ── Column groups ────────────────────────────────────────────────
# Study_Hours is right-skewed, so it gets a log transform before scaling.
SKEWED_NUMERIC_COL = ["Study_Hours"]

# Other numeric columns just get standard scaling.
OTHER_NUMERIC_COLS = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]

# Stress level has a natural order, so it's ordinal-encoded (not one-hot).
ORDINAL_COL = ["Stress_Level"]
STRESS_ORDER = [["Low", "Medium", "High", "Very High"]]

# These have no natural order, so they're one-hot encoded.
ONE_HOT_COLS = [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]

# The full, ordered list of columns the model expects as input.
# The API must build its input DataFrame with exactly these columns.
FEATURE_COLUMNS = (
    SKEWED_NUMERIC_COL + OTHER_NUMERIC_COLS + ORDINAL_COL + ONE_HOT_COLS
)

TARGET_COLUMN = "Mental_Health_Score"

# The top countries kept as their own category; everything else becomes "Other".
# Frozen here (not recomputed at predict time) so the API's behavior can never
# silently change if the training data changes.
TOP_COUNTRIES = [
    "USA", "India", "UK", "Canada", "Germany",
    "Australia", "Mexico", "Turkey", "France", "Brazil",
]


def group_country(country: str) -> str:
    """Map a raw country string to one of the top-10 countries, else 'Other'."""
    return country if country in TOP_COUNTRIES else "Other"


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used inside the model pipeline.

    - Study_Hours: log1p transform, then standard scaling
    - other numeric columns: standard scaling
    - Stress_Level: ordinal encoding (Low < Medium < High < Very High)
    - remaining categoricals: one-hot encoding
    """
    skewed_pipeline = Pipeline(steps=[
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("skewed", skewed_pipeline, SKEWED_NUMERIC_COL),
        ("numeric", StandardScaler(), OTHER_NUMERIC_COLS),
        ("ordinal", OrdinalEncoder(categories=STRESS_ORDER), ORDINAL_COL),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLS),
    ])

    return preprocessor
