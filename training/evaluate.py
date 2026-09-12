"""
Evaluate the saved model on the untouched test set and update model_metadata.json
with the honest final numbers.

Usage:
    python training/evaluate.py

This loads the exact test-set row indices that train.py held out (from
models/test_split.json) rather than re-splitting, so we're 100% sure these
rows were never used for training or hyperparameter tuning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `python training/x.py` from project root

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_core.preprocessing import group_country, FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = Path(__file__).parent.parent / "data" / "Student Social Media And Mental Health Impact.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "mental_health_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SPLIT_PATH = MODEL_DIR / "test_split.json"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run training/train.py first.")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError("No saved test split found. Run training/train.py first.")

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    df["Grouped_country"] = df["Country"].apply(group_country)

    test_indices = json.loads(SPLIT_PATH.read_text())["test_indices"]
    test_df = df.loc[test_indices]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    print("Final test-set metrics (never seen during training or tuning):")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")

    metadata = json.loads(METADATA_PATH.read_text())
    metadata["test_metrics"] = {"r2": r2, "mae": mae, "rmse": rmse}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nUpdated {METADATA_PATH} with test_metrics.")


if __name__ == "__main__":
    main()
