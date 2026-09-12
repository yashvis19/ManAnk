"""
Train the final model and save it to models/mental_health_model.pkl,
along with models/model_metadata.json.

Usage:
    python training/train.py

This script assumes model_comparison.py has already been used to decide
which algorithm to use (Random Forest, based on the comparison in this
project — see models/model_comparison.json after running that script).
It re-splits the data the same way, tunes hyperparameters with
RandomizedSearchCV, and fits the final pipeline on the training set only.
The test set is untouched here — evaluate.py is what reports final,
honest numbers on data the model has never seen.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `python training/x.py` from project root

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from ml_core.preprocessing import (
    build_preprocessor,
    group_country,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "Student Social Media And Mental Health Impact.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "mental_health_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SPLIT_PATH = MODEL_DIR / "test_split.json"  # row indices held out, used by evaluate.py
BASELINES_PATH = MODEL_DIR / "feature_baselines.json"  # used by explainability at predict time

RANDOM_STATE = 42
MODEL_VERSION = "2.0.0"


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    df["Grouped_country"] = df["Country"].apply(group_country)
    return df


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_and_clean_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Keep the same split as evaluate.py by fixing random_state and saving
    # the exact test-set indices, so evaluate.py can reload the *same*
    # untouched rows rather than trusting that re-splitting gives identical results.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    SPLIT_PATH.write_text(json.dumps({"test_indices": X_test.index.tolist()}))

    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [None, 8, 12, 16, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=25,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV R2:", search.best_score_)

    joblib.dump(best_pipeline, MODEL_PATH, compress=3)

    # Save typical ("baseline") values for each feature from the training set.
    # explainability.py uses these at prediction time: it swaps one feature at
    # a time for its baseline and re-predicts, so the drop/rise in score shows
    # how much that feature actually pulled the prediction away from "typical".
    baselines = {}
    for col in FEATURE_COLUMNS:
        if X_train[col].dtype.kind in "if":  # numeric
            baselines[col] = float(X_train[col].mean())
        else:  # categorical
            baselines[col] = X_train[col].mode().iloc[0]
    BASELINES_PATH.write_text(json.dumps(baselines, indent=2))

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "RandomForestRegressor",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "best_params": search.best_params_,
        "cv_best_r2": search.best_score_,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "notes": "Final held-out test metrics are written to model_metadata.json "
                 "by evaluate.py after this script runs. cv_best_r2 above is "
                 "cross-validated training performance, not a test-set score.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print("\nNext: run `python training/evaluate.py` to get honest test-set metrics.")


if __name__ == "__main__":
    main()
