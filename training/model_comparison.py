"""
Compare candidate models using cross-validation on the TRAINING split only.

Run directly to print a comparison table:
    python training/model_comparison.py

This never touches the held-out test set. That set is reserved for
evaluate.py, after a model has already been chosen here.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `python training/x.py` from project root

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from ml_core.preprocessing import build_preprocessor, group_country, FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = Path(__file__).parent.parent / "data" / "Student Social Media And Mental Health Impact.csv"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    # Physical activity hours can't be negative; clip rather than drop the row
    # so we keep the rest of that student's otherwise-valid data.
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    df["Grouped_country"] = df["Country"].apply(group_country)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def get_candidate_models():
    """Models to compare. Add/remove entries here to extend the comparison."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def compare_models(X_train, y_train, cv_folds: int = 5) -> dict:
    """Run k-fold cross-validation for each candidate, return metric summary."""
    preprocessor = build_preprocessor()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    results = {}
    for name, model in get_candidate_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1
        )
        results[name] = {
            "r2_mean": float(np.mean(scores["test_r2"])),
            "r2_std": float(np.std(scores["test_r2"])),
            "mae_mean": float(-np.mean(scores["test_mae"])),
            "rmse_mean": float(-np.mean(scores["test_rmse"])),
        }
    return results


def main():
    X, y = load_data()
    # Same split logic as train.py, so the comparison reflects the real
    # training data the final model will be trained on.
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = compare_models(X_train, y_train)

    print(f"{'Model':<20}{'R2':>10}{'MAE':>10}{'RMSE':>10}")
    print("-" * 50)
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["r2_mean"]):
        print(f"{name:<20}{m['r2_mean']:>10.4f}{m['mae_mean']:>10.4f}{m['rmse_mean']:>10.4f}")

    out_path = Path(__file__).parent.parent / "models" / "model_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved comparison results to {out_path}")


if __name__ == "__main__":
    main()
