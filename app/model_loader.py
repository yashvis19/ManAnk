"""Loads the trained pipeline and its metadata once, and caches them in memory."""

import functools
import json
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "mental_health_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


@functools.lru_cache(maxsize=1)
def get_model():
    """Load the trained sklearn pipeline once and cache it in memory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python training/train.py` first."
        )
    return joblib.load(MODEL_PATH)


@functools.lru_cache(maxsize=1)
def get_metadata() -> dict:
    """Load model_metadata.json once and cache it in memory."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. Run `python training/train.py` "
            "then `python training/evaluate.py` first."
        )
    return json.loads(METADATA_PATH.read_text())
