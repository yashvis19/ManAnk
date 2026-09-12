import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.prediction_service import predict_single
from app.schemas import BatchPredictionResponse, BatchPredictionRow, StudentData

router = APIRouter()

MAX_BATCH_ROWS = 500

# CSV columns must match StudentData's field names exactly.
REQUIRED_COLUMNS = list(StudentData.model_fields.keys())


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """Upload a CSV of students and get a prediction for each row.

    Required columns (case-sensitive): age, gender, country, academic_level,
    most_used_platform, purpose_of_use, avg_daily_usage_hours, daily_unlocks,
    study_hours, physical_activity_hours, sleep_hours_per_night, stress_level.

    Rows that fail validation are reported individually with an error message
    rather than failing the whole batch.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(df)} rows; batch limit is {MAX_BATCH_ROWS}.",
        )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing_cols}",
        )

    results = []
    successful = 0
    for idx, row in df.iterrows():
        try:
            student = StudentData(**row[REQUIRED_COLUMNS].to_dict())
            prediction = predict_single(student)
            results.append(BatchPredictionRow(
                row_index=int(idx),
                predicted_mental_health_score=prediction.predicted_mental_health_score,
            ))
            successful += 1
        except ValidationError as e:
            results.append(BatchPredictionRow(
                row_index=int(idx),
                error=f"Validation error: {e.errors()[0]['msg']}",
            ))
        except Exception as e:
            results.append(BatchPredictionRow(row_index=int(idx), error=str(e)))

    return BatchPredictionResponse(
        total_rows=len(df),
        successful=successful,
        failed=len(df) - successful,
        results=results,
    )
