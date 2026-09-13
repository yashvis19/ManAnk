#ManAnk- Mental Health Score Prediction System

An end-to-end machine learning application that predicts a student's mental
health score (0-10) from academic, lifestyle, and social-media-usage habits.
Built as a Random Forest regression pipeline served through FastAPI, with
per-prediction explainability, batch (CSV) inference, automated tests, and
Docker deployment.

**This is not a clinical or diagnostic tool.** It is a statistical estimate
based on a survey dataset, intended for informational and educational use only.

---

## Table of contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML methodology](#ml-methodology)
- [Evaluation results](#evaluation-results)
- [Project structure](#project-structure)
- [Setup instructions](#setup-instructions)
- [Retraining the model](#retraining-the-model)
- [API usage](#api-usage)
- [Running tests](#running-tests)
- [Docker](#docker)
- [Limitations](#limitations)
- [Future scope](#future-scope)

---

## Architecture

```
┌──────────────┐      HTTP/JSON       ┌───────────────────┐      joblib      ┌──────────────────┐
│   Frontend   │ ───────────────────► │   FastAPI backend  │ ───────────────► │  Trained sklearn  │
│ (HTML/CSS/JS)│ ◄─────────────────── │  (app/)             │ ◄─────────────── │  Pipeline (.pkl)  │
└──────────────┘                      └───────────────────┘                  └──────────────────┘
                                               │
                                               ▼
                                     explainability.py
                                  (per-prediction feature
                                    contribution via ablation)
```

- **Training** (`training/`) is a set of scripts, not a notebook — running
  them always reproduces the same model from the same data.
- **`ml_core/`** holds the preprocessing logic shared by both training and
  the live API, so the two can never quietly drift out of sync.
- **`app/`** is the FastAPI backend, split into schemas, model loading,
  prediction logic, explainability, and routes.
- **`frontend/`** is a plain HTML/CSS/JS single-page app — no build step.

## Dataset

[Student Social Media & Mental Health dataset](data/Student%20Social%20Media%20And%20Mental%20Health%20Impact.csv) —
5,000 rows, 13 columns, no missing values, no duplicate rows. Each row is one
student's self-reported age, gender, country, academic level, most-used
platform, purpose of use, daily usage hours, daily phone unlocks, study
hours, physical activity hours, sleep hours, stress level, and a
`Mental_Health_Score` target (0-10).

One known data-quality issue was handled during cleaning: a small number of
`Physical_Activity_Hours` values were negative (a plausible source of
false/typo entries), and were clipped to zero rather than dropping the row,
to preserve the rest of that student's otherwise-valid data.

`Country` has 111 distinct raw values. It's grouped into the 10 most common
countries plus an `Other` bucket (`Grouped_country`), which is the feature
the model actually trains on — this keeps one-hot encoding from creating
over a hundred mostly-empty columns.

## ML methodology

1. **Baseline comparison** (`training/model_comparison.py`): Linear
   Regression, Random Forest, and Gradient Boosting are each evaluated with
   **5-fold cross-validation** on the training split only, scored on R²,
   MAE, and RMSE. This picks the model family before any hyperparameter
   tuning happens, so the comparison isn't biased by tuning one model and
   not the others.
2. **Hyperparameter tuning** (`training/train.py`): the winning model
   (Random Forest) is tuned with `RandomizedSearchCV` (25 candidate
   configurations, 5-fold CV) over tree count, depth, split/leaf sizes, and
   feature sampling strategy.
3. **Final evaluation** (`training/evaluate.py`): the tuned model is scored
   **once**, on a 20% test split that was held out before tuning ever
   started and never touched again until this step. These are the numbers
   reported below.
4. **Preprocessing**, applied inside the sklearn `Pipeline` (so it can never
   be applied inconsistently between training and serving):
   - `Study_Hours`: log1p transform (it's right-skewed) then standard scaling
   - other numeric columns: standard scaling
   - `Stress_Level`: ordinal encoding (`Low < Medium < High < Very High`)
   - remaining categoricals: one-hot encoding

## Evaluation results

Cross-validated comparison (training split, 5-fold CV):

| Model              | R²     | MAE    | RMSE   |
|--------------------|--------|--------|--------|
| **Random Forest**  | 0.867  | 0.340  | 0.461  |
| Gradient Boosting  | 0.790  | 0.449  | 0.579  |
| Linear Regression  | 0.721  | 0.525  | 0.667  |

Final tuned Random Forest, scored once on the untouched test set:

| Metric | Value |
|--------|-------|
| R²     | 0.901 |
| MAE    | 0.309 |
| RMSE   | 0.421 |

On average, predictions are within about **0.31 points** (on a 0-10 scale)
of the actual reported score. Full metrics, best hyperparameters, and
training metadata are in
[`models/model_metadata.json`](models/model_metadata.json), regenerated
every time `train.py` and `evaluate.py` are run.

## Project structure

```
mental-health-score/
├── data/                       dataset + a sample CSV for batch testing
├── ml_core/                    preprocessing shared by training and the API
├── training/                   train.py, evaluate.py, model_comparison.py
├── models/                     saved model, metadata, feature baselines
├── app/                        FastAPI backend
│   ├── main.py                 app setup, CORS, router wiring
│   ├── schemas.py               request/response models + validation rules
│   ├── model_loader.py          cached model + metadata loading
│   ├── preprocessing.py         request -> model-input DataFrame
│   ├── prediction_service.py    core predict-and-explain logic
│   ├── explainability.py        per-prediction feature contributions
│   └── routes/                  health, predict, batch_predict endpoints
├── tests/                      pytest suite (28 tests)
├── frontend/                   static HTML/CSS/JS UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup instructions

Requires Python 3.11+.

```bash
git clone <your-repo-url>
cd mental-health-score
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The trained model is already included in `models/`, so you can run the API
immediately:

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs (Swagger UI).

To run the frontend, open `frontend/index.html` directly in a browser, or
serve it:

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://127.0.0.1:5500`. The frontend's `API_BASE` in
`script.js` defaults to `http://127.0.0.1:8000` — update it if your backend
is deployed elsewhere.

## Retraining the model

Run these from the project root, in order:

```bash
python training/model_comparison.py   # compare candidate models (prints a table)
python training/train.py              # tune + fit the final model, save it + metadata
python training/evaluate.py           # score the final model on the untouched test set
```

This regenerates `models/mental_health_model.pkl`, `model_metadata.json`,
`feature_baselines.json`, and `test_split.json`.

## API usage

**`POST /predict`** — single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 21, "gender": "Female", "country": "India",
    "academic_level": "Undergraduate", "most_used_platform": "Instagram",
    "purpose_of_use": "Entertainment", "avg_daily_usage_hours": 4.5,
    "daily_unlocks": 60, "study_hours": 3.0, "physical_activity_hours": 1.0,
    "sleep_hours_per_night": 6.5, "stress_level": "High"
  }'
```

Response includes the score, the top 5 factors that moved it (with
direction and magnitude), the model version, and a disclaimer.

**`POST /predict/batch`** — batch prediction from a CSV upload

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -F "file=@data/sample_batch_students.csv"
```

Required CSV columns (case-sensitive): `age, gender, country,
academic_level, most_used_platform, purpose_of_use, avg_daily_usage_hours,
daily_unlocks, study_hours, physical_activity_hours,
sleep_hours_per_night, stress_level`. Rows that fail validation are reported
individually rather than failing the whole batch.

**`GET /health`** — readiness check, also reports model type/version and
test-set accuracy.

## Running tests

```bash
pytest tests/ -v
```

28 tests covering API endpoints, input validation (range and category
checks), and model/pipeline loading.

## Docker

```bash
docker compose up --build
```

This builds the backend image (copies `app/`, `ml_core/`, and `models/`
only — not the dataset or training scripts) and serves the frontend via
nginx. Backend: `http://localhost:8000`. Frontend: `http://localhost:5500`.

To build/run just the backend:

```bash
docker build -t mental-health-api .
docker run -p 8000:8000 mental-health-api
```

## Limitations

- Trained on a single survey-style dataset (5,000 self-reported rows); it
  reflects patterns in that data, not a validated psychological instrument.
- No demographic fairness or bias auditing has been performed across
  gender, country, or academic-level subgroups.
- The explainability method (feature ablation against a "typical student"
  baseline) shows correlation-based contribution within this model, not a
  causal claim about what affects mental health.
- Not validated against any clinical mental-health assessment (e.g. PHQ-9,
  GAD-7) — the 0-10 score is specific to this dataset's own labeling.


