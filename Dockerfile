# Backend image: FastAPI + trained model, served with uvicorn.
FROM python:3.12-slim

WORKDIR /code

# Install dependencies first so Docker can cache this layer when only
# application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running API actually needs (not data/, training/, tests/).
COPY app/ ./app/
COPY ml_core/ ./ml_core/
COPY models/ ./models/

EXPOSE 8000

# Basic container-level health check hitting the API's own /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
