# FastAPI backend. Build context is the repository root (see compose.yaml),
# so the image can include models/phresh_text_model.joblib.

# Same Python minor version as the environment that saved the model
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --only-binary=:all: -r backend/requirements.txt

# main.py loads the model from ../models relative to backend/
COPY backend/ backend/
COPY models/phresh_text_model.joblib models/phresh_text_model.joblib

RUN useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8000

# No --reload in production
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
