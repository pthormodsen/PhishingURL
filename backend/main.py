from pathlib import Path
from urllib.parse import urlsplit

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.features import url_to_model_text

app = FastAPI()

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "phresh_text_model.joblib"
)

model = joblib.load(model_path)
phishing_column = list(model.classes_).index(0)
THRESHOLD = 0.5


class PredictionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


@app.post("/predict")
def predict(request: PredictionRequest):
    url = request.url.strip()

    if "://" not in url:
        url = "https://" + url

    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname is not None
            and not any(character.isspace() for character in url)
        )
    except ValueError:
        valid = False

    if not valid:
        raise HTTPException(
            status_code=400,
            detail="Enter a valid HTTP or HTTPS URL without spaces.",
        )

    text = url_to_model_text(url)
    score = float(
        model.predict_proba([text])[0, phishing_column]
    )

    prediction = 0 if score >= THRESHOLD else 1

    return {
        "prediction": prediction,
        "label": (
            "Potential phishing"
            if prediction == 0
            else "Not flagged by the model"
        ),
        "phishing_score": score,
        # The text the model actually saw, after preprocessing
        "model_input": text,
    }