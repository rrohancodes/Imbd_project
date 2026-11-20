import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==================================
# LOAD MODELS
# ==================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

classification_model = joblib.load(
    MODEL_DIR / "classification_model.joblib"
)

regression_model = joblib.load(
    MODEL_DIR / "regression_model.joblib"
)

metadata = joblib.load(
    MODEL_DIR / "model_metadata.joblib"
)

threshold = metadata["threshold"]


# ==================================
# INITIALIZE FASTAPI
# ==================================

app = FastAPI(
    title="IMDb Prediction API",
    description="Predict IMDb rating and rating category",
    version="1.0"
)


# ==================================
# CORS CONFIGURATION
# ==================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================
# REQUEST MODEL
# ==================================

class MovieInput(BaseModel):

    Meta_score: float = Field(ge=0, le=100)
    No_of_Votes: int = Field(gt=0)
    Gross: float = Field(ge=0)
    Runtime: int = Field(gt=0)
    Released_Year: int

    Genre: str
    Certificate: str
    Director: str
    Overview: str


# ==================================
# PREPARE INPUT
# ==================================

def prepare_input(movie: MovieInput):

    data = movie.model_dump()

    data["No_of_Votes"] = np.log1p(data["No_of_Votes"])
    data["Gross"] = np.log1p(data["Gross"])

    return pd.DataFrame([data])


# ==================================
# HOME ROUTE
# ==================================

@app.get("/")
def home():

    return {
        "message": "IMDb Prediction API is running!"
    }


# ==================================
# PREDICTION ROUTE
# ==================================

@app.post("/predict")
def predict(movie: MovieInput):

    X = prepare_input(movie)

    # Classification
    prediction = classification_model.predict(X)[0]

    probabilities = classification_model.predict_proba(X)[0]

    classes = classification_model.named_steps["model"].classes_

    confidence = {
        str(cls): round(float(prob) * 100, 2)
        for cls, prob in zip(classes, probabilities)
    }

    # Regression
    rating = regression_model.predict(X)[0]

    return {
        "predicted_category": str(prediction),
        "confidence": confidence,
        "predicted_rating": round(float(rating), 2),
        "classification_threshold": round(float(threshold), 2)
    }