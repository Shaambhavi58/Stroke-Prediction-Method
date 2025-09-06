#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel, Field, constr
from joblib import load
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model
import os
MODEL_PATH = os.environ.get("MODEL_PATH", "../model/stroke_pipeline.joblib")
pipe = load(MODEL_PATH)

app = FastAPI(title="Stroke Prediction API", version="1.0.0")

# Allow local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StrokeInput(BaseModel):
    gender: constr(strip_whitespace=True) = Field(..., example="Female")
    ever_married: constr(strip_whitespace=True) = Field(..., example="Yes")
    work_type: constr(strip_whitespace=True) = Field(..., example="Private")
    Residence_type: constr(strip_whitespace=True) = Field(..., example="Urban")
    smoking_status: constr(strip_whitespace=True) = Field(..., example="never smoked")
    age: float = Field(..., ge=0, le=120, example=45)
    hypertension: int = Field(..., ge=0, le=1, example=0)
    heart_disease: int = Field(..., ge=0, le=1, example=0)
    avg_glucose_level: float = Field(..., ge=0, le=500, example=105.3)
    bmi: float = Field(..., ge=10, le=80, example=26.5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: StrokeInput):
    X = {
        "gender": [payload.gender],
        "ever_married": [payload.ever_married],
        "work_type": [payload.work_type],
        "Residence_type": [payload.Residence_type],
        "smoking_status": [payload.smoking_status],
        "age": [payload.age],
        "hypertension": [payload.hypertension],
        "heart_disease": [payload.heart_disease],
        "avg_glucose_level": [payload.avg_glucose_level],
        "bmi": [payload.bmi],
    }
    import pandas as pd
    df = pd.DataFrame(X)
    proba = float(pipe.predict_proba(df)[:,1][0])
    label = int(proba >= 0.5)
    return {"risk_probability": proba, "prediction": label}
