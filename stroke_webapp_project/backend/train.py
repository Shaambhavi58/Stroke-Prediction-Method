#!/usr/bin/env python3
"""
Train a stroke prediction pipeline.

Usage:
  python train.py --csv path/to/healthcare-dataset-stroke-data.csv
If --csv is omitted, a synthetic dataset will be generated for a demo model.

The trained pipeline is saved to: model/stroke_pipeline.joblib
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump

# Typical Kaggle stroke dataset columns
CATEGORICAL = ["gender","ever_married","work_type","Residence_type","smoking_status"]
NUMERIC = ["age","hypertension","heart_disease","avg_glucose_level","bmi"]
TARGET = "stroke"

def load_real_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Clean/standardize as needed
    # Drop 'id' if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Basic NA handling
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    return df

def make_synthetic(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    genders = np.array(["Male","Female","Other"])
    ever_married = np.array(["Yes","No"])
    work_types = np.array(["Private","Self-employed","Govt_job","children","Never_worked"])
    residence = np.array(["Urban","Rural"])
    smoking = np.array(["formerly smoked","never smoked","smokes","Unknown"])

    age = rng.normal(45, 18, n).clip(0, 100)
    hypertension = (rng.random(n) < (age/100)*0.2).astype(int)
    heart_disease = (rng.random(n) < (age/100)*0.15).astype(int)
    avg_glucose_level = rng.normal(110, 35, n).clip(40, 300)
    bmi = rng.normal(27, 6, n).clip(12, 60)

    df = pd.DataFrame({
        "gender": rng.choice(genders, n, p=[0.48, 0.51, 0.01]),
        "ever_married": rng.choice(ever_married, n, p=[0.6, 0.4]),
        "work_type": rng.choice(work_types, n, p=[0.65, 0.18, 0.12, 0.04, 0.01]),
        "Residence_type": rng.choice(residence, n, p=[0.5, 0.5]),
        "smoking_status": rng.choice(smoking, n, p=[0.18, 0.48, 0.2, 0.14]),
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
    })
    # Generate a pseudo stroke risk signal
    logit = (-6.0
             + 0.05*(df["age"]-50)
             + 0.7*df["hypertension"]
             + 0.6*df["heart_disease"]
             + 0.015*(df["avg_glucose_level"]-110)
             + 0.03*(df["bmi"]-27)
             + 0.15*(df["smoking_status"].isin(["smokes","formerly smoked"]).astype(int))
             + 0.1*(df["ever_married"].eq("Yes")).astype(int)
            )
    prob = 1/(1+np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    df[TARGET] = y
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to real Kaggle stroke CSV")
    parser.add_argument("--out", type=str, default="../model/stroke_pipeline.joblib", help="Output path")
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        df = load_real_csv(args.csv)
        missing = set(CATEGORICAL+NUMERIC+[TARGET]) - set(df.columns)
        if missing:
            print("ERROR: CSV missing columns:", missing)
            sys.exit(2)
    else:
        print("No CSV provided or path not found; creating a synthetic demo dataset...")
        df = make_synthetic()

    X = df[CATEGORICAL+NUMERIC]
    y = df[TARGET]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC)
    ])
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, preds)
    acc = accuracy_score(yte, (preds>0.5).astype(int))
    print(f"Validation AUC: {auc:.3f}  |  Acc@0.5: {acc:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump(pipe, args.out)
    print("Saved pipeline to:", args.out)

if __name__ == "__main__":
    main()
