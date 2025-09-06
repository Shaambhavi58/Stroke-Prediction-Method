import streamlit as st
import pandas as pd
from joblib import load
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "model/stroke_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    st.warning("No model found. Training a demo model...")
    import subprocess, sys
    subprocess.run([sys.executable, "backend/train.py"], check=True)

pipe = load(MODEL_PATH)

st.title("Stroke Risk Predictor (Streamlit)")
st.write("Enter details and click Predict.")

gender = st.selectbox("Gender", ["Female","Male","Other"])
ever_married = st.selectbox("Ever married", ["Yes","No"])
work_type = st.selectbox("Work type", ["Private","Self-employed","Govt_job","children","Never_worked"])
Residence_type = st.selectbox("Residence type", ["Urban","Rural"])
smoking_status = st.selectbox("Smoking status", ["never smoked","formerly smoked","smokes","Unknown"])

age = st.number_input("Age", 0, 120, 45)
hypertension = st.selectbox("Hypertension (0/1)", [0,1])
heart_disease = st.selectbox("Heart disease (0/1)", [0,1])
avg_glucose_level = st.number_input("Avg glucose level", 0.0, 500.0, 110.0)
bmi = st.number_input("BMI", 10.0, 80.0, 27.0)

if st.button("Predict"):
    df = pd.DataFrame([{
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "smoking_status": smoking_status,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
    }])
    proba = float(pipe.predict_proba(df)[:,1][0])
    label = int(proba >= 0.5)
    st.success(f"Risk Probability: {proba*100:.1f}%  |  Prediction: {'High Risk' if label else 'Low Risk'}")
