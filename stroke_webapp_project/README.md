# Stroke Prediction Web App

Two parts:
- **Backend**: FastAPI (`backend/`) serving `/predict`
- **Frontend**: static HTML/JS form (`frontend/index.html`)

A demo model is auto-trained on synthetic data the first time you run the backend. To use **your real notebook data**, run the training script with the Kaggle CSV you used in your notebook and it will save a proper pipeline.

## Quickstart (local)

### 1) Create & activate venv (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Start backend
```bash
cd backend
pip install -r requirements.txt
bash run.sh            # Windows: python train.py && uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3) Open frontend
Simply open `frontend/index.html` in your browser (double-click). It will call `http://127.0.0.1:8000/predict` by default.

## Train on your real dataset

If your notebook used the Kaggle file `healthcare-dataset-stroke-data.csv`, run:
```bash
cd backend
python train.py --csv /path/to/healthcare-dataset-stroke-data.csv
```
This will output `model/stroke_pipeline.joblib`, which the API loads automatically.

> Note: The script expects columns:
`gender, ever_married, work_type, Residence_type, smoking_status, age, hypertension, heart_disease, avg_glucose_level, bmi, stroke`

If your notebook used different feature names, update `CATEGORICAL`/`NUMERIC` lists in `train.py` and the form fields in `frontend/index.html` to match your schema.

## API

- `GET /health` → `{"status": "ok"}`
- `POST /predict` with JSON body:
```json
{
  "gender":"Female",
  "ever_married":"Yes",
  "work_type":"Private",
  "Residence_type":"Urban",
  "smoking_status":"never smoked",
  "age":45,
  "hypertension":0,
  "heart_disease":0,
  "avg_glucose_level":110.0,
  "bmi":27.0
}
```
Returns:
```json
{"risk_probability": 0.42, "prediction": 0}
```

## Deploy options

- **Render** / **Railway**: Point the service at `backend` and set start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Vercel** for frontend only: Deploy `frontend/` as static site, keep backend on Render/Railway/EC2.
- **Docker** (optional): Create an image for the backend and serve the `frontend/` via any static host.

## Streamlit (optional alternative)

If you prefer a single-file app:
```bash
pip install streamlit scikit-learn pandas joblib
streamlit run streamlit_app.py
```
See `streamlit_app.py` in the project root.
