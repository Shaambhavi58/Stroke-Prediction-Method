#!/usr/bin/env bash
set -e
# Create demo model if not present
if [ ! -f ../model/stroke_pipeline.joblib ]; then
  echo "No model found; training a demo model..."
  python3 train.py
fi
uvicorn main:app --host 0.0.0.0 --port 8000
