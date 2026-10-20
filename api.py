"""FastAPI app to serve predictions.
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd

app = FastAPI(title='EV Battery Failure Prediction')

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.pkl')
SCALER_PATH = os.environ.get('SCALER_PATH', 'models/scaler.pkl')
LE_PATH = os.environ.get('LE_PATH', 'models/label_encoder.pkl')

if not os.path.exists(MODEL_PATH):
    print('Warning: model not found at', MODEL_PATH)
if not os.path.exists(SCALER_PATH):
    print('Warning: scaler not found at', SCALER_PATH)

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
label_encoder = joblib.load(LE_PATH) if os.path.exists(LE_PATH) else None

class PredictRequest(BaseModel):
    features: dict


@app.get('/')
def root():
    return {'status': 'ok'}


@app.post('/predict')
def predict(req: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail='Model or scaler not available on server')
    # Order features by model expected input
    feat_dict = req.features
    X = pd.DataFrame([feat_dict])
    # Keep only numeric columns and align order using model n_features_in_ if present
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        raise HTTPException(status_code=400, detail='No numeric features provided')
    # If model was trained on a fixed set of features, we can't guarantee column order—try to align
    if hasattr(model, 'n_features_in_'):
        expected_n = model.n_features_in_
        if X_num.shape[1] != expected_n:
            # try to pad/trim columns
            # client should provide same columns used in training
            raise HTTPException(status_code=400, detail=f'Expected {expected_n} numeric features, got {X_num.shape[1]}')

    X_scaled = scaler.transform(X_num.values)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0].tolist() if hasattr(model, 'predict_proba') else None

    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        try:
            pred_label = label_encoder.inverse_transform([int(pred)])[0]
        except Exception:
            pred_label = str(pred)
    else:
        pred_label = str(pred)

    return {'prediction': pred_label, 'raw_prediction': int(pred), 'probabilities': prob}
