# EV Battery Failure Prediction

Project scaffold for exploring EV battery production dataset, cleaning data, training a model, and serving predictions via FastAPI + Docker.

Files:

- data_exploration.py — exploratory data analysis and visualizations saved to `reports/`.
- data_cleaning.py — basic cleaning and saving cleaned CSV to `data/cleaned.csv`.
- model_training.py — trains a RandomForest model and saves `models/model.pkl` and `models/scaler.pkl`.
- api.py — FastAPI app exposing `/predict` endpoint.
- Dockerfile — container image for the API.
- requirements.txt — Python dependencies.

Usage (quick):

1. Move your dataset into the project root and name it `ev_battery.csv` (or pass `--csv PATH`).
2. Run exploration: `python data_exploration.py --csv ev_battery.csv`
3. Clean data: `python data_cleaning.py --csv ev_battery.csv`
4. Train model: `python model_training.py`
5. Start API: `uvicorn api:app --host 0.0.0.0 --port 8000`

