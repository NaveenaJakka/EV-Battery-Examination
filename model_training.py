"""Train a model to predict EV battery failure.
- Expects cleaned data at data/cleaned.csv (default)
- Determines target column automatically: 'failure'|'target' or last column
- Produces models/model.pkl and models/scaler.pkl

Run: python model_training.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def load_cleaned(path='data/cleaned.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned CSV not found at {path}. Run data_cleaning.py first or pass --cleaned PATH")
    return pd.read_csv(path)


def choose_target(df):
    for candidate in ['failure', 'target', 'is_failed', 'label']:
        if candidate in df.columns:
            return candidate
    # fallback: last column
    return df.columns[-1]


def train(save_dir='models', cleaned_path='data/cleaned.csv'):
    df = load_cleaned(cleaned_path)
    target = choose_target(df)
    print('Using target column:', target)

    X = df.drop(columns=[target])
    y = df[target]

    # Encode y if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))

    # For simplicity, drop non-numeric columns (after one-hot they should be numeric)
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError('No numeric features found after cleaning/encoding. Check data_cleaning step.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    print('Classification report:\n', classification_report(y_test, preds))
    if probs is not None and len(np.unique(y_test)) == 2:
        print('ROC AUC:', roc_auc_score(y_test, probs))

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    print('Saved model and scaler to', save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned', default='data/cleaned.csv')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    train(save_dir=args.out, cleaned_path=args.cleaned)
