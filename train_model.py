"""Train the direction classifier (Random Forest or XGBoost)."""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = ["rsi", "macd", "sentiment_score"]

df = pd.read_csv("train_data.csv")

# Map legacy training columns to MVP feature schema
if "rsi" not in df.columns and "rsi_14" in df.columns:
    df["rsi"] = df["rsi_14"]
if "sentiment_score" not in df.columns:
    df["sentiment_score"] = 0.0

# Binary target: 1 = up, 0 = down (map legacy -1 to 0)
y = df["target"].replace({-1: 0}).astype(int)
X = df[FEATURE_COLUMNS]

model_type = os.getenv("ML_MODEL_TYPE", "random_forest").strip().lower()

if model_type == "xgboost":
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
else:
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
    )

model.fit(X, y)

joblib.dump(model, "model.pkl")
print(f"Model ({model_type}) trained on {len(X)} rows and saved as model.pkl")
print(f"Features: {FEATURE_COLUMNS}")
