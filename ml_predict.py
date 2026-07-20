"""Machine learning inference helpers."""

from __future__ import annotations

from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["rsi", "macd", "sentiment_score"]


def load_sklearn_model(path: str) -> Optional[Any]:
    """Load a scikit-learn model from disk.

    Args:
        path: Filesystem path to the serialized model.

    Returns:
        Loaded model or None on failure.
    """
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None


def predict_action(model: Any, features: pd.DataFrame) -> int:
    """Predict a directional action from a feature row.

    Args:
        model: Trained classifier.
        features: Single-row DataFrame or array-like features.

    Returns:
        1 for buy/up, 0 for sell/down, with hold encoded as 0 when flat logic applies.
    """
    if model is None:
        return 0
    try:
        if isinstance(features, pd.DataFrame):
            if all(col in features.columns for col in FEATURE_COLUMNS):
                values = features[FEATURE_COLUMNS].to_numpy()
            else:
                values = features.to_numpy()
        else:
            values = np.asarray(features)
        prediction = int(model.predict(values)[0])
        return 1 if prediction == 1 else 0
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return 0
