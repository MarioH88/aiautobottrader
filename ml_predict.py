"""Machine learning inference helpers shared by the backtester."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from bot.features import FEATURE_COLUMNS as RICH_FEATURE_COLUMNS
from bot.features import LABEL_DOWN, LABEL_UP

LEGACY_FEATURE_COLUMNS = ["rsi", "macd", "sentiment_score"]
# Kept for backwards compatibility with older imports.
FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS


class ModelBundle:
    """Uniform wrapper over a train_model.py bundle or a bare legacy estimator."""

    def __init__(self, artifact: Any) -> None:
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.feature_columns: List[str] = list(artifact.get("feature_columns", RICH_FEATURE_COLUMNS))
            self.n_classes = int(artifact.get("n_classes", 3))
            self.min_confidence = float(artifact.get("recommended_min_confidence", 0.6))
        else:
            self.model = artifact
            self.feature_columns = LEGACY_FEATURE_COLUMNS
            self.n_classes = 2
            self.min_confidence = 0.58

    def probabilities(self, features: pd.DataFrame) -> Dict[int, float]:
        values = features[self.feature_columns].to_numpy(dtype=float)
        proba_fn = getattr(self.model, "predict_proba", None)
        classes = getattr(self.model, "classes_", None)
        if proba_fn is None or classes is None:
            return {int(self.model.predict(values)[0]): 1.0}
        probs = np.asarray(proba_fn(values)[0], dtype=float)
        return {int(c): float(p) for c, p in zip(classes, probs)}


def load_sklearn_model(path: str) -> Optional[ModelBundle]:
    """Load a model bundle (or legacy estimator) from disk."""
    try:
        return ModelBundle(joblib.load(path))
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None


def predict_action(model: Optional[ModelBundle], features: pd.DataFrame) -> int:
    """Return 1 (buy), -1 (sell), or 0 (hold) for a single feature row."""
    if model is None:
        return 0
    try:
        probs = model.probabilities(features)
        if model.n_classes >= 3:
            p_up = probs.get(LABEL_UP, 0.0)
            p_down = probs.get(LABEL_DOWN, 0.0)
            if p_up >= model.min_confidence and p_up >= p_down:
                return 1
            if p_down >= model.min_confidence and p_down > p_up:
                return -1
            return 0
        label = max(probs, key=probs.get)
        if probs[label] < model.min_confidence:
            return 0
        return 1 if label == 1 else -1
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return 0
