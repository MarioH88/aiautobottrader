"""ML-driven strategy evaluation engine."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from bot.features import FEATURE_COLUMNS, LABEL_DOWN, LABEL_UP
from bot.schemas import MarketBarState, TradeSignal

logger = logging.getLogger("trading_bot")

LEGACY_FEATURE_COLUMNS = ["rsi", "macd", "sentiment_score"]
DEFAULT_MIN_CONFIDENCE = 0.58


class StrategyEngine:
    """Predicts short-term direction with a trained classifier and gates on confidence.

    Supports two artifact formats in ``model.pkl``:
      * a bundle dict ``{"model", "feature_columns", "n_classes", "recommended_min_confidence"}``
        written by ``train_model.py`` (3-class up/flat/down on the rich feature set), or
      * a bare legacy estimator trained on ``rsi, macd, sentiment_score`` with a binary target.
    """

    FEATURE_COLUMNS = FEATURE_COLUMNS

    def __init__(
        self,
        model_path: str = "model.pkl",
        min_signal_confidence: Optional[float] = None,
    ) -> None:
        """Load the classifier if available.

        Args:
            model_path: Path to the serialized model or bundle.
            min_signal_confidence: Minimum class probability required to act. ``None`` uses the
                threshold stored in the bundle (or 0.58 for legacy models).
        """
        self._model_path = model_path
        self._configured_min_confidence = min_signal_confidence
        self._model: Optional[Any] = None
        self._feature_columns: List[str] = LEGACY_FEATURE_COLUMNS
        self._n_classes: int = 2
        self._min_confidence: float = min_signal_confidence or DEFAULT_MIN_CONFIDENCE
        self._load_model()

    # ------------------------------------------------------------------ loading
    def _apply_artifact(self, artifact: Any) -> None:
        """Configure the engine from a bundle dict or bare estimator."""
        if isinstance(artifact, dict) and "model" in artifact:
            self._model = artifact["model"]
            self._feature_columns = list(artifact.get("feature_columns", FEATURE_COLUMNS))
            self._n_classes = int(artifact.get("n_classes", 3))
            recommended = artifact.get("recommended_min_confidence")
        else:
            self._model = artifact
            self._feature_columns = LEGACY_FEATURE_COLUMNS
            self._n_classes = 2
            recommended = None
        if self._configured_min_confidence is not None:
            self._min_confidence = float(self._configured_min_confidence)
        elif recommended is not None:
            self._min_confidence = float(recommended)
        else:
            self._min_confidence = DEFAULT_MIN_CONFIDENCE

    def _load_model(self) -> None:
        """Load the model artifact from disk."""
        if not os.path.exists(self._model_path):
            logger.error("[CRITICAL] Model file not found at %s", self._model_path)
            return
        try:
            self._apply_artifact(joblib.load(self._model_path))
            logger.info(
                "Loaded ML model from %s (%d classes, %d features, min_confidence=%.2f)",
                self._model_path,
                self._n_classes,
                len(self._feature_columns),
                self._min_confidence,
            )
        except Exception as exc:
            logger.error("[CRITICAL] Failed to load model: %s", exc)
            self._model = None

    @property
    def min_confidence(self) -> float:
        """Active confidence threshold."""
        return self._min_confidence

    # ------------------------------------------------------------------ helpers
    def _feature_vector(self, bar: MarketBarState) -> Optional[pd.DataFrame]:
        """Assemble the model input row from a bar, or None if a feature is missing."""
        values: Dict[str, float] = {}
        for column in self._feature_columns:
            if column == "rsi":
                values[column] = bar.rsi
            elif column == "macd":
                values[column] = bar.macd
            elif column == "sentiment_score":
                values[column] = bar.sentiment_score
            elif column == "macro_sentiment":
                values[column] = bar.macro_sentiment
            elif column in bar.features:
                values[column] = bar.features[column]
            else:
                logger.error("[ERROR] Missing feature %s for %s", column, bar.symbol)
                return None
        return pd.DataFrame([values], columns=self._feature_columns)

    def _rule_based_signal(self, bar: MarketBarState) -> TradeSignal:
        """Fallback RSI/MACD rule set when the model is unavailable."""
        if bar.rsi < 35 and bar.macd > 0:
            return TradeSignal(
                symbol=bar.symbol,
                action="buy",
                confidence=0.55,
                reason="rule-rsi-macd",
                target_allocation_pct=0.08,
            )
        if bar.rsi > 70 and bar.macd < 0:
            return TradeSignal(
                symbol=bar.symbol,
                action="sell",
                confidence=0.55,
                reason="rule-rsi-macd",
                target_allocation_pct=0.0,
            )
        return TradeSignal(symbol=bar.symbol, action="hold", confidence=0.0, reason="rule-hold")

    @staticmethod
    def _buy_allocation(confidence: float, sentiment: float) -> float:
        return min(0.15, 0.05 + confidence * 0.10 + max(sentiment, 0) * 0.05)

    def _class_probabilities(self, features: pd.DataFrame) -> Dict[int, float]:
        """Return {class_label: probability}; falls back to a hard prediction."""
        proba_fn = getattr(self._model, "predict_proba", None)
        classes = getattr(self._model, "classes_", None)
        if proba_fn is None or classes is None:
            predicted = int(self._model.predict(features)[0])
            return {predicted: 0.6}
        probs = np.asarray(proba_fn(features)[0], dtype=float)
        return {int(label): float(p) for label, p in zip(classes, probs)}

    # ------------------------------------------------------------------ public
    def evaluate(self, bar: MarketBarState) -> TradeSignal:
        """Evaluate a bar and return a buy / sell / hold signal."""
        if self._model is None:
            logger.error("[ERROR] Model unavailable; using rule-based fallback for %s", bar.symbol)
            return self._rule_based_signal(bar)

        features = self._feature_vector(bar)
        if features is None:
            return self._rule_based_signal(bar)

        try:
            probs = self._class_probabilities(features)
            if self._n_classes >= 3:
                p_up = probs.get(LABEL_UP, 0.0)
                p_down = probs.get(LABEL_DOWN, 0.0)
                if p_up >= self._min_confidence and p_up >= p_down:
                    return TradeSignal(
                        symbol=bar.symbol,
                        action="buy",
                        confidence=p_up,
                        reason="ml-up",
                        target_allocation_pct=self._buy_allocation(p_up, bar.sentiment_score),
                    )
                if p_down >= self._min_confidence and p_down > p_up:
                    return TradeSignal(
                        symbol=bar.symbol,
                        action="sell",
                        confidence=p_down,
                        reason="ml-down",
                        target_allocation_pct=0.0,
                    )
                return TradeSignal(
                    symbol=bar.symbol,
                    action="hold",
                    confidence=max(p_up, p_down),
                    reason="ml-low-confidence" if max(p_up, p_down) < self._min_confidence else "ml-flat",
                    target_allocation_pct=0.0,
                )

            # Legacy binary model: 1 = up, 0 = down.
            prediction = max(probs, key=probs.get)
            confidence = probs[prediction]
            if confidence < self._min_confidence:
                return TradeSignal(
                    symbol=bar.symbol,
                    action="hold",
                    confidence=confidence,
                    reason="ml-low-confidence",
                    target_allocation_pct=0.0,
                )
            if prediction == 1:
                return TradeSignal(
                    symbol=bar.symbol,
                    action="buy",
                    confidence=confidence,
                    reason="ml-5m-up",
                    target_allocation_pct=self._buy_allocation(confidence, bar.sentiment_score),
                )
            return TradeSignal(
                symbol=bar.symbol,
                action="sell",
                confidence=confidence,
                reason="ml-5m-down",
                target_allocation_pct=0.0,
            )
        except Exception as exc:
            logger.error("[ERROR] Prediction failed for %s: %s", bar.symbol, exc)
            return self._rule_based_signal(bar)
