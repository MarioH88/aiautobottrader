"""ML-driven strategy evaluation engine."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import joblib
import pandas as pd

from bot.schemas import MarketBarState, TradeSignal

logger = logging.getLogger("trading_bot")


class StrategyEngine:
    """Predicts short-term direction using a pre-trained Random Forest."""

    FEATURE_COLUMNS = ["rsi", "macd", "sentiment_score"]

    def __init__(self, model_path: str = "model.pkl") -> None:
        """Load the pre-trained classifier if available.

        Args:
            model_path: Filesystem path to the serialized scikit-learn model.
        """
        self._model_path = model_path
        self._model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from disk."""
        if not os.path.exists(self._model_path):
            logger.error("[CRITICAL] Model file not found at %s", self._model_path)
            return
        try:
            self._model = joblib.load(self._model_path)
            logger.info("Loaded ML model from %s", self._model_path)
        except Exception as exc:
            logger.error("[CRITICAL] Failed to load model: %s", exc)
            self._model = None

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

    def evaluate(self, bar: MarketBarState) -> TradeSignal:
        """Evaluate a bar and return a directional trade signal.

        Args:
            bar: Validated market feature state.

        Returns:
            TradeSignal with action buy, sell, or hold.
        """
        if self._model is None:
            logger.error("[ERROR] Model unavailable; using rule-based fallback for %s", bar.symbol)
            return self._rule_based_signal(bar)

        features = pd.DataFrame(
            [[bar.rsi, bar.macd, bar.sentiment_score]],
            columns=self.FEATURE_COLUMNS,
        )
        try:
            prediction = int(self._model.predict(features)[0])
            proba_fn = getattr(self._model, "predict_proba", None)
            confidence = float(max(proba_fn(features)[0])) if proba_fn else 0.6
            if prediction == 1:
                return TradeSignal(
                    symbol=bar.symbol,
                    action="buy",
                    confidence=confidence,
                    reason="ml-5m-up",
                    target_allocation_pct=min(0.15, 0.05 + confidence * 0.10 + max(bar.sentiment_score, 0) * 0.05),
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
