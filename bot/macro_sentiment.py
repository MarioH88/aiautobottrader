"""Macro sentiment loader for Kaggle-style market mood datasets."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger("trading_bot")


class MacroSentimentService:
    """Serves a macro market-mood score from live FRED data or a local CSV export."""

    def __init__(self, csv_path: str = "macro_sentiment.csv", source: str = "fred") -> None:
        """Initialize the macro sentiment service.

        Args:
            csv_path: Path to a Kaggle-exported CSV with date and sentiment columns (fallback).
            source: ``"fred"`` for live FRED series, ``"csv"`` for the static file.
        """
        self._csv_path = csv_path
        self._source = (source or "fred").strip().lower()
        self._score: float = 0.0
        self._loaded = False
        self._fred: Optional[object] = None
        self._load()

    def _load(self) -> None:
        """Load the latest macro score (FRED first, CSV fallback)."""
        if self._source == "fred":
            try:
                from bot.macro_fred import MacroFredService

                self._fred = MacroFredService()
                if self._fred.is_loaded:
                    self._score = self._fred.get_score()
                    self._loaded = True
                    return
                logger.warning("[ERROR] FRED macro unavailable; falling back to CSV")
            except Exception as exc:
                logger.error("[ERROR] FRED macro init failed: %s", exc)
        self._load_csv()

    @property
    def fred(self) -> Optional[object]:
        """Underlying MacroFredService when the FRED source is active (used for training joins)."""
        return self._fred

    def _load_csv(self) -> None:
        """Load the most recent macro sentiment score from disk."""
        if not os.path.exists(self._csv_path):
            logger.info("[INFO] Macro sentiment file not found: %s (using 0.0)", self._csv_path)
            return
        try:
            frame = pd.read_csv(self._csv_path)
            if frame.empty:
                return
            sentiment_col = _find_column(frame, ["macro_sentiment", "sentiment", "score", "compound"])
            if sentiment_col is None:
                logger.error("[ERROR] Macro CSV missing sentiment column in %s", self._csv_path)
                return
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = frame.dropna(subset=["date"]).sort_values("date")
            value = float(frame[sentiment_col].iloc[-1])
            self._score = max(-1.0, min(1.0, value))
            self._loaded = True
            logger.info("[INFO] Macro sentiment loaded: %.3f from %s", self._score, self._csv_path)
        except Exception as exc:
            logger.error("[ERROR] Failed to load macro sentiment: %s", exc)

    def reload(self) -> None:
        """Reload macro sentiment from disk."""
        self._load()

    def get_score(self) -> float:
        """Return the latest macro sentiment in [-1.0, 1.0]."""
        return self._score

    @property
    def is_loaded(self) -> bool:
        """Return True if a macro dataset was loaded successfully."""
        return self._loaded


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name case-insensitively."""
    lower_map = {col.lower(): col for col in frame.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def combine_sentiment(
    news_score: float,
    macro_score: float,
    news_weight: float = 0.6,
) -> float:
    """Blend per-symbol news sentiment with macro market mood.

    Args:
        news_score: VADER score from Alpaca news headlines.
        macro_score: Macro mood score from Kaggle dataset.
        news_weight: Weight applied to news vs macro (0-1).

    Returns:
        Combined sentiment bounded to [-1.0, 1.0].
    """
    macro_weight = 1.0 - news_weight
    combined = (news_score * news_weight) + (macro_score * macro_weight)
    return max(-1.0, min(1.0, combined))
