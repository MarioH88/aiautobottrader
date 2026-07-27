"""Technical indicator helpers using pandas-ta with pure-pandas fallback."""

from __future__ import annotations

import pandas as pd

from bot.indicators_ta import compute_atr, compute_rsi_macd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline technical features for backtesting compatibility.

    Args:
        df: OHLCV DataFrame with a lowercase ``close`` column.

    Returns:
        DataFrame enriched with RSI, MACD, ATR, and legacy column aliases.
    """
    enriched = compute_rsi_macd(df)
    enriched["atr"] = compute_atr(enriched, length=14)
    enriched["rsi_14"] = enriched["rsi"]
    enriched["macd_signal"] = enriched["macd"].ewm(span=9, adjust=False).mean()
    enriched["macd_hist"] = enriched["macd"] - enriched["macd_signal"]
    enriched["sma_20"] = enriched["close"].rolling(20).mean()
    enriched["ema_20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    enriched["bb_high"] = enriched["close"].rolling(20).mean() + 2 * enriched["close"].rolling(20).std()
    enriched["bb_low"] = enriched["close"].rolling(20).mean() - 2 * enriched["close"].rolling(20).std()
    if "sentiment_score" not in enriched.columns:
        enriched["sentiment_score"] = 0.0
    return enriched
