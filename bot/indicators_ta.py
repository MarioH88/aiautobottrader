"""Technical indicator computations with pandas-ta preferred, pure-pandas fallback."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("trading_bot")

_PANDAS_TA_AVAILABLE = False

try:
    import pandas_ta as pta  # type: ignore

    _PANDAS_TA_AVAILABLE = True
except Exception as exc:
    logger.warning("[ERROR] pandas-ta unavailable (%s); using pure-pandas fallback", exc)


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Compute RSI using pandas rolling means."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> pd.Series:
    """Compute MACD line using EMA(12) - EMA(26)."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


def _atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def compute_rsi_macd(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rsi and macd columns to an OHLCV frame.

    Args:
        frame: DataFrame containing a ``close`` column.

    Returns:
        Copy of the frame with ``rsi`` and ``macd`` columns populated.
    """
    enriched = frame.copy()
    if _PANDAS_TA_AVAILABLE:
        try:
            enriched["rsi"] = pta.rsi(enriched["close"], length=14)
            macd_frame = pta.macd(enriched["close"])
            enriched["macd"] = macd_frame.iloc[:, 0] if macd_frame is not None else _macd(enriched["close"])
            return enriched.dropna(subset=["rsi", "macd"])
        except Exception as exc:
            logger.error("[ERROR] pandas-ta computation failed: %s", exc)
    enriched["rsi"] = _rsi(enriched["close"])
    enriched["macd"] = _macd(enriched["close"])
    return enriched.dropna(subset=["rsi", "macd"])


def compute_atr(frame: pd.DataFrame, length: int = 14) -> float:
    """Return the latest ATR value for a frame.

    Args:
        frame: OHLCV DataFrame with high, low, close columns.
        length: ATR lookback period.

    Returns:
        Latest ATR value, or 0.0 if insufficient data.
    """
    if frame.empty or len(frame) < length + 1:
        return 0.0
    if _PANDAS_TA_AVAILABLE:
        try:
            atr_series = pta.atr(frame["high"], frame["low"], frame["close"], length=length)
            if atr_series is not None and not atr_series.empty:
                value = float(atr_series.iloc[-1])
                return value if pd.notna(value) else 0.0
        except Exception as exc:
            logger.error("[ERROR] pandas-ta ATR failed: %s", exc)
    atr_series = _atr(frame, length=length)
    value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    return value if pd.notna(value) else 0.0
