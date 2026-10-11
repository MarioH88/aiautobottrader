"""Shared feature engineering and labeling used by both training and live trading.

Keeping this in one module guarantees the model sees identical features offline and online.
All indicators are pure pandas so results do not depend on optional TA libraries.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd

# Order matters: the trained model expects exactly these columns in this order.
FEATURE_COLUMNS: List[str] = [
    "rsi",
    "macd",
    "macd_hist",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "vol_ratio",
    "atr_pct",
    "range_pct",
    "dist_sma20",
    "dist_sma50",
    "bb_pos",
    "hour_sin",
    "hour_cos",
    "sentiment_score",
    "macro_sentiment",
]

# Label encoding for the 3-class direction target.
LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2
LABEL_NAMES = {LABEL_DOWN: "down", LABEL_FLAT: "flat", LABEL_UP: "up"}

# Bars required before every feature is non-NaN (sma50 is the longest lookback).
WARMUP_BARS = 60


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def build_features(
    frame: pd.DataFrame,
    sentiment_score: float = 0.0,
    macro_sentiment: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """Return a copy of an OHLCV frame enriched with every column in FEATURE_COLUMNS.

    Args:
        frame: DataFrame with timestamp, open, high, low, close, volume columns.
        sentiment_score: Constant news sentiment to attach (live) — history has no per-bar news.
        macro_sentiment: Constant (live) or per-row Series aligned to ``frame`` (training).

    Returns:
        Enriched frame. Rows inside the warm-up window will contain NaNs; callers drop them.
    """
    if frame.empty:
        return frame.copy()
    out = frame.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    out["rsi"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_hist"] = macd - signal

    for n in (1, 3, 6, 12):
        out[f"ret_{n}"] = close.pct_change(n)

    vol_mean = volume.rolling(20).mean()
    out["vol_ratio"] = (volume / vol_mean.replace(0, np.nan)).fillna(1.0)

    atr = _atr(out)
    out["atr"] = atr
    out["atr_pct"] = atr / close
    out["range_pct"] = (high - low) / close

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    std20 = close.rolling(20).std()
    out["dist_sma20"] = close / sma20 - 1
    out["dist_sma50"] = close / sma50 - 1
    out["bb_pos"] = ((close - sma20) / (2 * std20.replace(0, np.nan))).clip(-3, 3).fillna(0.0)

    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    angle = 2 * math.pi * minute_of_day / 1440.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)

    out["sentiment_score"] = float(sentiment_score)
    if isinstance(macro_sentiment, pd.Series):
        out["macro_sentiment"] = macro_sentiment.reset_index(drop=True).astype(float).fillna(0.0)
    else:
        out["macro_sentiment"] = float(macro_sentiment)
    return out


def make_labels(frame: pd.DataFrame, horizon_bars: int = 6, atr_mult: float = 0.5) -> pd.Series:
    """Create a 3-class direction label from forward returns scaled by ATR.

    Up (2) when the close ``horizon_bars`` ahead is more than ``atr_mult * ATR`` above the current
    close, down (0) when it is more than that below, otherwise flat (1). The last ``horizon_bars``
    rows have no future and are labeled NaN.
    """
    close = frame["close"].astype(float)
    atr = frame["atr"].astype(float) if "atr" in frame.columns else _atr(frame)
    forward_ret = close.shift(-horizon_bars) / close - 1
    threshold = (atr_mult * atr / close).abs()
    label = pd.Series(LABEL_FLAT, index=frame.index, dtype="float64")
    label[forward_ret > threshold] = LABEL_UP
    label[forward_ret < -threshold] = LABEL_DOWN
    label[forward_ret.isna() | threshold.isna()] = np.nan
    return label


def feature_row(frame: pd.DataFrame) -> dict[str, float]:
    """Return the last row's feature values as a plain dict (used to populate MarketBarState)."""
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    values: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = row.get(column, np.nan)
        values[column] = float(value) if pd.notna(value) else 0.0
    return values
