"""Macro market-mood score built from live FRED series (no API key required).

Uses the public ``fredgraph.csv`` endpoint. Each series' 20-day change is z-scored against its
own one-year history, signed so that positive means risk-on, then averaged into [-1, 1].
A daily score history is kept so training rows can be joined to the macro regime they occurred in.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("trading_bot")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# series_id -> sign (+1: higher is risk-on, -1: higher is risk-off)
DEFAULT_SERIES: Dict[str, int] = {
    "VIXCLS": -1,  # CBOE volatility index
    "BAMLH0A0HYM2": -1,  # High-yield credit spread
    "T10Y2Y": +1,  # 10y-2y yield curve
    "DGS10": -1,  # 10y treasury yield
    "DTWEXBGS": -1,  # Trade-weighted dollar index
    "SP500": +1,  # S&P 500 level
}

CHANGE_WINDOW_DAYS = 20
ZSCORE_WINDOW_DAYS = 252


class MacroFredService:
    """Fetches FRED series, caches them locally, and serves a bounded macro score."""

    def __init__(
        self,
        cache_path: str = "macro_fred_cache.csv",
        refresh_hours: float = 12.0,
        series: Optional[Dict[str, int]] = None,
        timeout_sec: float = 20.0,
    ) -> None:
        self._cache_path = cache_path
        self._refresh_sec = refresh_hours * 3600
        self._series = series or DEFAULT_SERIES
        self._timeout = timeout_sec
        self._scores: pd.Series = pd.Series(dtype="float64")
        self._loaded = False
        self.reload()

    # ------------------------------------------------------------------ fetching
    def _fetch_series(self, series_id: str) -> pd.Series:
        url = FRED_CSV_URL.format(series=series_id)
        response = requests.get(url, timeout=self._timeout)
        response.raise_for_status()
        from io import StringIO

        frame = pd.read_csv(StringIO(response.text))
        date_col = frame.columns[0]
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        values = pd.to_numeric(frame[series_id], errors="coerce")
        series = pd.Series(values.to_numpy(), index=frame[date_col]).dropna()
        series.name = series_id
        return series

    def _cache_is_fresh(self) -> bool:
        if not os.path.exists(self._cache_path):
            return False
        age = time.time() - os.path.getmtime(self._cache_path)
        return age < self._refresh_sec

    def _load_raw(self) -> pd.DataFrame:
        if self._cache_is_fresh():
            cached = pd.read_csv(self._cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return cached
        columns = []
        for series_id in self._series:
            try:
                columns.append(self._fetch_series(series_id))
            except Exception as exc:
                logger.error("[ERROR] FRED fetch failed for %s: %s", series_id, exc)
        if not columns:
            if os.path.exists(self._cache_path):
                logger.warning("[ERROR] Using stale FRED cache after fetch failure")
                return pd.read_csv(self._cache_path, index_col=0, parse_dates=True)
            return pd.DataFrame()
        raw = pd.concat(columns, axis=1).sort_index()
        raw = raw.ffill()
        try:
            raw.to_csv(self._cache_path)
        except Exception as exc:
            logger.error("[ERROR] Could not write FRED cache: %s", exc)
        return raw

    # ------------------------------------------------------------------ scoring
    def _score_frame(self, raw: pd.DataFrame) -> pd.Series:
        if raw.empty:
            return pd.Series(dtype="float64")
        parts = []
        for series_id, sign in self._series.items():
            if series_id not in raw.columns:
                continue
            values = raw[series_id].astype(float)
            # Percent change for level series, absolute change for rates/spreads (already in %).
            if series_id in {"SP500", "VIXCLS", "DTWEXBGS"}:
                change = values.pct_change(CHANGE_WINDOW_DAYS)
            else:
                change = values.diff(CHANGE_WINDOW_DAYS)
            mean = change.rolling(ZSCORE_WINDOW_DAYS, min_periods=60).mean()
            std = change.rolling(ZSCORE_WINDOW_DAYS, min_periods=60).std()
            z = ((change - mean) / std.replace(0, np.nan)).clip(-3, 3) / 3.0
            parts.append(sign * z)
        if not parts:
            return pd.Series(dtype="float64")
        score = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
        return score.dropna().clip(-1.0, 1.0)

    def reload(self) -> None:
        """Refresh the macro score history from cache or FRED."""
        try:
            raw = self._load_raw()
            self._scores = self._score_frame(raw)
            self._loaded = not self._scores.empty
            if self._loaded:
                logger.info(
                    "[INFO] FRED macro score %.3f as of %s (%d daily points)",
                    float(self._scores.iloc[-1]),
                    self._scores.index[-1].date(),
                    len(self._scores),
                )
        except Exception as exc:
            logger.error("[ERROR] FRED macro scoring failed: %s", exc)
            self._loaded = False

    # ------------------------------------------------------------------ public
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_score(self) -> float:
        """Latest macro score in [-1, 1] (0.0 when unavailable)."""
        if self._scores.empty:
            return 0.0
        return float(self._scores.iloc[-1])

    def daily_scores(self) -> pd.Series:
        """Full daily score history indexed by date."""
        return self._scores.copy()

    def scores_for_timestamps(self, timestamps: pd.Series) -> pd.Series:
        """Map each bar timestamp to the most recent macro score at or before that date."""
        if self._scores.empty:
            return pd.Series(0.0, index=timestamps.index)
        ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        lookup = self._scores.copy()
        lookup.index = pd.to_datetime(lookup.index).normalize()
        left = pd.DataFrame({"date": ts.to_numpy()}, index=timestamps.index).sort_values("date")
        right = lookup.rename("macro").reset_index().rename(columns={"index": "date"})
        right.columns = ["date", "macro"]
        merged = pd.merge_asof(left, right.sort_values("date"), on="date", direction="backward")
        merged.index = left.index
        return merged["macro"].reindex(timestamps.index).fillna(0.0)
