"""Historical and streaming bar ingestion with technical feature computation."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.features import build_features, feature_row
from bot.indicators_ta import compute_atr
from bot.schemas import MarketBarState
from bot.symbols import is_crypto_symbol, normalize_symbol

logger = logging.getLogger("trading_bot")


class DataPipeline:
    """Fetches OHLCV bars and computes rolling technical features."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        bar_limit: int = 200,
        atr_period: int = 14,
        timeframe_min: int = 5,
        data_feed: str = "iex",
    ) -> None:
        """Initialize the data pipeline.

        Args:
            api_key: Alpaca API key identifier.
            api_secret: Alpaca API secret key.
            bar_limit: Number of bars to pull per REST request (>= feature warm-up).
            atr_period: ATR lookback for trailing stop calculations.
            timeframe_min: Bar size in minutes; must match the model's training data.
            data_feed: ``"iex"`` (free tier) or ``"sip"`` (paid) for stock bars.
        """
        self._stock_client = StockHistoricalDataClient(api_key, api_secret)
        self._crypto_client = CryptoHistoricalDataClient(api_key, api_secret)
        self._bar_limit = bar_limit
        self._atr_period = atr_period
        self._timeframe = TimeFrame(max(1, int(timeframe_min)), TimeFrameUnit.Minute)
        self._feed = DataFeed.SIP if data_feed == "sip" else DataFeed.IEX
        self._frames: Dict[str, pd.DataFrame] = {}
        # Calendar days needed to cover bar_limit bars of regular-session trading, with slack
        # for weekends/holidays. Alpaca's `limit` counts from `start`, so we fetch a window
        # and keep the most recent bar_limit bars ourselves.
        session_bars_per_day = max(1, 390 // max(1, int(timeframe_min)))
        trading_days = math.ceil(bar_limit / session_bars_per_day)
        self._lookback_days = trading_days * 2 + 4

    def _window_start(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self._lookback_days)

    def ingest_stream_frame(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Merge a WebSocket-backed frame into the pipeline cache.

        Args:
            symbol: Ticker symbol.
            frame: Cached bars from the stream client.

        Returns:
            Stored OHLCV frame.
        """
        if frame.empty:
            return frame
        key = normalize_symbol(symbol)
        self._frames[key] = frame.copy()
        return self._frames[key]

    async def _fetch_stock_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch recent stock bars via REST."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe,
            start=self._window_start(),
            feed=self._feed,
        )
        response = await asyncio.to_thread(self._stock_client.get_stock_bars, request)
        bars = list(response.data.get(symbol, []))[-self._bar_limit :]
        if not bars:
            raise RuntimeError(f"No bars returned for {symbol}")
        rows: List[dict] = []
        for bar in bars:
            rows.append(
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "symbol": symbol,
                }
            )
        frame = pd.DataFrame(rows)
        self._frames[symbol] = frame
        return frame

    async def _fetch_crypto_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch recent crypto bars via REST."""
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe,
            start=self._window_start(),
        )
        response = await asyncio.to_thread(self._crypto_client.get_crypto_bars, request)
        bars = list(response.data.get(symbol, []))[-self._bar_limit :]
        if not bars:
            raise RuntimeError(f"No crypto bars returned for {symbol}")
        rows: List[dict] = []
        for bar in bars:
            rows.append(
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "symbol": symbol,
                }
            )
        frame = pd.DataFrame(rows)
        self._frames[symbol] = frame
        return frame

    async def fetch_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch recent 1-minute OHLCV bars for a symbol via REST.

        Args:
            symbol: Ticker symbol (stock or crypto).

        Returns:
            DataFrame with lowercase OHLCV columns and a timestamp column.

        Raises:
            RuntimeError: If Alpaca returns no bars.
        """
        symbol = normalize_symbol(symbol)
        try:
            if is_crypto_symbol(symbol):
                return await self._fetch_crypto_bars(symbol)
            return await self._fetch_stock_bars(symbol)
        except Exception as exc:
            logger.error("[ERROR] Failed to fetch bars for %s: %s", symbol, exc)
            raise

    def compute_features(
        self,
        frame: pd.DataFrame,
        sentiment_score: float = 0.0,
        macro_sentiment: float = 0.0,
    ) -> pd.DataFrame:
        """Compute the full model feature set (see bot.features.FEATURE_COLUMNS).

        Args:
            frame: OHLCV DataFrame.
            sentiment_score: News sentiment in [-1.0, 1.0].
            macro_sentiment: Macro mood in [-1.0, 1.0].

        Returns:
            DataFrame enriched with every feature column plus ``atr``.
        """
        if frame.empty:
            return frame
        return build_features(frame, sentiment_score=sentiment_score, macro_sentiment=macro_sentiment)

    def latest_bar_state(
        self,
        frame: pd.DataFrame,
        symbol: str,
        sentiment_score: float = 0.0,
        macro_sentiment: float = 0.0,
    ) -> Optional[MarketBarState]:
        """Build a validated MarketBarState from the latest enriched bar.

        Args:
            frame: Feature-enriched OHLCV DataFrame.
            symbol: Ticker symbol.
            sentiment_score: Combined news+macro sentiment in [-1.0, 1.0].
            macro_sentiment: Macro mood score from Kaggle dataset.

        Returns:
            Validated MarketBarState or None if the frame is insufficient.
        """
        if frame.empty:
            return None
        feature_frame = self.compute_features(
            frame, sentiment_score=sentiment_score, macro_sentiment=macro_sentiment
        )
        if feature_frame.empty:
            return None
        row = feature_frame.iloc[-1]
        if pd.isna(row.get("dist_sma50")):
            logger.error(
                "[ERROR] %s: only %d bars, need %d+ for the feature warm-up",
                symbol,
                len(feature_frame),
                60,
            )
            return None
        try:
            return MarketBarState(
                timestamp=str(row["timestamp"]),
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                rsi=float(row["rsi"]),
                macd=float(row["macd"]),
                sentiment_score=float(sentiment_score),
                macro_sentiment=float(macro_sentiment),
                atr=float(row.get("atr", 0.0) or 0.0),
                features=feature_row(feature_frame),
            )
        except Exception as exc:
            logger.error("[ERROR] Failed to validate MarketBarState for %s: %s", symbol, exc)
            return None

    def get_cached_frame(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return the last cached frame for a symbol."""
        return self._frames.get(symbol)

    def latest_atr(self, symbol: str) -> float:
        """Return the latest ATR for a cached symbol frame."""
        frame = self._frames.get(symbol)
        if frame is None or frame.empty:
            return 0.0
        return compute_atr(frame, length=self._atr_period)
