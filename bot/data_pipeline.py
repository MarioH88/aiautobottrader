"""Historical and streaming bar ingestion with technical feature computation."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.indicators_ta import compute_atr, compute_rsi_macd
from bot.schemas import MarketBarState

logger = logging.getLogger("trading_bot")


class DataPipeline:
    """Fetches OHLCV bars and computes rolling technical features."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        bar_limit: int = 120,
        atr_period: int = 14,
    ) -> None:
        """Initialize the data pipeline.

        Args:
            api_key: Alpaca API key identifier.
            api_secret: Alpaca API secret key.
            bar_limit: Number of 1-minute bars to pull per REST request.
            atr_period: ATR lookback for trailing stop calculations.
        """
        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._bar_limit = bar_limit
        self._atr_period = atr_period
        self._frames: Dict[str, pd.DataFrame] = {}

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
        self._frames[symbol.upper()] = frame.copy()
        return self._frames[symbol.upper()]

    async def fetch_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch recent 1-minute OHLCV bars for a symbol via REST.

        Args:
            symbol: Ticker symbol.

        Returns:
            DataFrame with lowercase OHLCV columns and a timestamp column.

        Raises:
            RuntimeError: If Alpaca returns no bars.
        """
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            limit=self._bar_limit,
        )
        try:
            response = await asyncio.to_thread(self._client.get_stock_bars, request)
            bars = response.data.get(symbol, [])
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
        except Exception as exc:
            logger.error("[ERROR] Failed to fetch bars for %s: %s", symbol, exc)
            raise

    def compute_features(
        self,
        frame: pd.DataFrame,
        sentiment_score: float = 0.0,
    ) -> pd.DataFrame:
        """Compute RSI, MACD, and ATR features.

        Args:
            frame: OHLCV DataFrame.
            sentiment_score: Optional sentiment token in [-1.0, 1.0].

        Returns:
            DataFrame enriched with rsi, macd, atr, and sentiment_score columns.
        """
        if frame.empty:
            return frame
        enriched = compute_rsi_macd(frame.copy())
        enriched["atr"] = compute_atr(frame, length=self._atr_period)
        enriched["sentiment_score"] = sentiment_score
        return enriched

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
        feature_frame = self.compute_features(frame, sentiment_score=sentiment_score)
        if feature_frame.empty:
            return None
        row = feature_frame.iloc[-1]
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
