"""Real-time Alpaca WebSocket bar streaming."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Awaitable, Callable, Deque, Dict, List, Optional

import pandas as pd
from alpaca.data.live import StockDataStream

logger = logging.getLogger("trading_bot")

BarHandler = Callable[[str, dict], Awaitable[None]]


class StreamClient:
    """Persistent WebSocket client for live 1-minute bar updates."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        on_bar: BarHandler,
        max_bars: int = 500,
    ) -> None:
        """Initialize the stream client.

        Args:
            api_key: Alpaca API key identifier.
            api_secret: Alpaca API secret key.
            on_bar: Async callback invoked as ``on_bar(symbol, bar_dict)``.
            max_bars: Rolling in-memory bar buffer size per symbol.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._on_bar = on_bar
        self._max_bars = max_bars
        self._symbols: List[str] = []
        self._stream: Optional[StockDataStream] = None
        self._running = False
        self._bars: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=max_bars))

    @property
    def bars(self) -> Dict[str, Deque[dict]]:
        """Return the rolling bar buffers keyed by symbol."""
        return self._bars

    def get_frame(self, symbol: str) -> pd.DataFrame:
        """Return cached bars for a symbol as a DataFrame."""
        bucket = self._bars.get(symbol.upper())
        if not bucket:
            return pd.DataFrame()
        return pd.DataFrame(list(bucket))

    async def start(self, symbols: List[str]) -> None:
        """Subscribe to live bar streams and block until stopped.

        Args:
            symbols: Ticker symbols to subscribe.
        """
        self._symbols = [s.upper() for s in symbols]
        await self.stop()
        self._running = True

        async def _async_bar_handler(bar: object) -> None:
            symbol = str(getattr(bar, "symbol", "")).upper()
            record = {
                "timestamp": getattr(bar, "timestamp", None).isoformat()
                if getattr(bar, "timestamp", None)
                else "",
                "open": float(getattr(bar, "open", 0)),
                "high": float(getattr(bar, "high", 0)),
                "low": float(getattr(bar, "low", 0)),
                "close": float(getattr(bar, "close", 0)),
                "volume": int(getattr(bar, "volume", 0)),
                "symbol": symbol,
            }
            self._bars[symbol].append(record)
            try:
                await self._on_bar(symbol, record)
            except Exception as exc:
                logger.error("[CRITICAL] Bar handler failed for %s: %s", symbol, exc)

        self._stream = StockDataStream(self._api_key, self._api_secret)
        for symbol in self._symbols:
            self._stream.subscribe_bars(_async_bar_handler, symbol)
        logger.info("[INFO] Bar stream subscribed | symbols=%s", self._symbols)
        await self._stream._run_forever()

    async def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        if self._stream is not None:
            try:
                await self._stream.stop_ws()
            except Exception as exc:
                logger.error("[ERROR] Failed to stop bar stream: %s", exc)
            self._stream = None
