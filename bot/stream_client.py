"""Real-time Alpaca WebSocket bar streaming for stocks and crypto."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Awaitable, Callable, Deque, Dict, List, Optional

import pandas as pd
from alpaca.data.live import CryptoDataStream, StockDataStream

from bot.symbols import normalize_symbol, partition_symbols

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
        self._stock_symbols: List[str] = []
        self._crypto_symbols: List[str] = []
        self._stock_stream: Optional[StockDataStream] = None
        self._crypto_stream: Optional[CryptoDataStream] = None
        self._running = False
        self._bars: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=max_bars))

    @property
    def bars(self) -> Dict[str, Deque[dict]]:
        """Return the rolling bar buffers keyed by symbol."""
        return self._bars

    def get_frame(self, symbol: str) -> pd.DataFrame:
        """Return cached bars for a symbol as a DataFrame."""
        key = normalize_symbol(symbol)
        bucket = self._bars.get(key)
        if not bucket:
            return pd.DataFrame()
        return pd.DataFrame(list(bucket))

    async def _handle_bar(self, bar: object) -> None:
        """Normalize a stream bar and invoke the user callback."""
        symbol = normalize_symbol(str(getattr(bar, "symbol", "")))
        record = {
            "timestamp": getattr(bar, "timestamp", None).isoformat()
            if getattr(bar, "timestamp", None)
            else "",
            "open": float(getattr(bar, "open", 0)),
            "high": float(getattr(bar, "high", 0)),
            "low": float(getattr(bar, "low", 0)),
            "close": float(getattr(bar, "close", 0)),
            "volume": float(getattr(bar, "volume", 0)),
            "symbol": symbol,
        }
        self._bars[symbol].append(record)
        try:
            await self._on_bar(symbol, record)
        except Exception as exc:
            logger.error("[CRITICAL] Bar handler failed for %s: %s", symbol, exc)

    async def start(self, symbols: List[str]) -> None:
        """Subscribe to live bar streams and block until stopped.

        Args:
            symbols: Ticker symbols to subscribe (stocks and/or crypto).
        """
        self._stock_symbols, self._crypto_symbols = partition_symbols(symbols)
        await self.stop()
        self._running = True

        tasks: List[asyncio.Task[None]] = []
        if self._stock_symbols:
            tasks.append(asyncio.create_task(self._run_stock_stream(), name="stock_stream"))
        if self._crypto_symbols:
            tasks.append(asyncio.create_task(self._run_crypto_stream(), name="crypto_stream"))

        if not tasks:
            logger.error("[ERROR] No valid symbols provided for bar stream")
            return

        logger.info(
            "[INFO] Bar stream subscribed | stocks=%s | crypto=%s",
            self._stock_symbols,
            self._crypto_symbols,
        )
        await asyncio.gather(*tasks)

    async def _run_stock_stream(self) -> None:
        """Run the stock WebSocket stream."""
        self._stock_stream = StockDataStream(self._api_key, self._api_secret)
        for symbol in self._stock_symbols:
            self._stock_stream.subscribe_bars(self._handle_bar, symbol)
        await self._stock_stream._run_forever()

    async def _run_crypto_stream(self) -> None:
        """Run the crypto WebSocket stream."""
        self._crypto_stream = CryptoDataStream(self._api_key, self._api_secret)
        for symbol in self._crypto_symbols:
            self._crypto_stream.subscribe_bars(self._handle_bar, symbol)
        await self._crypto_stream._run_forever()

    async def stop(self) -> None:
        """Stop all WebSocket streams."""
        self._running = False
        if self._stock_stream is not None:
            try:
                await self._stock_stream.stop_ws()
            except Exception as exc:
                logger.error("[ERROR] Failed to stop stock bar stream: %s", exc)
            self._stock_stream = None
        if self._crypto_stream is not None:
            try:
                await self._crypto_stream.stop_ws()
            except Exception as exc:
                logger.error("[ERROR] Failed to stop crypto bar stream: %s", exc)
            self._crypto_stream = None
