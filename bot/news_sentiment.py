"""Alpaca news ingestion and VADER sentiment scoring."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

from alpaca.data.historical.news import NewsClient
from alpaca.data.live import NewsDataStream
from alpaca.data.requests import NewsRequest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("trading_bot")


class NewsSentimentService:
    """Streams and tokenizes news headlines into directional sentiment scores."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lookback_minutes: int = 30,
        max_headlines: int = 50,
    ) -> None:
        """Initialize the news sentiment service.

        Args:
            api_key: Alpaca API key identifier.
            api_secret: Alpaca API secret key.
            lookback_minutes: Rolling window for headline relevance.
            max_headlines: Maximum headlines retained per symbol.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._lookback = timedelta(minutes=lookback_minutes)
        self._max_headlines = max_headlines
        self._analyzer = SentimentIntensityAnalyzer()
        self._rest_client = NewsClient(api_key, api_secret)
        self._headlines: Dict[str, Deque[tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=max_headlines)
        )
        self._stream: Optional[NewsDataStream] = None
        self._symbols: List[str] = []
        self._running = False

    def score_text(self, text: str) -> float:
        """Tokenize headline text and return a bounded sentiment score.

        Args:
            text: Raw headline or summary text.

        Returns:
            Sentiment score in [-1.0, 1.0] using VADER compound polarity.
        """
        if not text or not text.strip():
            return 0.0
        compound = self._analyzer.polarity_scores(text)["compound"]
        return max(-1.0, min(1.0, compound))

    def _record_score(self, symbol: str, score: float, ts: Optional[datetime] = None) -> None:
        """Store a headline sentiment score for a symbol."""
        timestamp = ts or datetime.now(timezone.utc)
        self._headlines[symbol.upper()].append((timestamp, score))
        logger.info(
            "[INFO] News sentiment %s score=%.3f headlines=%d",
            symbol.upper(),
            score,
            len(self._headlines[symbol.upper()]),
        )

    def _purge_stale(self, symbol: str) -> None:
        """Remove headlines outside the lookback window."""
        cutoff = datetime.now(timezone.utc) - self._lookback
        bucket = self._headlines[symbol]
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()

    def get_sentiment(self, symbol: str) -> float:
        """Return the rolling average sentiment for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Average sentiment in [-1.0, 1.0], or 0.0 if no headlines exist.
        """
        symbol = symbol.upper()
        self._purge_stale(symbol)
        bucket = self._headlines[symbol]
        if not bucket:
            return 0.0
        return sum(score for _, score in bucket) / len(bucket)

    def _handle_news_item(self, news: object) -> None:
        """Process a news payload from REST or WebSocket."""
        try:
            if isinstance(news, dict):
                symbols = news.get("symbols", []) or []
                headline = news.get("headline", "") or ""
                summary = news.get("summary", "") or ""
                created = news.get("created_at")
            else:
                symbols = getattr(news, "symbols", None) or []
                headline = getattr(news, "headline", "") or ""
                summary = getattr(news, "summary", "") or ""
                created = getattr(news, "created_at", None)
            text = f"{headline}. {summary}".strip()
            score = self.score_text(text)
            timestamp = created or datetime.now(timezone.utc)
            for sym in symbols:
                self._record_score(str(sym).upper(), score, timestamp)
        except Exception as exc:
            logger.error("[ERROR] Failed to process news item: %s", exc)

    async def fetch_recent(self, symbols: List[str]) -> None:
        """Poll recent headlines via REST as a bootstrap/fallback."""
        request = NewsRequest(symbols=",".join(symbols), limit=20)
        try:
            response = await asyncio.to_thread(self._rest_client.get_news, request)
            data = getattr(response, "data", {}) or {}
            items = data.get("news", []) if isinstance(data, dict) else []
            for item in items:
                self._handle_news_item(item)
        except Exception as exc:
            logger.error("[ERROR] REST news fetch failed: %s", exc)

    async def start_stream(self, symbols: List[str]) -> None:
        """Start the Alpaca NewsDataStream WebSocket feed.

        Args:
            symbols: Symbols to subscribe for news events.
        """
        self._symbols = [s.upper() for s in symbols]
        self._running = True
        await self.fetch_recent(self._symbols)
        await self.stop()

        async def _on_news(news: object) -> None:
            self._handle_news_item(news)

        self._stream = NewsDataStream(self._api_key, self._api_secret)
        self._stream.subscribe_news(_on_news, *self._symbols)
        logger.info("[INFO] News stream subscribed | symbols=%s", self._symbols)
        await self._stream._run_forever()

    async def stop(self) -> None:
        """Stop the news WebSocket stream."""
        self._running = False
        if self._stream is not None:
            try:
                await self._stream.stop_ws()
            except Exception as exc:
                logger.error("[ERROR] Failed to stop news stream: %s", exc)
            self._stream = None
