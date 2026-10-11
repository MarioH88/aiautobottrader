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

from bot.sentiment_models import build_scorer
from bot.symbols import canonical_symbol_key, news_aliases, normalize_symbol

logger = logging.getLogger("trading_bot")


class NewsSentimentService:
    """Streams and tokenizes news headlines into directional sentiment scores."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lookback_minutes: int = 30,
        max_headlines: int = 50,
        sentiment_model: str = "finbert",
    ) -> None:
        """Initialize the news sentiment service.

        Args:
            sentiment_model: ``"finbert"`` (default, falls back to VADER if unavailable) or ``"vader"``.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._lookback = timedelta(minutes=lookback_minutes)
        self._max_headlines = max_headlines
        self._scorer = build_scorer(sentiment_model)
        logger.info("[INFO] News sentiment scorer: %s", self._scorer.name)
        self._rest_client = NewsClient(api_key, api_secret)
        self._headlines: Dict[str, Deque[tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=max_headlines)
        )
        self._stream: Optional[NewsDataStream] = None
        self._symbols: List[str] = []
        self._running = False

    @property
    def scorer_name(self) -> str:
        """Name of the active scorer (finbert or vader)."""
        return self._scorer.name

    def score_text(self, text: str) -> float:
        """Score headline text and return a bounded sentiment in [-1, 1]."""
        if not text or not text.strip():
            return 0.0
        try:
            return self._scorer.score(text)
        except Exception as exc:
            logger.error("[ERROR] Sentiment scoring failed: %s", exc)
            return 0.0

    def _record_score(self, symbol: str, score: float, ts: Optional[datetime] = None) -> None:
        """Store a headline sentiment score under the canonical symbol key."""
        key = canonical_symbol_key(symbol)
        timestamp = ts or datetime.now(timezone.utc)
        self._headlines[key].append((timestamp, score))
        logger.info(
            "[INFO] News sentiment %s score=%.3f headlines=%d",
            key,
            score,
            len(self._headlines[key]),
        )

    def _purge_stale(self, symbol: str) -> None:
        """Remove headlines outside the lookback window."""
        key = canonical_symbol_key(symbol)
        cutoff = datetime.now(timezone.utc) - self._lookback
        bucket = self._headlines[key]
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()

    def get_sentiment(self, symbol: str) -> float:
        """Return the rolling average sentiment for a symbol (alias-aware)."""
        key = canonical_symbol_key(symbol)
        self._purge_stale(key)
        bucket = self._headlines[key]
        if not bucket:
            # Fall back across aliases if news was stored under a different tag.
            for alias in news_aliases(symbol):
                alias_key = canonical_symbol_key(alias)
                self._purge_stale(alias_key)
                other = self._headlines.get(alias_key)
                if other:
                    return sum(score for _, score in other) / len(other)
            return 0.0
        return sum(score for _, score in bucket) / len(bucket)

    @staticmethod
    def _unpack_news(news: object) -> tuple[list, str, Optional[datetime]]:
        """Extract (symbols, text, created_at) from a dict or SDK news object."""
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
        return list(symbols), f"{headline}. {summary}".strip(), created

    def _store(self, symbols: list, score: float, created: Optional[datetime]) -> None:
        timestamp = created or datetime.now(timezone.utc)
        for sym in symbols:
            # Store under every alias so BTC/USD lookups match BTCUSD tags.
            for alias in news_aliases(str(sym)):
                self._record_score(alias, score, timestamp)

    def _handle_news_item(self, news: object) -> None:
        """Process a single news payload from REST or WebSocket."""
        try:
            symbols, text, created = self._unpack_news(news)
            self._store(symbols, self.score_text(text), created)
        except Exception as exc:
            logger.error("[ERROR] Failed to process news item: %s", exc)

    def _handle_news_batch(self, items: list) -> None:
        """Score a batch of headlines in one pass (much faster for FinBERT)."""
        unpacked = []
        for item in items:
            try:
                unpacked.append(self._unpack_news(item))
            except Exception as exc:
                logger.error("[ERROR] Failed to unpack news item: %s", exc)
        if not unpacked:
            return
        try:
            scores = self._scorer.score_many([text for _, text, _ in unpacked])
        except Exception as exc:
            logger.error("[ERROR] Batch sentiment scoring failed: %s", exc)
            scores = [0.0] * len(unpacked)
        for (symbols, _, created), score in zip(unpacked, scores):
            self._store(symbols, score, created)

    async def fetch_recent(self, symbols: List[str]) -> None:
        """Poll recent headlines via REST as a bootstrap/fallback."""
        query_symbols: List[str] = []
        for symbol in symbols:
            query_symbols.extend(news_aliases(symbol))
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for sym in query_symbols:
            if sym not in seen:
                seen.add(sym)
                ordered.append(sym)
        request = NewsRequest(symbols=",".join(ordered), limit=50)
        try:
            response = await asyncio.to_thread(self._rest_client.get_news, request)
            data = getattr(response, "data", {}) or {}
            items = data.get("news", []) if isinstance(data, dict) else []
            self._handle_news_batch(list(items))
        except Exception as exc:
            logger.error("[ERROR] REST news fetch failed: %s", exc)

    async def start_stream(self, symbols: List[str]) -> None:
        """Start the Alpaca NewsDataStream WebSocket feed."""
        self._symbols = [normalize_symbol(s) for s in symbols]
        self._running = True
        await self.fetch_recent(self._symbols)
        await self.stop()

        async def _on_news(news: object) -> None:
            self._handle_news_item(news)

        stream_symbols: List[str] = []
        for symbol in self._symbols:
            stream_symbols.extend(news_aliases(symbol))
        self._stream = NewsDataStream(self._api_key, self._api_secret)
        self._stream.subscribe_news(_on_news, *sorted(set(stream_symbols)))
        logger.info("[INFO] News stream subscribed | symbols=%s", sorted(set(stream_symbols)))
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
