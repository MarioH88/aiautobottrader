"""Resilient network wrapper with retry, rate-limit, and recovery logic."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger("trading_bot")

T = TypeVar("T")


class ConnectionManager:
    """Manages retries, HTTP 429 backoff, and WebSocket recovery."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 60.0,
        ws_reconnect_delay_sec: float = 120.0,
    ) -> None:
        """Initialize connection manager thresholds.

        Args:
            max_retries: Maximum retry attempts before surfacing failure.
            base_delay_sec: Initial exponential backoff delay.
            max_delay_sec: Backoff ceiling in seconds.
            ws_reconnect_delay_sec: Delay after WebSocket connection-limit errors.
        """
        self.max_retries = max_retries
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self.ws_reconnect_delay_sec = ws_reconnect_delay_sec
        self._connected = True
        self._rate_limited_until: float = 0.0

    @property
    def is_connected(self) -> bool:
        """Return whether the manager considers the network healthy."""
        return self._connected

    def mark_disconnected(self, reason: str) -> None:
        """Mark the network as degraded."""
        self._connected = False
        logger.error("[CRITICAL] Connection lost: %s", reason)

    def mark_connected(self) -> None:
        """Mark the network as healthy."""
        if not self._connected:
            logger.info("[INFO] Connection restored")
        self._connected = True

    @staticmethod
    def is_rate_limit_error(exc: BaseException) -> bool:
        """Detect HTTP 429 or Alpaca rate-limit responses."""
        message = str(exc).lower()
        return (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
            or "40110000" in message
        )

    @staticmethod
    def is_connection_limit_error(exc: BaseException) -> bool:
        """Detect Alpaca WebSocket connection limit errors."""
        message = str(exc).lower()
        return "connection limit" in message or "conn limit" in message

    def _backoff_delay(self, attempt: int, rate_limited: bool = False) -> float:
        """Compute delay for the given attempt."""
        multiplier = 3.0 if rate_limited else 2.0
        delay = min(self.base_delay_sec * (multiplier ** attempt), self.max_delay_sec)
        return delay

    async def run_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        label: str,
    ) -> T:
        """Execute an async operation with exponential backoff.

        Args:
            operation: Zero-argument async callable.
            label: Human-readable label for logs.

        Returns:
            Result of the operation.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                result = await operation()
                self.mark_connected()
                return result
            except Exception as exc:
                last_exc = exc
                rate_limited = self.is_rate_limit_error(exc)
                delay = self._backoff_delay(attempt, rate_limited=rate_limited)
                level = "[CRITICAL]" if rate_limited else "[ERROR]"
                logger.error(
                    "%s %s failed (attempt %d/%d): %s — retry in %.1fs",
                    level,
                    label,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                if rate_limited:
                    self.mark_disconnected(f"rate limit during {label}")
                await asyncio.sleep(delay)
        raise RuntimeError(f"{label} failed after {self.max_retries} retries: {last_exc}")

    async def supervise(
        self,
        label: str,
        runner: Callable[[], Awaitable[None]],
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_connection_limit: Optional[Callable[[], Awaitable[None]]] = None,
        max_connection_limit_retries: int = 2,
    ) -> None:
        """Continuously run a long-lived task and recover from failures.

        Args:
            label: Task label for logging.
            runner: Async callable that runs until it raises or completes.
            on_disconnect: Optional cleanup invoked before reconnect attempts.
            on_connection_limit: Optional handler when connection limit is exhausted.
            max_connection_limit_retries: Stop retrying after this many limit errors.
        """
        limit_retries = 0
        while True:
            try:
                logger.info("[INFO] Starting supervised task: %s", label)
                await runner()
                limit_retries = 0
            except asyncio.CancelledError:
                logger.info("[INFO] Supervised task cancelled: %s", label)
                raise
            except Exception as exc:
                self.mark_disconnected(f"{label}: {exc}")
                logger.error("[CRITICAL] %s crashed: %s", label, exc)
                if on_disconnect:
                    try:
                        await on_disconnect()
                    except Exception as cleanup_exc:
                        logger.error("[ERROR] Disconnect handler failed: %s", cleanup_exc)
                if self.is_connection_limit_error(exc):
                    limit_retries += 1
                    if limit_retries >= max_connection_limit_retries:
                        logger.error(
                            "[CRITICAL] %s hit connection limit %d times — stopping WebSocket retries",
                            label,
                            limit_retries,
                        )
                        if on_connection_limit:
                            try:
                                await on_connection_limit()
                            except Exception as handler_exc:
                                logger.error("[ERROR] Connection-limit handler failed: %s", handler_exc)
                        return
                    delay = self.ws_reconnect_delay_sec
                    logger.error(
                        "[CRITICAL] WebSocket connection limit hit (%d/%d) — waiting %.0fs before retry",
                        limit_retries,
                        max_connection_limit_retries,
                        delay,
                    )
                else:
                    delay = self._backoff_delay(0)
                logger.info("[INFO] Reconnecting %s in %.1fs", label, delay)
                await asyncio.sleep(delay)

    @staticmethod
    def parse_retry_after(exc: BaseException) -> Optional[float]:
        """Extract Retry-After seconds from an error message if present."""
        match = re.search(r"retry[- ]after[:\s]+(\d+)", str(exc), re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
