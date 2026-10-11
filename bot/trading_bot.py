"""Async orchestrator for the autonomous trading bot."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional

from bot.config import BotConfig
from bot.connection_manager import ConnectionManager
from bot.data_pipeline import DataPipeline
from bot.execution_client import ExecutionClient
from bot.logging_setup import configure_logging, log_execution_tick
from bot.macro_sentiment import MacroSentimentService, combine_sentiment
from bot.news_sentiment import NewsSentimentService
from bot.flag_file import read_flag
from bot.schemas import MarketBarState
from bot.scheduler import (
    describe_active_window,
    is_trading_window_active,
    load_scheduler_config,
)
from bot.daily_status import write_bot_status
from bot.stream_client import StreamClient
from bot.strategy_engine import StrategyEngine

logger = logging.getLogger("trading_bot")


class TradingBot:
    """Event-driven trading daemon coordinating data, strategy, and execution."""

    def __init__(self, config: Optional[BotConfig] = None) -> None:
        """Initialize bot components.

        Args:
            config: Optional runtime configuration. Defaults to environment-backed BotConfig.
        """
        self.config = config or BotConfig()
        configure_logging(self.config.runtime_log_file)
        self.connection_manager = ConnectionManager(
            max_retries=self.config.max_retries,
            base_delay_sec=self.config.retry_base_delay_sec,
            ws_reconnect_delay_sec=self.config.ws_reconnect_delay_sec,
        )
        self.data_pipeline = DataPipeline(
            self.config.api_key,
            self.config.api_secret,
            bar_limit=self.config.bar_limit,
            atr_period=self.config.atr_period,
            timeframe_min=self.config.bar_timeframe_min,
            data_feed=self.config.data_feed,
        )
        self.strategy_engine = StrategyEngine(
            model_path=self.config.model_path,
            min_signal_confidence=self.config.min_signal_confidence,
        )
        self.execution_client = ExecutionClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            paper=self.config.paper,
            stop_loss_pct=self.config.stop_loss_pct,
            atr_multiplier=self.config.atr_multiplier,
            order_qty=self.config.order_qty,
            max_allocation_pct=self.config.max_allocation_pct,
            log_file=self.config.log_file,
            max_daily_buys=self.config.max_daily_buys,
        )
        self.macro_service: Optional[MacroSentimentService] = None
        if self.config.enable_macro_sentiment:
            self.macro_service = MacroSentimentService(
                csv_path=self.config.macro_sentiment_csv,
                source=self.config.macro_source,
            )
        self.news_service: Optional[NewsSentimentService] = None
        if self.config.enable_news:
            self.news_service = NewsSentimentService(
                self.config.api_key,
                self.config.api_secret,
                lookback_minutes=self.config.news_lookback_minutes,
                sentiment_model=self.config.sentiment_model,
            )
        self.stream_client = StreamClient(
            self.config.api_key,
            self.config.api_secret,
            on_bar=self._on_stream_bar,
            max_bars=self.config.bar_limit,
        )
        self._running = False
        self._worker_tasks: List[asyncio.Task[None]] = []
        self._main_task: Optional[asyncio.Task[None]] = None
        self._latest_bars: Dict[str, MarketBarState] = {}
        self._tick_lock = asyncio.Lock()
        self._websocket_enabled = self.config.use_websocket

    def _combined_sentiment(self, symbol: str) -> tuple[float, float]:
        """Return combined and macro sentiment scores for a symbol."""
        news_score = 0.0
        if self.news_service is not None:
            news_score = self.news_service.get_sentiment(symbol)
        macro_score = self.macro_service.get_score() if self.macro_service is not None else 0.0
        combined = combine_sentiment(
            news_score,
            macro_score,
            news_weight=self.config.sentiment_news_weight,
        )
        return combined, macro_score

    async def execution_tick(self, symbol: str, from_stream: bool = False) -> None:
        """Run one full ingest -> predict -> execute cycle for a symbol.

        Args:
            symbol: Ticker symbol to process.
            from_stream: Whether the trigger originated from a WebSocket bar.
        """
        async with self._tick_lock:
            log_execution_tick(logger, symbol)
            try:
                sentiment, macro = self._combined_sentiment(symbol)
                if from_stream:
                    frame = self.stream_client.get_frame(symbol)
                    if frame.empty:
                        frame = await self.connection_manager.run_with_retry(
                            lambda: self.data_pipeline.fetch_bars(symbol),
                            label=f"fetch_bars:{symbol}",
                        )
                    else:
                        self.data_pipeline.ingest_stream_frame(symbol, frame)
                else:
                    frame = await self.connection_manager.run_with_retry(
                        lambda: self.data_pipeline.fetch_bars(symbol),
                        label=f"fetch_bars:{symbol}",
                    )

                bar_state = self.data_pipeline.latest_bar_state(
                    frame,
                    symbol,
                    sentiment_score=sentiment,
                    macro_sentiment=macro,
                )
                if bar_state is None:
                    logger.error("[ERROR] Insufficient feature data for %s", symbol)
                    return

                self._latest_bars[symbol] = bar_state
                await self.connection_manager.run_with_retry(
                    self.execution_client.sync_positions,
                    label="sync_positions",
                )
                signal = self.strategy_engine.evaluate(bar_state)
                logger.info(
                    "[INFO] Signal %s %s confidence=%.2f sentiment=%.2f macro=%.2f alloc=%.1f%% reason=%s",
                    signal.action.upper(),
                    symbol,
                    signal.confidence,
                    sentiment,
                    macro,
                    signal.target_allocation_pct * 100,
                    signal.reason,
                )
                await self.connection_manager.run_with_retry(
                    lambda: self.execution_client.process_signal(signal, bar_state),
                    label=f"process_signal:{symbol}",
                )
                schedule = load_scheduler_config(self.config.scheduler_config_file)
                write_bot_status(
                    self.config.status_file,
                    paper=self.config.paper,
                    symbols=self.config.symbols,
                    running=self._running,
                    scheduler_enabled=bool(schedule.get("enabled")),
                    in_trading_window=is_trading_window_active(schedule),
                    flag_command=read_flag(self.config.flag_file) or "stop",
                    last_signal=signal,
                    last_bar=bar_state,
                )
            except Exception as exc:
                logger.error("[CRITICAL] Execution tick failed for %s: %s", symbol, exc)
                await self.execution_client.protect_open_positions(self._latest_bars)

    async def _on_stream_bar(self, symbol: str, _bar: dict) -> None:
        """Handle a live WebSocket bar event."""
        if not self._running:
            return
        await self.execution_tick(symbol, from_stream=True)

    async def _rest_fallback_loop(self) -> None:
        """Poll REST when WebSocket is disabled or disconnected."""
        interval = (
            self.config.tick_interval_sec
            if not self._websocket_enabled
            else self.config.rest_fallback_interval_sec
        )
        while self._running:
            if not self._websocket_enabled or not self.connection_manager.is_connected:
                logger.info("[INFO] REST polling active (every %ds)", interval)
                if not self.connection_manager.is_connected:
                    logger.error("[CRITICAL] REST fallback — protecting open positions")
                    await self.execution_client.protect_open_positions(self._latest_bars)
                for symbol in self.config.symbols:
                    if not self._running:
                        break
                    await self.execution_tick(symbol, from_stream=False)
            await asyncio.sleep(interval)
            if not self._running:
                break
            if self._websocket_enabled and self.connection_manager.is_connected:
                continue

    async def _news_poll_loop(self) -> None:
        """Poll news headlines via REST (no extra WebSocket)."""
        while self._running and self.news_service is not None:
            try:
                await self.news_service.fetch_recent(self.config.symbols)
            except Exception as exc:
                logger.error("[ERROR] News poll failed: %s", exc)
            await asyncio.sleep(self.config.news_poll_interval_sec)

    async def _run_bar_stream(self) -> None:
        """Supervise the live bar WebSocket."""
        await self.connection_manager.supervise(
            label="bar_stream",
            runner=lambda: self.stream_client.start(self.config.symbols),
            on_disconnect=self._on_stream_disconnect,
            on_connection_limit=self._disable_websocket,
        )

    async def _disable_websocket(self) -> None:
        """Fall back to REST-only mode when Alpaca connection slots are exhausted."""
        if not self._websocket_enabled:
            return
        self._websocket_enabled = False
        logger.error(
            "[CRITICAL] Disabling WebSocket — using REST polling only (set USE_WEBSOCKET=false in .env)"
        )
        await self.stream_client.stop()
        if not any(task.get_name() == "rest_fallback" for task in self._worker_tasks):
            task = asyncio.create_task(self._rest_fallback_loop(), name="rest_fallback")
            self._worker_tasks.append(task)

    async def _run_news_stream(self) -> None:
        """Supervise the live news WebSocket."""
        if self.news_service is None:
            return
        await self.connection_manager.supervise(
            label="news_stream",
            runner=lambda: self.news_service.start_stream(self.config.symbols),  # type: ignore[union-attr]
            on_disconnect=None,
        )

    async def _on_stream_disconnect(self) -> None:
        """Protect positions and cancel orphan orders when the bar stream drops."""
        logger.error("[CRITICAL] Bar stream disconnected — cancelling open orders and enforcing stops")
        await self.execution_client.cancel_open_orders()
        await self.stream_client.stop()
        await self.execution_client.protect_open_positions(self._latest_bars)

    async def run_loop(self) -> None:
        """Start Phase 2 event-driven services."""
        self._running = True
        logger.info(
            "[INFO] Trading bot started | symbols=%s | websocket=%s | news=%s | paper=%s",
            self.config.symbols,
            self._websocket_enabled,
            self.config.enable_news,
            self.config.paper,
        )

        # Bootstrap historical frames and news before streaming
        if self.news_service is not None:
            await self.news_service.fetch_recent(self.config.symbols)
        for symbol in self.config.symbols:
            try:
                await self.execution_tick(symbol, from_stream=False)
            except Exception as exc:
                logger.error("[ERROR] Bootstrap tick failed for %s: %s", symbol, exc)

        if self._websocket_enabled:
            self._worker_tasks.append(asyncio.create_task(self._run_bar_stream()))
        else:
            self._worker_tasks.append(
                asyncio.create_task(self._rest_fallback_loop(), name="rest_fallback")
            )

        if self.config.enable_news and self.news_service is not None:
            if self.config.use_news_websocket:
                self._worker_tasks.append(asyncio.create_task(self._run_news_stream()))
            else:
                self._worker_tasks.append(asyncio.create_task(self._news_poll_loop()))

        if self._websocket_enabled:
            self._worker_tasks.append(
                asyncio.create_task(self._rest_fallback_loop(), name="rest_fallback")
            )

        try:
            await asyncio.gather(*self._worker_tasks)
        except asyncio.CancelledError:
            logger.info("[INFO] Worker tasks cancelled")

    async def start(self) -> None:
        """Start the async event loop."""
        if self._main_task is not None and not self._main_task.done():
            return
        self._websocket_enabled = self.config.use_websocket
        if self._websocket_enabled and self.config.bar_timeframe_min != 1:
            # The stream delivers 1-minute bars; feeding them to a model trained on
            # BAR_TIMEFRAME_MIN bars would silently corrupt every feature.
            logger.error(
                "[ERROR] USE_WEBSOCKET ignored: model uses %d-minute bars but the stream is 1-minute. "
                "Using REST polling.",
                self.config.bar_timeframe_min,
            )
            self._websocket_enabled = False
        self._main_task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        """Stop all async services."""
        self._running = False
        self._websocket_enabled = False
        await self.stream_client.stop()
        if self.news_service is not None:
            await self.news_service.stop()
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        if self._main_task is not None:
            self._main_task.cancel()
            await asyncio.gather(self._main_task, return_exceptions=True)
            self._main_task = None
        logger.info("[INFO] Trading bot stopped")

    async def watch_flag(self) -> None:
        """Poll the flag file and start/stop the bot according to schedule windows."""
        last_cmd: Optional[str] = None
        logger.info(
            "[INFO] Flag watcher active | file=%s | %s",
            self.config.flag_file,
            describe_active_window(load_scheduler_config(self.config.scheduler_config_file)),
        )
        while True:
            schedule = load_scheduler_config(self.config.scheduler_config_file)
            flag_cmd = read_flag(self.config.flag_file)

            if flag_cmd == "emergency_stop":
                cmd = "emergency_stop"
            elif schedule.get("enabled") and not is_trading_window_active(schedule):
                # Outside the trading window, force a pause unless emergency already set.
                cmd = "stop" if flag_cmd == "start" else (flag_cmd or "stop")
            else:
                cmd = flag_cmd

            if cmd != last_cmd:
                if cmd == "start":
                    logger.info("[INFO] %s", describe_active_window(schedule))
                    await self.start()
                elif cmd in {"stop", "emergency_stop"}:
                    await self.stop()
                    if cmd == "emergency_stop":
                        logger.error("[CRITICAL] Emergency stop activated")
                    elif schedule.get("enabled") and not is_trading_window_active(schedule):
                        logger.info("[INFO] Outside trading window — bot paused")
                elif cmd:
                    logger.error("[ERROR] Unknown flag command: %s", cmd)
                last_cmd = cmd
            await asyncio.sleep(2)
