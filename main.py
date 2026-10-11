"""Autonomous AI Trading Bot entry point."""

from __future__ import annotations

import asyncio
import logging

from bot.config import BotConfig
from bot.flag_file import read_flag, write_flag
from bot.logging_setup import configure_logging
from bot.process_lock import acquire_process_lock, release_process_lock
from bot.scheduler import describe_active_window, is_trading_window_active, load_scheduler_config
from bot.trading_bot import TradingBot

logger = logging.getLogger("trading_bot")


async def main() -> None:
    """Boot the trading bot and optionally request auto-start via the flag file."""
    config = BotConfig()
    configure_logging(config.runtime_log_file)
    bot = TradingBot(config)
    schedule = load_scheduler_config(config.scheduler_config_file)
    if not acquire_process_lock(config.process_lock_file):
        raise SystemExit(1)

    logger.info(
        "[INFO] Autonomous AI Trading Bot ready (paper=%s, auto_start=%s, symbols=%s)",
        config.paper,
        config.auto_start,
        config.symbols,
    )
    if not config.paper:
        logger.error(
            "[CRITICAL] LIVE trading account detected — use paper keys until you trust daily behavior"
        )
    logger.info("[INFO] %s", describe_active_window(schedule))

    if config.auto_start:
        current = read_flag(config.flag_file)
        if current == "emergency_stop":
            logger.error("[CRITICAL] AUTO_START skipped — emergency_stop flag is set")
        else:
            # Only write the flag; watch_flag() is the sole start/stop authority
            # so scheduler windows are always respected.
            write_flag(config.flag_file, "start")
            if schedule.get("enabled") and not is_trading_window_active(schedule):
                logger.info("[INFO] AUTO_START queued — waiting for trading window")
            else:
                logger.info("[INFO] AUTO_START requested — watch_flag will start the bot")

    try:
        await bot.watch_flag()
    finally:
        release_process_lock(config.process_lock_file)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[INFO] Shutdown requested by user")
