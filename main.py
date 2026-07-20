"""Autonomous AI Trading Bot entry point."""

from __future__ import annotations

import asyncio
import logging

from bot.config import BotConfig
from bot.flag_file import write_flag
from bot.logging_setup import configure_logging
from bot.trading_bot import TradingBot

logger = logging.getLogger("trading_bot")


async def main() -> None:
    """Boot the trading bot and optionally auto-start execution."""
    config = BotConfig()
    configure_logging(config.runtime_log_file)
    bot = TradingBot(config)
    logger.info(
        "[INFO] Autonomous AI Trading Bot ready (paper=%s, auto_start=%s, symbols=%s)",
        config.paper,
        config.auto_start,
        config.symbols,
    )

    if config.auto_start:
        write_flag(config.flag_file, "start")
        await bot.start()
        logger.info("[INFO] Auto-trading enabled — bot started automatically")

    await bot.watch_flag()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[INFO] Shutdown requested by user")
