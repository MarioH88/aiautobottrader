"""Structured logging configuration for the trading bot."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class Iso8601Formatter(logging.Formatter):
    """Formatter that emits ISO-8601 timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Return UTC ISO-8601 timestamp."""
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def configure_logging(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging.

    Args:
        log_file: Path to the rolling runtime log file.
        level: Minimum log level.

    Returns:
        Configured root logger for the bot.
    """
    logger = logging.getLogger("trading_bot")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = Iso8601Formatter("[%(levelname)s] %(asctime)s %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_execution_tick(logger: logging.Logger, symbol: str) -> None:
    """Emit a distinctive execution tick marker."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(">>> [%s] EXECUTION TICK: %s <<<", now, symbol)
