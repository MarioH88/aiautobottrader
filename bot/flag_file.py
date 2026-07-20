"""UTF-8 safe bot control flag read/write (Windows PowerShell compatible)."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("trading_bot")

VALID_COMMANDS = {"start", "stop", "emergency_stop"}


def read_flag(flag_file: str) -> Optional[str]:
    """Read a control flag, tolerating UTF-8 and UTF-16 encodings from PowerShell."""
    if not os.path.exists(flag_file):
        return None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le"):
        try:
            with open(flag_file, "r", encoding=encoding) as handle:
                value = handle.read().strip().lower().lstrip("\ufeff")
                if value in VALID_COMMANDS:
                    return value
                if value:
                    logger.error("[ERROR] Unknown flag command: %s", value)
                    return None
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.error("[ERROR] Failed to read flag file %s: %s", flag_file, exc)
            return None
    logger.error("[ERROR] Failed to decode flag file %s with known encodings", flag_file)
    return None


def write_flag(flag_file: str, command: str) -> bool:
    """Write a control flag as UTF-8 without BOM."""
    command = command.strip().lower()
    if command not in VALID_COMMANDS:
        logger.error("[ERROR] Invalid flag command: %s", command)
        return False
    try:
        with open(flag_file, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(command)
        return True
    except Exception as exc:
        logger.error("[ERROR] Failed to write flag file %s: %s", flag_file, exc)
        return False
