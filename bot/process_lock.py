"""Prevent multiple bot processes (avoids duplicate WebSocket / order loops)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("trading_bot")


def _pid_alive(pid: int) -> bool:
    """Return True if a process id appears to be running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_process_lock(lock_path: str) -> bool:
    """Create or take over a stale lock file. Returns False if another live bot holds it."""
    path = Path(lock_path)
    if path.exists():
        try:
            existing = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing = -1
        if _pid_alive(existing):
            logger.error(
                "[CRITICAL] Another bot process is already running (pid=%s). "
                "Stop it before starting main.py again.",
                existing,
            )
            return False
        logger.warning("[ERROR] Removing stale bot lock (pid=%s)", existing)

    path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_process_lock(lock_path: str) -> None:
    """Remove the lock file if this process owns it."""
    path = Path(lock_path)
    if not path.exists():
        return
    try:
        owner = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        owner = -1
    if owner == os.getpid():
        path.unlink(missing_ok=True)
