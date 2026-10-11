"""Write a small JSON snapshot for the dashboard and daily ops checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from bot.schemas import MarketBarState, TradeSignal


def write_bot_status(
    path: str,
    *,
    paper: bool,
    symbols: list[str],
    running: bool,
    scheduler_enabled: bool,
    in_trading_window: bool,
    flag_command: str,
    last_signal: Optional[TradeSignal] = None,
    last_bar: Optional[MarketBarState] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the latest bot heartbeat and last signal to disk."""
    payload: Dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "paper": paper,
        "symbols": symbols,
        "running": running,
        "scheduler_enabled": scheduler_enabled,
        "in_trading_window": in_trading_window,
        "flag_command": flag_command,
    }
    if last_signal is not None:
        payload["last_signal"] = {
            "symbol": last_signal.symbol,
            "action": last_signal.action,
            "confidence": last_signal.confidence,
            "reason": last_signal.reason,
        }
    if last_bar is not None:
        payload["last_bar"] = {
            "symbol": last_bar.symbol,
            "close": last_bar.close,
            "rsi": last_bar.rsi,
            "timestamp": last_bar.timestamp,
        }
    if extra:
        payload.update(extra)

    target = Path(path)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
