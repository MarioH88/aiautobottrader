"""Trading schedule windows for automated start/stop."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

logger = logging.getLogger("trading_bot")

DEFAULT_SCHEDULER_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "timezone": "America/New_York",
    "windows": [
        {
            "days": ["mon", "tue", "wed", "thu", "fri"],
            "start": "09:30",
            "end": "16:00",
        }
    ],
}

DAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def default_scheduler_path(project_root: Optional[Path] = None) -> Path:
    """Return the default scheduler config path."""
    root = project_root or Path(__file__).resolve().parent.parent
    return root / "scheduler_config.json"


def load_scheduler_config(path: Path | str) -> Dict[str, Any]:
    """Load scheduler config from disk, falling back to defaults."""
    config_path = Path(path)
    if not config_path.exists():
        return dict(DEFAULT_SCHEDULER_CONFIG)
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        merged = dict(DEFAULT_SCHEDULER_CONFIG)
        merged.update(data)
        if not merged.get("windows"):
            merged["windows"] = list(DEFAULT_SCHEDULER_CONFIG["windows"])
        return merged
    except Exception as exc:
        logger.error("[ERROR] Failed to load scheduler config %s: %s", config_path, exc)
        return dict(DEFAULT_SCHEDULER_CONFIG)


def save_scheduler_config(path: Path | str, config: Dict[str, Any]) -> bool:
    """Persist scheduler config to disk."""
    config_path = Path(path)
    try:
        payload = dict(DEFAULT_SCHEDULER_CONFIG)
        payload.update(config)
        with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return True
    except Exception as exc:
        logger.error("[ERROR] Failed to save scheduler config %s: %s", config_path, exc)
        return False


def _parse_hhmm(value: str) -> tuple[int, int]:
    """Parse HH:MM into hour and minute integers."""
    hour, minute = value.strip().split(":", 1)
    return int(hour), int(minute)


def validate_hhmm(value: str) -> bool:
    """Return True when value is a valid HH:MM clock time."""
    try:
        hour, minute = _parse_hhmm(value)
        return 0 <= hour <= 23 and 0 <= minute <= 59
    except Exception:
        return False


def _window_days(window: Dict[str, Any]) -> set[int]:
    """Return weekday indices (Mon=0) enabled for a window."""
    days: set[int] = set()
    for day in window.get("days", []):
        key = str(day).strip().lower()
        if key in DAY_ALIASES:
            days.add(DAY_ALIASES[key])
    return days


def _minutes_in_window(now_minutes: int, start_minutes: int, end_minutes: int) -> bool:
    """Support same-day and overnight (wrap) windows."""
    if start_minutes == end_minutes:
        return True
    if start_minutes < end_minutes:
        return start_minutes <= now_minutes < end_minutes
    # Overnight: e.g. 22:00 -> 06:00
    return now_minutes >= start_minutes or now_minutes < end_minutes


def is_trading_window_active(
    config: Dict[str, Any],
    now: Optional[datetime] = None,
) -> bool:
    """Return True when the current time falls inside a configured window."""
    if not config.get("enabled"):
        return True

    timezone_name = str(config.get("timezone", "America/New_York"))
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        logger.error("[ERROR] Invalid scheduler timezone %s — using UTC", timezone_name)
        tz = ZoneInfo("UTC")

    current = now.astimezone(tz) if now is not None else datetime.now(tz)
    windows: List[Dict[str, Any]] = config.get("windows", [])
    if not windows:
        return True

    for window in windows:
        days = _window_days(window)
        if days and current.weekday() not in days:
            continue
        start_h, start_m = _parse_hhmm(str(window.get("start", "09:30")))
        end_h, end_m = _parse_hhmm(str(window.get("end", "16:00")))
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        now_minutes = current.hour * 60 + current.minute
        if _minutes_in_window(now_minutes, start_minutes, end_minutes):
            return True
    return False


def describe_active_window(config: Dict[str, Any]) -> str:
    """Human-readable summary of scheduler state."""
    if not config.get("enabled"):
        return "Scheduler disabled — bot may run anytime"
    active = is_trading_window_active(config)
    status = "inside trading window" if active else "outside trading window"
    return f"Scheduler enabled — {status} ({config.get('timezone', 'America/New_York')})"
