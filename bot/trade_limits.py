"""Simple daily trade caps based on trade_log.csv."""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path


def count_buys_today(log_file: str) -> int:
    """Count buy rows in the trade log for the current UTC date."""
    path = Path(log_file)
    if not path.exists():
        return 0
    today = date.today()
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("side") or row.get("action") or "").lower() != "buy":
                continue
            raw_ts = row.get("timestamp") or row.get("time") or ""
            if not raw_ts:
                continue
            try:
                if raw_ts.endswith("Z"):
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                else:
                    ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if ts.date() == today:
                count += 1
    return count
