"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from bot.symbols import normalize_symbol

# Always load .env from the project root (aiautobottrader/), not parent folders.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str) -> Optional[float]:
    """Parse an optional float environment variable (empty/missing -> None)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class BotConfig(BaseModel):
    """Bot configuration schema."""

    api_key: str = Field(default_factory=lambda: os.getenv("APCA_API_KEY_ID", ""))
    api_secret: str = Field(default_factory=lambda: os.getenv("APCA_API_SECRET_KEY", ""))
    paper: bool = Field(default_factory=lambda: _is_paper_trading())
    symbols: List[str] = Field(default_factory=lambda: _parse_symbols())
    model_path: str = "model.pkl"
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))
    order_qty: float = float(os.getenv("ORDER_QTY", "1.0"))
    # Bars per REST pull; must cover the feature warm-up (sma50) plus headroom.
    bar_limit: int = int(os.getenv("BAR_LIMIT", "200"))
    bar_timeframe_min: int = int(os.getenv("BAR_TIMEFRAME_MIN", "5"))
    data_feed: str = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
    tick_interval_sec: int = int(os.getenv("TICK_INTERVAL_SEC", "60"))
    log_file: str = "trade_log.csv"
    runtime_log_file: str = "bot_runtime.log"
    flag_file: str = "bot_control.flag"
    auto_start: bool = Field(default_factory=lambda: _env_bool("AUTO_START", True))

    # Phase 2
    use_websocket: bool = Field(default_factory=lambda: _env_bool("USE_WEBSOCKET", False))
    enable_news: bool = Field(default_factory=lambda: _env_bool("ENABLE_NEWS", True))
    use_news_websocket: bool = Field(default_factory=lambda: _env_bool("USE_NEWS_WEBSOCKET", False))
    news_poll_interval_sec: int = int(os.getenv("NEWS_POLL_INTERVAL_SEC", "300"))
    ws_reconnect_delay_sec: float = float(os.getenv("WS_RECONNECT_DELAY_SEC", "120"))
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
    atr_multiplier: float = float(os.getenv("ATR_MULTIPLIER", "2.0"))
    news_lookback_minutes: int = int(os.getenv("NEWS_LOOKBACK_MINUTES", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "5"))
    retry_base_delay_sec: float = float(os.getenv("RETRY_BASE_DELAY_SEC", "1.0"))
    rest_fallback_interval_sec: int = int(os.getenv("REST_FALLBACK_INTERVAL_SEC", "120"))

    # Sentiment + allocation
    enable_macro_sentiment: bool = Field(default_factory=lambda: _env_bool("ENABLE_MACRO_SENTIMENT", True))
    macro_source: str = os.getenv("MACRO_SOURCE", "fred").strip().lower()
    macro_sentiment_csv: str = os.getenv("MACRO_SENTIMENT_CSV", "macro_sentiment.csv")
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "finbert").strip().lower()
    sentiment_news_weight: float = float(os.getenv("SENTIMENT_NEWS_WEIGHT", "0.6"))
    max_allocation_pct: float = float(os.getenv("MAX_ALLOCATION_PCT", "0.10"))
    ml_model_type: str = os.getenv("ML_MODEL_TYPE", "xgboost")
    # None -> use the threshold recommended by train_model.py (stored in model.pkl).
    min_signal_confidence: Optional[float] = Field(default_factory=lambda: _env_float("MIN_SIGNAL_CONFIDENCE"))
    max_daily_buys: int = int(os.getenv("MAX_DAILY_BUYS", "20"))
    scheduler_config_file: str = os.getenv("SCHEDULER_CONFIG_FILE", "scheduler_config.json")
    status_file: str = os.getenv("BOT_STATUS_FILE", "bot_status.json")
    process_lock_file: str = os.getenv("BOT_PROCESS_LOCK", "bot.pid")

    @field_validator("api_key", "api_secret")
    @classmethod
    def require_credentials(cls, value: str) -> str:
        """Ensure Alpaca credentials are present."""
        if not value:
            raise ValueError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in .env")
        return value


def _is_paper_trading() -> bool:
    """Detect paper vs live mode from env or base URL."""
    explicit = os.getenv("ALPACA_PAPER")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    return "paper" in base_url.lower()


def _parse_symbols() -> List[str]:
    """Parse comma-separated symbols from the environment."""
    raw = os.getenv("TRADING_SYMBOLS", "AAPL,TSLA")
    return [normalize_symbol(symbol) for symbol in raw.split(",") if symbol.strip()]
