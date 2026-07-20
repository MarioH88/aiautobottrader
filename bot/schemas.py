"""Pydantic schemas for market state and position tracking."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketBarState(BaseModel):
    """Validated market feature state for a single bar."""

    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    rsi: float
    macd: float
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    macro_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    atr: float = Field(default=0.0, ge=0.0)

    @field_validator("volume", mode="before")
    @classmethod
    def coerce_volume(cls, value: object) -> int:
        """Coerce volume to integer."""
        return int(float(value))


class PositionState(BaseModel):
    """Active position matrix tracked in memory."""

    symbol: str
    entry_price: float
    current_quantity: float
    current_stop_loss: float
    trailing_activation_price: float
    atr_at_entry: float = 0.0


class TradeSignal(BaseModel):
    """Directional signal emitted by the strategy engine."""

    symbol: str
    action: str  # "buy", "sell", or "hold"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    target_allocation_pct: float = Field(default=0.0, ge=0.0, le=1.0)


class OrderResult(BaseModel):
    """Result of an order submission attempt."""

    success: bool
    symbol: str
    side: str
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    error: Optional[str] = None
