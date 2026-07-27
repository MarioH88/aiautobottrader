"""Dynamic portfolio allocation based on ML confidence and sentiment."""

from __future__ import annotations

import logging
import math

from bot.schemas import MarketBarState, TradeSignal
from bot.symbols import is_crypto_symbol

logger = logging.getLogger("trading_bot")


class AllocationEngine:
    """Computes position size from signal confidence, sentiment, and buying power."""

    def __init__(
        self,
        base_qty: float = 1.0,
        max_allocation_pct: float = 0.10,
        min_confidence: float = 0.55,
    ) -> None:
        """Initialize allocation parameters.

        Args:
            base_qty: Preferred share quantity when buying.
            max_allocation_pct: Max fraction of buying power per symbol.
            min_confidence: Minimum ML confidence required to size up.
        """
        self._base_qty = base_qty
        self._max_allocation_pct = max_allocation_pct
        self._min_confidence = min_confidence

    def target_quantity(
        self,
        signal: TradeSignal,
        bar: MarketBarState,
        buying_power: float,
        held_qty: float,
    ) -> float:
        """Calculate dynamic order quantity for a signal.

        Supports fractional shares when buying power cannot cover a whole share
        (common for crypto and low-balance accounts).

        Args:
            signal: Strategy output with confidence and action.
            bar: Latest market bar including sentiment and price.
            buying_power: Available cash/buying power from Alpaca.
            held_qty: Currently held share quantity.

        Returns:
            Share quantity to buy or sell.
        """
        if signal.action == "sell":
            return held_qty
        if signal.action != "buy" or bar.close <= 0 or buying_power <= 0:
            return 0.0

        confidence_factor = max(0.5, signal.confidence) if signal.confidence >= self._min_confidence else 0.5
        sentiment_factor = 1.0 + (bar.sentiment_score * 0.25)
        rsi_factor = 1.0 if 30 <= bar.rsi <= 70 else 0.75

        budget_pct = signal.target_allocation_pct if signal.target_allocation_pct > 0 else self._max_allocation_pct
        budget_pct = min(budget_pct, self._max_allocation_pct)
        budget = buying_power * budget_pct * confidence_factor * sentiment_factor * rsi_factor
        budget = min(budget, buying_power)

        allow_fractional = is_crypto_symbol(bar.symbol) or budget < bar.close
        if allow_fractional:
            qty = round(budget / bar.close, 6)
        else:
            qty = float(max(int(self._base_qty), int(budget // bar.close)))

        if qty <= 0:
            return 0.0
        if qty * bar.close > buying_power:
            qty = round(buying_power / bar.close, 6)
        if qty <= 0 or math.isclose(qty, 0.0):
            return 0.0

        logger.info(
            "[INFO] Allocation %s qty=%.6f budget=%.2f conf=%.2f sentiment=%.2f",
            bar.symbol,
            qty,
            budget,
            signal.confidence,
            bar.sentiment_score,
        )
        return float(qty)
