"""Dynamic portfolio allocation based on ML confidence and sentiment."""

from __future__ import annotations

import logging

from bot.schemas import MarketBarState, TradeSignal

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
            base_qty: Minimum share quantity when buying.
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
        if signal.action != "buy" or bar.close <= 0:
            return 0.0

        confidence_factor = max(0.5, signal.confidence) if signal.confidence >= self._min_confidence else 0.5
        sentiment_factor = 1.0 + (bar.sentiment_score * 0.25)
        rsi_factor = 1.0 if 30 <= bar.rsi <= 70 else 0.75

        budget_pct = signal.target_allocation_pct if signal.target_allocation_pct > 0 else self._max_allocation_pct
        budget_pct = min(budget_pct, self._max_allocation_pct)
        budget = buying_power * budget_pct * confidence_factor * sentiment_factor * rsi_factor
        qty = int(budget // bar.close)
        qty = max(int(self._base_qty), qty)

        logger.info(
            "[INFO] Allocation %s qty=%d budget=%.2f conf=%.2f sentiment=%.2f",
            bar.symbol,
            qty,
            budget,
            signal.confidence,
            bar.sentiment_score,
        )
        return float(qty)
