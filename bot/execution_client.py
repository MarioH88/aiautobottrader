"""Alpaca REST order execution, ATR trailing stops, and position management."""

from __future__ import annotations

import asyncio
import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopOrderRequest

from bot.allocation_engine import AllocationEngine
from bot.schemas import MarketBarState, OrderResult, PositionState, TradeSignal
from bot.symbols import normalize_symbol, to_trading_symbol
from notify import send_discord

logger = logging.getLogger("trading_bot")


class ExecutionClient:
    """Submits orders and enforces ATR-adjusted trailing stop-loss rules."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        stop_loss_pct: float = 0.02,
        atr_multiplier: float = 2.0,
        order_qty: float = 1.0,
        max_allocation_pct: float = 0.10,
        log_file: str = "trade_log.csv",
    ) -> None:
        """Initialize the execution client."""
        self._client = TradingClient(api_key, api_secret, paper=paper)
        self._stop_loss_pct = stop_loss_pct
        self._atr_multiplier = atr_multiplier
        self._order_qty = order_qty
        self._allocator = AllocationEngine(base_qty=order_qty, max_allocation_pct=max_allocation_pct)
        self._log_file = log_file
        self._positions: Dict[str, PositionState] = {}
        self._discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    @property
    def positions(self) -> Dict[str, PositionState]:
        """Return the in-memory position matrix."""
        return self._positions

    def _notify(self, message: str) -> None:
        """Best-effort Discord notification."""
        if not self._discord_webhook:
            return
        try:
            send_discord(message, self._discord_webhook)
        except Exception as exc:
            logger.error("[ERROR] Discord notify failed: %s", exc)

    def _initial_stop(self, entry_price: float, atr: float) -> float:
        """Compute the wider of static or ATR-based initial stop."""
        static_stop = entry_price * (1.0 - self._stop_loss_pct)
        atr_stop = entry_price - (self._atr_multiplier * atr) if atr > 0 else static_stop
        return min(static_stop, atr_stop)

    async def sync_positions(self) -> None:
        """Refresh open positions from Alpaca and seed stop levels."""
        try:
            remote_positions = await asyncio.to_thread(self._client.get_all_positions)
            seen = set()
            for pos in remote_positions:
                symbol = normalize_symbol(pos.symbol)
                seen.add(symbol)
                entry = float(pos.avg_entry_price)
                qty = float(pos.qty)
                if symbol not in self._positions:
                    stop = entry * (1.0 - self._stop_loss_pct)
                    self._positions[symbol] = PositionState(
                        symbol=symbol,
                        entry_price=entry,
                        current_quantity=qty,
                        current_stop_loss=stop,
                        trailing_activation_price=entry,
                        atr_at_entry=0.0,
                    )
                    await self._ensure_broker_stop(symbol)
                else:
                    self._positions[symbol].current_quantity = qty
            for symbol in list(self._positions):
                if symbol not in seen:
                    await self._cancel_broker_stop(symbol)
                    del self._positions[symbol]
        except Exception as exc:
            logger.error("[ERROR] Failed to sync positions: %s", exc)

    async def get_buying_power(self) -> float:
        """Return available buying power from Alpaca."""
        try:
            account = await asyncio.to_thread(self._client.get_account)
            return float(account.buying_power)
        except Exception as exc:
            logger.error("[ERROR] Failed to fetch buying power: %s", exc)
            return 0.0

    async def cancel_open_orders(self, symbol: Optional[str] = None) -> None:
        """Cancel open orders to avoid orphan exposure after disconnects."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = await asyncio.to_thread(self._client.get_orders, filter=request)
            target = to_trading_symbol(symbol) if symbol else None
            for order in orders:
                if target and normalize_symbol(order.symbol) != normalize_symbol(target):
                    continue
                await asyncio.to_thread(self._client.cancel_order_by_id, order.id)
                logger.info("[INFO] Cancelled open order %s for %s", order.id, order.symbol)
        except Exception as exc:
            logger.error("[ERROR] Failed to cancel open orders: %s", exc)

    async def _cancel_broker_stop(self, symbol: str) -> None:
        """Cancel a tracked stop order if present."""
        position = self._positions.get(symbol)
        if position is None or not position.stop_order_id:
            return
        try:
            await asyncio.to_thread(self._client.cancel_order_by_id, position.stop_order_id)
            logger.info("[INFO] Cancelled broker stop %s for %s", position.stop_order_id, symbol)
        except Exception as exc:
            logger.error("[ERROR] Failed cancelling stop for %s: %s", symbol, exc)
        position.stop_order_id = None

    async def _place_broker_stop(self, symbol: str, qty: float, stop_price: float) -> Optional[str]:
        """Submit a stop-loss sell order to Alpaca."""
        trading_symbol = to_trading_symbol(symbol)
        rounded = round(float(stop_price), 2)
        tif = TimeInForce.DAY if qty != int(qty) else TimeInForce.GTC
        request = StopOrderRequest(
            symbol=trading_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=tif,
            stop_price=rounded,
        )
        try:
            order = await asyncio.to_thread(self._client.submit_order, request)
            logger.info(
                "[INFO] Broker stop placed %s stop=%.2f qty=%.6f id=%s",
                trading_symbol,
                rounded,
                qty,
                order.id,
            )
            return str(order.id)
        except Exception as exc:
            logger.error(
                "[ERROR] Broker stop rejected for %s (software stop remains active): %s",
                trading_symbol,
                exc,
            )
            return None

    async def _ensure_broker_stop(self, symbol: str) -> None:
        """Ensure an open position has a broker stop order."""
        position = self._positions.get(symbol)
        if position is None or position.current_quantity <= 0:
            return
        if position.stop_order_id:
            return
        order_id = await self._place_broker_stop(
            symbol, position.current_quantity, position.current_stop_loss
        )
        position.stop_order_id = order_id

    async def _replace_broker_stop(self, symbol: str) -> None:
        """Cancel and replace the broker stop when the trail ratchets up."""
        position = self._positions.get(symbol)
        if position is None or position.current_quantity <= 0:
            return
        await self._cancel_broker_stop(symbol)
        position.stop_order_id = await self._place_broker_stop(
            symbol, position.current_quantity, position.current_stop_loss
        )

    async def update_trailing_stop(self, bar: MarketBarState) -> None:
        """Ratchet the trailing stop using ATR volatility adjustment."""
        position = self._positions.get(bar.symbol)
        if position is None or position.current_quantity <= 0:
            return

        if bar.close > position.trailing_activation_price:
            position.trailing_activation_price = bar.close
            atr = bar.atr if bar.atr > 0 else position.atr_at_entry
            if atr > 0:
                atr_stop = bar.close - (self._atr_multiplier * atr)
                new_stop = max(position.current_stop_loss, atr_stop)
                if new_stop > position.current_stop_loss + 1e-6:
                    position.current_stop_loss = new_stop
                    logger.info(
                        "[INFO] Trailing stop updated %s stop=%.2f close=%.2f atr=%.4f",
                        bar.symbol,
                        position.current_stop_loss,
                        bar.close,
                        atr,
                    )
                    await self._replace_broker_stop(bar.symbol)

    def _append_trade_log(
        self,
        symbol: str,
        action: str,
        price: float,
        qty: float,
        reason: str,
    ) -> None:
        """Append a trade record to the CSV log."""
        exists = os.path.isfile(self._log_file)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "qty": qty,
            "reason": reason,
        }
        try:
            with open(self._log_file, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            logger.error("[CRITICAL] Failed to write trade log: %s", exc)

    async def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        reason: str,
        reference_price: float,
        atr: float = 0.0,
    ) -> OrderResult:
        """Submit a market order to Alpaca and manage broker stops."""
        symbol = normalize_symbol(symbol)
        trading_symbol = to_trading_symbol(symbol)
        tif = TimeInForce.DAY if qty != int(qty) else TimeInForce.GTC
        request = MarketOrderRequest(
            symbol=trading_symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
        )
        try:
            order = await asyncio.to_thread(self._client.submit_order, request)
            action = "buy" if side == OrderSide.BUY else "sell"
            self._append_trade_log(symbol, action, reference_price, qty, reason)
            logger.info(
                "[INFO] Order submitted: %s %s x%.4f @ ~%.2f (%s)",
                action.upper(),
                symbol,
                qty,
                reference_price,
                reason,
            )
            self._notify(f"{action.upper()} {qty:.6f} {symbol} @ ~{reference_price:.2f} ({reason})")
            if side == OrderSide.BUY:
                stop = self._initial_stop(reference_price, atr)
                stop_id = await self._place_broker_stop(symbol, qty, stop)
                self._positions[symbol] = PositionState(
                    symbol=symbol,
                    entry_price=reference_price,
                    current_quantity=qty,
                    current_stop_loss=stop,
                    trailing_activation_price=reference_price,
                    atr_at_entry=atr,
                    stop_order_id=stop_id,
                )
            elif symbol in self._positions:
                await self._cancel_broker_stop(symbol)
                del self._positions[symbol]
            return OrderResult(
                success=True,
                symbol=symbol,
                side=action,
                quantity=qty,
                price=reference_price,
                order_id=str(order.id),
            )
        except Exception as exc:
            logger.error("[CRITICAL] Order submission failed for %s: %s", symbol, exc)
            self._notify(f"ORDER FAILED {symbol}: {exc}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                quantity=qty,
                error=str(exc),
            )

    async def check_stop_loss(self, bar: MarketBarState) -> Optional[OrderResult]:
        """Evaluate ATR trailing or static stop-loss for an open position."""
        await self.update_trailing_stop(bar)
        position = self._positions.get(bar.symbol)
        if position is None or position.current_quantity <= 0:
            return None
        if bar.close <= position.current_stop_loss:
            logger.error(
                "[CRITICAL] Stop-loss triggered for %s at %.2f (stop=%.2f)",
                bar.symbol,
                bar.close,
                position.current_stop_loss,
            )
            await self._cancel_broker_stop(bar.symbol)
            return await self.submit_market_order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                qty=position.current_quantity,
                reason="atr-trailing-stop",
                reference_price=bar.close,
            )
        return None

    async def process_signal(self, signal: TradeSignal, bar: MarketBarState) -> Optional[OrderResult]:
        """Translate a strategy signal into an order when appropriate."""
        stop_result = await self.check_stop_loss(bar)
        if stop_result is not None:
            return stop_result

        held = self._positions.get(bar.symbol)
        held_qty = held.current_quantity if held else 0.0
        buying_power = await self.get_buying_power()

        if signal.action == "buy" and held_qty <= 0:
            qty = self._allocator.target_quantity(signal, bar, buying_power, held_qty)
            if qty <= 0:
                logger.info("[INFO] Insufficient buying power for %s", bar.symbol)
                return None
            return await self.submit_market_order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                qty=qty,
                reason=signal.reason,
                reference_price=bar.close,
                atr=bar.atr,
            )
        if signal.action == "sell" and held_qty > 0:
            return await self.submit_market_order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                qty=held_qty,
                reason=signal.reason,
                reference_price=bar.close,
            )
        logger.info("[INFO] Hold signal for %s (reason=%s)", bar.symbol, signal.reason)
        return None

    async def protect_open_positions(self, bars: Dict[str, MarketBarState]) -> None:
        """Ensure open positions remain protected during connectivity loss."""
        await self.sync_positions()
        await self.cancel_open_orders()
        for symbol, position in list(self._positions.items()):
            if position.current_quantity <= 0:
                continue
            # Mass cancel cleared broker stops — re-arm them.
            position.stop_order_id = None
            await self._ensure_broker_stop(symbol)
            bar = bars.get(symbol)
            if bar is None:
                logger.error(
                    "[CRITICAL] No live bar for open position %s — stop=%.2f",
                    symbol,
                    position.current_stop_loss,
                )
                continue
            await self.check_stop_loss(bar)
