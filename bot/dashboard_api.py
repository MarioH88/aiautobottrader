"""Alpaca helpers for the Streamlit dashboard using alpaca-py."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoSnapshotRequest, StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

from bot.symbols import is_crypto_symbol, normalize_symbol, partition_symbols

load_dotenv()
logger = logging.getLogger("dashboard")


def _paper_mode() -> bool:
    explicit = os.getenv("ALPACA_PAPER")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    return "paper" in base_url.lower()


class DashboardAlpaca:
    """Thin wrapper around alpaca-py for dashboard operations."""

    def __init__(self) -> None:
        api_key = os.getenv("APCA_API_KEY_ID", "")
        api_secret = os.getenv("APCA_API_SECRET_KEY", "")
        if not api_key or not api_secret:
            raise ValueError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in .env")
        self._trading = TradingClient(api_key, api_secret, paper=_paper_mode())
        self._stock_data = StockHistoricalDataClient(api_key, api_secret)
        self._crypto_data = CryptoHistoricalDataClient(api_key, api_secret)

    def get_account(self) -> Any:
        return self._trading.get_account()

    def list_positions(self) -> list[Any]:
        return list(self._trading.get_all_positions())

    def submit_market_order(self, symbol: str, side: str, qty: float) -> Any:
        symbol = normalize_symbol(symbol)
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if qty != int(qty) else TimeInForce.GTC
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
        )
        return self._trading.submit_order(request)

    def get_snapshots(self, symbols: list[str]) -> dict[str, Any]:
        stocks, crypto = partition_symbols(symbols)
        out: dict[str, Any] = {}
        if stocks:
            request = StockSnapshotRequest(symbol_or_symbols=stocks)
            snaps = self._stock_data.get_stock_snapshot(request)
            out.update(snaps)
        if crypto:
            request = CryptoSnapshotRequest(symbol_or_symbols=crypto)
            snaps = self._crypto_data.get_crypto_snapshot(request)
            out.update(snaps)
        return out


def snapshot_last_price(snap: Any) -> Optional[float]:
    trade = getattr(snap, "latest_trade", None)
    if trade is None:
        return None
    price = getattr(trade, "price", None)
    if price is None:
        price = getattr(trade, "p", None)
    return float(price) if price is not None else None


def snapshot_quote(snap: Any) -> tuple[Optional[float], Optional[float]]:
    quote = getattr(snap, "latest_quote", None)
    if quote is None:
        return None, None
    bid = getattr(quote, "bid_price", None)
    ask = getattr(quote, "ask_price", None)
    if bid is None:
        bid = getattr(quote, "bp", None)
    if ask is None:
        ask = getattr(quote, "ap", None)
    return (float(bid) if bid else None, float(ask) if ask else None)


def snapshot_prev_close(snap: Any) -> Optional[float]:
    bar = getattr(snap, "previous_daily_bar", None) or getattr(snap, "prev_daily_bar", None)
    if bar is None:
        return None
    close = getattr(bar, "close", None)
    if close is None:
        close = getattr(bar, "c", None)
    return float(close) if close is not None else None


def snapshot_trade_time(snap: Any) -> Optional[Any]:
    trade = getattr(snap, "latest_trade", None)
    if trade is None:
        return None
    return getattr(trade, "timestamp", None) or getattr(trade, "t", None)
