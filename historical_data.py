"""Historical OHLCV helpers for stocks and crypto (Alpaca + Yahoo)."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import yfinance as yf
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from bot.symbols import is_crypto_symbol, normalize_symbol

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

_TIMEFRAMES = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "1D": TimeFrame(1, TimeFrameUnit.Day),
}


def _bars_to_frame(bars: list, symbol: str) -> pd.DataFrame:
    """Convert Alpaca bar objects into a normalized DataFrame."""
    if not bars:
        return pd.DataFrame()
    rows = []
    for bar in bars:
        rows.append(
            {
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


def fetch_alpaca_history(symbol: str, timeframe: str = "1Min", limit: int = 1000) -> pd.DataFrame:
    """Fetch historical bars from Alpaca for stocks or crypto."""
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")

    symbol = normalize_symbol(symbol)
    tf = _TIMEFRAMES.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

    if is_crypto_symbol(symbol):
        client = CryptoHistoricalDataClient(API_KEY, API_SECRET)
        request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
        response = client.get_crypto_bars(request)
        bars = response.data.get(symbol, [])
        return _bars_to_frame(bars, symbol)

    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
    response = client.get_stock_bars(request)
    bars = response.data.get(symbol, [])
    return _bars_to_frame(bars, symbol)


def fetch_yahoo_history(
    symbol: str,
    interval: str = "1m",
    period: str = "5d",
) -> pd.DataFrame:
    """Fetch historical bars from Yahoo Finance."""
    yahoo_symbol = symbol
    if is_crypto_symbol(symbol):
        # BTC/USD -> BTC-USD for Yahoo
        yahoo_symbol = normalize_symbol(symbol).replace("/", "-")
    data = yf.download(yahoo_symbol, interval=interval, period=period, progress=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns={col: str(col).lower() for col in data.columns})
    data = data.reset_index()
    if "datetime" in data.columns and "timestamp" not in data.columns:
        data = data.rename(columns={"datetime": "timestamp"})
    if "date" in data.columns and "timestamp" not in data.columns:
        data = data.rename(columns={"date": "timestamp"})
    data["symbol"] = normalize_symbol(symbol)
    return data


if __name__ == "__main__":
    print("Alpaca 1min bars:")
    print(fetch_alpaca_history("AAPL", "1Min", 10))
    print("Alpaca crypto bars:")
    print(fetch_alpaca_history("BTC/USD", "1Min", 10))
    print("Yahoo daily bars:")
    print(fetch_yahoo_history("AAPL", "1d", "1mo").tail())
