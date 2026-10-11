"""Build train_data.csv with the full feature set and ATR-scaled 3-class labels.

Sources
  TRAIN_DATA_SOURCE=alpaca   (default) Alpaca IEX bars, up to several years of intraday history
  TRAIN_DATA_SOURCE=yfinance Yahoo Finance; intraday history is capped by Yahoo (5m: 60d, 1h: 730d)

Each bar is joined to the FRED macro score for its date so the model learns the macro regime.
News sentiment is 0 for history (no per-bar headline archive) and only varies live.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.config import BotConfig
from bot.features import FEATURE_COLUMNS, LABEL_NAMES, build_features, make_labels
from bot.macro_fred import MacroFredService
from bot.symbols import is_crypto_symbol, normalize_symbol

# Yahoo intraday limits (days of history per interval). Yahoo rejects ranges that touch the
# limit exactly, so these sit one day inside the documented caps (7 / 60 / 730).
_YF_INTERVAL = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 90: "90m"}
_YF_MAX_DAYS = {"1m": 6, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729, "90m": 59}


def _bars_to_frame(bars: list, symbol: str) -> pd.DataFrame:
    rows = [
        {
            "timestamp": bar.timestamp.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "symbol": symbol,
        }
        for bar in bars
    ]
    return pd.DataFrame(rows)


async def _fetch_alpaca(symbol: str, cfg: BotConfig, days: int, timeframe_min: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    timeframe = TimeFrame(timeframe_min, TimeFrameUnit.Minute)
    if is_crypto_symbol(symbol):
        client = CryptoHistoricalDataClient(cfg.api_key, cfg.api_secret)
        request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
        response = await asyncio.to_thread(client.get_crypto_bars, request)
    else:
        client = StockHistoricalDataClient(cfg.api_key, cfg.api_secret)
        feed = DataFeed.SIP if cfg.data_feed == "sip" else DataFeed.IEX
        request = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end, feed=feed
        )
        response = await asyncio.to_thread(client.get_stock_bars, request)
    return _bars_to_frame(response.data.get(symbol, []), symbol)


async def _fetch_yfinance(symbol: str, days: int, timeframe_min: int) -> pd.DataFrame:
    interval = _YF_INTERVAL.get(timeframe_min)
    if interval is None:
        raise SystemExit(f"yfinance has no {timeframe_min}-minute interval; use one of {sorted(_YF_INTERVAL)}")
    max_days = _YF_MAX_DAYS[interval]
    if days > max_days:
        print(f"  note: Yahoo caps {interval} history at {max_days} days (requested {days}); using {max_days}")
        days = max_days
    from historical_data import fetch_yahoo_history

    data = await asyncio.to_thread(fetch_yahoo_history, symbol, interval, f"{days}d")
    if data is None or data.empty or "timestamp" not in data.columns:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(data["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "open": data["open"].astype(float),
            "high": data["high"].astype(float),
            "low": data["low"].astype(float),
            "close": data["close"].astype(float),
            "volume": data["volume"].astype(float),
            "symbol": symbol,
        }
    )


def _labeled_rows(
    frame: pd.DataFrame,
    symbol: str,
    macro: MacroFredService,
    horizon: int,
    atr_mult: float,
) -> pd.DataFrame:
    if frame.empty or len(frame) < 100:
        return pd.DataFrame()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    macro_series = macro.scores_for_timestamps(frame["timestamp"]) if macro.is_loaded else 0.0
    enriched = build_features(frame, sentiment_score=0.0, macro_sentiment=macro_series)
    enriched["target"] = make_labels(enriched, horizon_bars=horizon, atr_mult=atr_mult)
    enriched = enriched.dropna(subset=FEATURE_COLUMNS + ["target"])
    enriched["target"] = enriched["target"].astype(int)
    enriched["symbol"] = symbol
    return enriched[["symbol", "timestamp"] + FEATURE_COLUMNS + ["target"]]


async def main() -> None:
    cfg = BotConfig()
    source = os.getenv("TRAIN_DATA_SOURCE", "alpaca").strip().lower()
    days = int(os.getenv("TRAIN_LOOKBACK_DAYS", "730"))
    timeframe_min = cfg.bar_timeframe_min
    horizon = int(os.getenv("LABEL_HORIZON_BARS", "6"))
    atr_mult = float(os.getenv("LABEL_ATR_MULT", "0.5"))
    out_path = os.getenv("TRAIN_DATA_CSV", "train_data.csv")

    print(f"source={source} lookback={days}d timeframe={timeframe_min}m horizon={horizon} bars atr_mult={atr_mult}")
    macro = MacroFredService()
    print(f"FRED macro: {'loaded' if macro.is_loaded else 'unavailable (macro feature = 0)'}")

    parts: list[pd.DataFrame] = []
    for raw in cfg.symbols:
        symbol = normalize_symbol(raw)
        if source == "yfinance":
            frame = await _fetch_yfinance(symbol, days, timeframe_min)
        else:
            frame = await _fetch_alpaca(symbol, cfg, days, timeframe_min)
        labeled = _labeled_rows(frame, symbol, macro, horizon, atr_mult)
        if labeled.empty:
            print(f"  skip {symbol}: insufficient bars ({len(frame)})")
            continue
        parts.append(labeled)
        first = str(labeled["timestamp"].iloc[0])[:10]
        last = str(labeled["timestamp"].iloc[-1])[:10]
        print(f"  {symbol}: {len(labeled)} labeled rows ({first} -> {last})")

    if not parts:
        raise SystemExit("No training rows produced; check symbols, data source, and Alpaca access.")

    dataset = pd.concat(parts, ignore_index=True)
    dataset.to_csv(out_path, index=False)
    counts = dataset["target"].value_counts().sort_index()
    mix = ", ".join(f"{LABEL_NAMES[int(k)]}={int(v)}" for k, v in counts.items())
    print(f"Wrote {len(dataset)} rows x {len(FEATURE_COLUMNS)} features to {out_path} ({mix})")


if __name__ == "__main__":
    asyncio.run(main())
