"""Quick integration + unit test suite for the trading bot."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bot.config import BotConfig
from bot.connection_manager import ConnectionManager
from bot.data_pipeline import DataPipeline
from bot.flag_file import read_flag, write_flag
from bot.macro_sentiment import MacroSentimentService, combine_sentiment
from bot.news_sentiment import NewsSentimentService
from bot.scheduler import (
    is_trading_window_active,
    load_scheduler_config,
    save_scheduler_config,
)
from bot.schemas import MarketBarState, PositionState, TradeSignal
from bot.strategy_engine import StrategyEngine
from bot.symbols import is_crypto_symbol, news_aliases, normalize_symbol, partition_symbols
from bot.allocation_engine import AllocationEngine
from bot.execution_client import ExecutionClient
from backtest import summarize_trades
import pandas as pd


def test_symbol_utils() -> int:
    failures = 0
    print("=== Test 0a: Symbol helpers ===")
    try:
        assert is_crypto_symbol("BTC/USD")
        assert normalize_symbol("btcusd") == "BTC/USD"
        assert "BTC" in news_aliases("BTC/USD")
        assert "BTCUSD" in news_aliases("BTC/USD")
        stocks, crypto = partition_symbols(["AAPL", "BTCUSD", "ETH/USD"])
        assert stocks == ["AAPL"]
        assert "BTC/USD" in crypto
        print("  OK  crypto/stock normalization + news aliases")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


def test_scheduler_utils() -> int:
    failures = 0
    print("=== Test 0b: Scheduler windows ===")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler_config.json"
            config = {
                "enabled": True,
                "timezone": "America/New_York",
                "windows": [
                    {"days": ["mon", "tue", "wed", "thu", "fri"], "start": "09:30", "end": "16:00"},
                    {"days": ["mon", "tue", "wed", "thu", "fri"], "start": "22:00", "end": "06:00"},
                ],
            }
            assert save_scheduler_config(path, config)
            loaded = load_scheduler_config(path)
            monday_open = datetime(2026, 7, 27, 10, 0, tzinfo=ZoneInfo("America/New_York"))
            monday_closed = datetime(2026, 7, 27, 8, 0, tzinfo=ZoneInfo("America/New_York"))
            monday_overnight = datetime(2026, 7, 27, 23, 0, tzinfo=ZoneInfo("America/New_York"))
            assert is_trading_window_active(loaded, monday_open) is True
            assert is_trading_window_active(loaded, monday_closed) is False
            assert is_trading_window_active(loaded, monday_overnight) is True
            print("  OK  scheduler open/closed/overnight windows")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


def test_flag_bom() -> int:
    failures = 0
    print("=== Test 0c: Flag UTF-8/BOM ===")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot_control.flag"
            path.write_bytes(b"\xef\xbb\xbfstart")
            assert read_flag(str(path)) == "start"
            assert write_flag(str(path), "stop")
            assert path.read_bytes() == b"stop"
            print("  OK  BOM-tolerant flag read/write")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


async def test_trailing_stop_logic() -> int:
    failures = 0
    print("=== Test 0d: ATR trailing stop ratchet ===")
    try:
        client = ExecutionClient("k", "s", paper=True)
        client._positions["AAPL"] = PositionState(
            symbol="AAPL",
            entry_price=100.0,
            current_quantity=1.0,
            current_stop_loss=98.0,
            trailing_activation_price=100.0,
            atr_at_entry=1.0,
        )
        bar = MarketBarState(
            timestamp="t",
            symbol="AAPL",
            open=101,
            high=103,
            low=100,
            close=102,
            volume=10,
            rsi=50,
            macd=0.1,
            atr=1.0,
        )
        await client.update_trailing_stop(bar)
        assert client._positions["AAPL"].current_stop_loss == max(98.0, 102.0 - 2.0)
        print("  OK  trailing stop ratchets with ATR")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


def test_allocation_fractional() -> int:
    failures = 0
    print("=== Test 0e: Fractional allocation ===")
    try:
        engine = AllocationEngine(base_qty=1.0, max_allocation_pct=0.10)
        signal = TradeSignal(symbol="TSLA", action="buy", confidence=0.8, target_allocation_pct=0.1)
        bar = MarketBarState(
            timestamp="t",
            symbol="TSLA",
            open=300,
            high=301,
            low=299,
            close=300,
            volume=1,
            rsi=50,
            macd=0.0,
            sentiment_score=0.2,
        )
        qty = engine.target_quantity(signal, bar, buying_power=50.0, held_qty=0.0)
        assert 0 < qty < 1
        print(f"  OK  fractional qty={qty:.6f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


def test_strategy_confidence_gate() -> int:
    failures = 0
    print("=== Test 0g: ML confidence gate ===")
    try:

        import numpy as np

        class _FakeModel:
            classes_ = np.array([0, 1])

            def predict(self, _features):
                return [0]

            def predict_proba(self, _features):
                return [[0.70, 0.30]]

        engine = StrategyEngine(model_path="__missing__.pkl", min_signal_confidence=0.58)
        engine._apply_artifact(_FakeModel())  # legacy binary estimator path
        bar = MarketBarState(
            timestamp="t",
            symbol="AAPL",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            rsi=50,
            macd=0.0,
            sentiment_score=0.0,
        )
        strong = engine.evaluate(bar)
        assert strong.action == "sell" and strong.reason == "ml-5m-down"

        class _WeakModel:
            classes_ = np.array([0, 1])

            def predict(self, _features):
                return [1]

            def predict_proba(self, _features):
                return [[0.45, 0.55]]

        engine._apply_artifact(_WeakModel())
        weak = engine.evaluate(bar)
        assert weak.action == "hold" and weak.reason == "ml-low-confidence"

        class _ThreeClass:
            classes_ = np.array([0, 1, 2])

            def predict(self, _features):
                return [2]

            def predict_proba(self, _features):
                return [[0.10, 0.25, 0.65]]

        from bot.features import FEATURE_COLUMNS

        engine._apply_artifact(
            {"model": _ThreeClass(), "feature_columns": FEATURE_COLUMNS, "n_classes": 3,
             "recommended_min_confidence": 0.5}
        )
        rich_bar = bar.model_copy(update={"features": {c: 0.0 for c in FEATURE_COLUMNS}})
        up = engine.evaluate(rich_bar)
        assert up.action == "buy" and up.reason == "ml-up", up
        print("  OK  low confidence -> hold; 3-class up -> buy")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


def test_backtest_summary() -> int:
    failures = 0
    print("=== Test 0f: Backtest summary metrics ===")
    try:
        trades = pd.DataFrame(
            [
                {"action": "buy", "price": 100},
                {"action": "sell", "price": 110, "pnl": 10},
                {"action": "buy", "price": 110},
                {"action": "sell", "price": 105, "pnl": -5},
            ]
        )
        summary = summarize_trades(trades)
        assert summary["num_trades"] == 2
        assert summary["total_pnl"] == 5
        assert 0 < summary["win_rate"] <= 1
        print("  OK  summarize_trades metrics")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1
    return failures


async def run_tests() -> int:
    failures = 0
    failures += test_symbol_utils()
    failures += test_scheduler_utils()
    failures += test_flag_bom()
    failures += await test_trailing_stop_logic()
    failures += test_allocation_fractional()
    failures += test_strategy_confidence_gate()
    failures += test_backtest_summary()

    cfg = BotConfig()

    print("=== Test 1: Alpaca bar fetch ===")
    pipeline = DataPipeline(
        cfg.api_key,
        cfg.api_secret,
        bar_limit=cfg.bar_limit,
        timeframe_min=cfg.bar_timeframe_min,
        data_feed=cfg.data_feed,
    )
    frame = None
    try:
        frame = await pipeline.fetch_bars("AAPL")
        bar = pipeline.latest_bar_state(frame, "AAPL", sentiment_score=0.0)
        assert bar is not None, "No bar state"
        print(f"  OK  AAPL close={bar.close:.2f} rsi={bar.rsi:.1f} atr={bar.atr:.4f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 1b: Crypto OHLCV fetch ===")
    try:
        crypto_frame = await pipeline.fetch_bars("BTC/USD")
        crypto_bar = pipeline.latest_bar_state(crypto_frame, "BTC/USD", sentiment_score=0.0)
        assert crypto_bar is not None, "No crypto bar state"
        print(f"  OK  BTC/USD close={crypto_bar.close:.2f} rsi={crypto_bar.rsi:.1f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 2: News sentiment (FinBERT/VADER) + FRED macro ===")
    news = NewsSentimentService(
        cfg.api_key, cfg.api_secret, lookback_minutes=7 * 24 * 60, sentiment_model=cfg.sentiment_model
    )
    macro = MacroSentimentService(cfg.macro_sentiment_csv, source=cfg.macro_source)
    try:
        await news.fetch_recent(cfg.symbols + ["BTC/USD"])
        macro_score = macro.get_score()
        print(f"  OK  scorer={news.scorer_name} macro_loaded={macro.is_loaded}")
        for sym in cfg.symbols:
            combined = combine_sentiment(news.get_sentiment(sym), macro_score, cfg.sentiment_news_weight)
            print(f"  OK  {sym} news={news.get_sentiment(sym):.3f} macro={macro_score:.3f} combined={combined:.3f}")
        print(f"  OK  BTC/USD news={news.get_sentiment('BTC/USD'):.3f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 3: ML strategy engine ===")
    engine = StrategyEngine(cfg.model_path)
    try:
        assert frame is not None, "Stock frame unavailable from Test 1"
        combined = combine_sentiment(news.get_sentiment("AAPL"), macro.get_score(), cfg.sentiment_news_weight)
        bar = pipeline.latest_bar_state(frame, "AAPL", sentiment_score=combined, macro_sentiment=macro.get_score())
        signal = engine.evaluate(bar)
        assert len(bar.features) >= 15, f"feature vector incomplete: {len(bar.features)}"
        print(
            f"  OK  signal={signal.action} confidence={signal.confidence:.2f} "
            f"reason={signal.reason} features={len(bar.features)} threshold={engine.min_confidence:.2f}"
        )
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 4: Connection manager retry ===")
    mgr = ConnectionManager(max_retries=2, base_delay_sec=0.1)
    try:
        await mgr.run_with_retry(lambda: asyncio.sleep(0), label="noop")
        print("  OK  retry wrapper works")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 5: Signal evaluation (no live orders) ===")
    try:
        for sym in cfg.symbols[:2]:
            sym_frame = await pipeline.fetch_bars(sym)
            sentiment = news.get_sentiment(sym)
            bar = pipeline.latest_bar_state(
                sym_frame,
                sym,
                sentiment_score=combine_sentiment(sentiment, macro.get_score(), cfg.sentiment_news_weight),
                macro_sentiment=macro.get_score(),
            )
            assert bar is not None
            signal = engine.evaluate(bar)
            print(f"  OK  {sym} -> {signal.action} ({signal.reason})")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print()
    if failures:
        print(f"RESULT: {failures} test(s) failed")
    else:
        print("RESULT: All tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
