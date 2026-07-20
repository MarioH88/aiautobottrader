"""Quick integration test suite for the trading bot."""

from __future__ import annotations

import asyncio
import sys

from bot.config import BotConfig
from bot.connection_manager import ConnectionManager
from bot.data_pipeline import DataPipeline
from bot.macro_sentiment import MacroSentimentService, combine_sentiment
from bot.news_sentiment import NewsSentimentService
from bot.strategy_engine import StrategyEngine
from bot.trading_bot import TradingBot


async def run_tests() -> int:
    """Run smoke tests and return exit code."""
    failures = 0
    cfg = BotConfig()

    print("=== Test 1: Alpaca bar fetch ===")
    pipeline = DataPipeline(cfg.api_key, cfg.api_secret, bar_limit=50)
    try:
        frame = await pipeline.fetch_bars("AAPL")
        bar = pipeline.latest_bar_state(frame, "AAPL", sentiment_score=0.0)
        assert bar is not None, "No bar state"
        print(f"  OK  AAPL close={bar.close:.2f} rsi={bar.rsi:.1f} atr={bar.atr:.4f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 2: News + VADER sentiment ===")
    news = NewsSentimentService(cfg.api_key, cfg.api_secret)
    macro = MacroSentimentService(cfg.macro_sentiment_csv)
    try:
        await news.fetch_recent(cfg.symbols)
        macro_score = macro.get_score()
        for sym in cfg.symbols:
            combined = combine_sentiment(news.get_sentiment(sym), macro_score, cfg.sentiment_news_weight)
            print(f"  OK  {sym} news={news.get_sentiment(sym):.3f} macro={macro_score:.3f} combined={combined:.3f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 3: ML strategy engine ===")
    engine = StrategyEngine(cfg.model_path)
    try:
        assert pipeline.latest_bar_state(frame, "AAPL") is not None
        combined = combine_sentiment(news.get_sentiment("AAPL"), macro.get_score(), cfg.sentiment_news_weight)
        bar = pipeline.latest_bar_state(frame, "AAPL", sentiment_score=combined, macro_sentiment=macro.get_score())
        signal = engine.evaluate(bar)
        print(f"  OK  signal={signal.action} confidence={signal.confidence:.2f}")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 4: Connection manager retry ===")
    mgr = ConnectionManager(max_retries=2, base_delay_sec=0.1)
    try:
        result = await mgr.run_with_retry(lambda: asyncio.sleep(0), label="noop")
        print("  OK  retry wrapper works")
    except Exception as exc:
        print(f"  FAIL  {exc}")
        failures += 1

    print("=== Test 5: Full execution tick ===")
    bot = TradingBot(cfg)
    try:
        if bot.news_service:
            await bot.news_service.fetch_recent(cfg.symbols)
        for sym in cfg.symbols[:2]:
            await bot.execution_tick(sym, from_stream=False)
        print("  OK  execution ticks completed")
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
