# AI AutoBot Trader

Automated algorithmic trading system that processes live market data, uses machine learning to predict short-term price direction, and executes trades via the Alpaca API. The bot combines **news sentiment** (Alpaca News + VADER), **macro mood** (Kaggle-style CSV), and **price action** (RSI/MACD/ATR) to dynamically size positions while managing risk with ATR trailing stop-losses.

## Data Sources

| Source | API / File | Usage |
|--------|------------|-------|
| OHLCV bars | Alpaca Market Data (REST + optional WebSocket) | Features + execution reference |
| News headlines | Alpaca NewsClient (REST poll or WebSocket) | Per-symbol VADER sentiment |
| Macro sentiment | `macro_sentiment.csv` (Kaggle export) | Market-wide mood overlay |

## Architecture

```
main.py
  └── bot/trading_bot.py           # Async orchestrator
        ├── data_pipeline.py       # OHLCV + RSI/MACD/ATR
        ├── news_sentiment.py      # Alpaca News + VADER
        ├── macro_sentiment.py     # Kaggle macro mood CSV
        ├── strategy_engine.py     # ML direction forecast
        ├── allocation_engine.py   # Dynamic position sizing
        ├── execution_client.py    # Orders + trailing stops + orphan cancel
        ├── stream_client.py       # WebSocket live bars
        └── connection_manager.py  # Retry, 429 backoff, disconnect recovery
```

## Stack

- **alpaca-py** — REST trading, market data, news
- **pandas / numpy** — feature engineering
- **scikit-learn / XGBoost** — direction classifier (`ML_MODEL_TYPE`)
- **pandas-ta** (with pure-pandas fallback) — technical indicators
- **pydantic** — validated market/position schemas
- **vaderSentiment** — headline scoring
- **python-dotenv** — credential management
- **backtrader** — historical simulation (`backtest.py`)

## Setup

```bash
cd aiautobottrader
pip install -r requirements.txt
cp .env.example .env   # add Alpaca keys (paper recommended)
python train_model.py  # or ML_MODEL_TYPE=xgboost python train_model.py
```

Place a Kaggle macro sentiment export at `macro_sentiment.csv` (columns: `date`, `macro_sentiment`).

## Run

```bash
python main.py
python test_bot.py     # integration smoke tests
python backtest.py     # historical simulation
```

Control via `bot_control.flag`: `start` | `stop` | `emergency_stop`

```powershell
Set-Content -Path bot_control.flag -Value "start" -Encoding utf8 -NoNewline
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APCA_API_BASE_URL` | paper URL | Use paper endpoint for safe testing |
| `USE_WEBSOCKET` | `true` | Live bar WebSocket (disable if connection limit hit) |
| `ENABLE_NEWS` | `true` | Alpaca news sentiment |
| `USE_NEWS_WEBSOCKET` | `false` | News WebSocket (uses extra connection slot) |
| `ENABLE_MACRO_SENTIMENT` | `true` | Load Kaggle macro CSV |
| `MACRO_SENTIMENT_CSV` | `macro_sentiment.csv` | Path to macro dataset |
| `SENTIMENT_NEWS_WEIGHT` | `0.6` | News vs macro blend weight |
| `MAX_ALLOCATION_PCT` | `0.10` | Max buying-power fraction per symbol |
| `ML_MODEL_TYPE` | `random_forest` | `random_forest` or `xgboost` |
| `ATR_MULTIPLIER` | `2.0` | Trailing stop distance |
| `REST_FALLBACK_INTERVAL_SEC` | `120` | REST poll when WebSocket down |

## Risk & Resilience

- **ATR trailing stops** tighten as price moves favorably
- **Orphan order cancellation** on WebSocket disconnect or connectivity loss
- **REST fallback loop** continues polling when streams are unavailable
- **Connection manager** handles HTTP 429 and exponential backoff

## Disclaimer

For educational purposes only. Markets are noisy; ML models overfit easily. Use **paper trading** until you validate live behavior. Not financial advice.
