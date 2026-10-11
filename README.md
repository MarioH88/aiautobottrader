# AI AutoBot Trader

## Project Title and Purpose

**AI AutoBot Trader** is an automated algorithmic trading system that:

- Pulls live and historical market data (stocks and crypto) from the **Alpaca** API
- Scores financial news with **VADER** sentiment and blends optional macro mood data
- Uses **machine learning** (Random Forest / XGBoost) plus RSI, MACD, and ATR to generate buy/sell/hold signals
- Executes trades with dynamic / fractional position sizing and **ATR trailing stop-losses**
- Provides a **Streamlit dashboard** for live prices, manual trades, and a trading-window scheduler

**Purpose:** demonstrate an end-to-end ML trading pipeline (data → features → model → risk → execution → UI) for education and paper-trading experiments. This is **not** financial advice.

---

## Team Members

| Name | GitHub Username |
|------|-----------------|
| Mario Hayden | [MarioH88](https://github.com/MarioH88) |

Repository: [github.com/MarioH88/aiautobottrader](https://github.com/MarioH88/aiautobottrader)

---

## Important Setup Instructions

### Requirements

- Python **3.10+** (3.12 recommended)
- An [Alpaca](https://alpaca.markets/) account (**paper trading keys strongly recommended**)
- Git (to clone the repo)

### 1. Clone the repository

```bash
git clone https://github.com/MarioH88/aiautobottrader.git
cd aiautobottrader
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure secrets (required)

API keys are **not** stored in Git. Copy the example env file and add your credentials:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

| Variable | Required | Notes |
|----------|----------|--------|
| `APCA_API_KEY_ID` | Yes | From Alpaca → API Keys |
| `APCA_API_SECRET_KEY` | Yes | Keep secret; never commit |
| `APCA_API_BASE_URL` | Yes | Paper: `https://paper-api.alpaca.markets` · Live: `https://api.alpaca.markets` |
| `TRADING_SYMBOLS` | Optional | Default `AAPL,TSLA` (crypto e.g. `BTC/USD` also supported) |
| `DISCORD_WEBHOOK_URL` | Optional | Trade / failure alerts |

> Use the **paper** base URL until you have validated the bot.

### 4. Train the ML model

```bash
python build_training_data.py   # ~2 years of 5m bars (Alpaca IEX) + FRED macro regime per bar
python train_model.py           # 3-class up/flat/down XGBoost, out-of-sample report, model.pkl
```

This writes `train_data.csv` and `model.pkl` (both gitignored — regenerate locally).
`train_model.py` prints held-out accuracy and a confidence-threshold table; the bot only acts when
the model's up/down probability clears the **recommended threshold** stored in `model.pkl`
(override with `MIN_SIGNAL_CONFIDENCE`). If the report says *no measurable edge*, stay on paper.

Alternate data source: `TRAIN_DATA_SOURCE=yfinance` (Yahoo caps 5-minute history at 60 days; use
`BAR_TIMEFRAME_MIN=60` for 2 years of hourly bars).

---

## Daily use (recommended workflow)

1. **One bot process only** — `main.py` uses `bot.pid` so a second copy exits instead of hitting Alpaca connection limits.
2. **Market hours** — `scheduler_config.json` defaults to **enabled**, Mon–Fri 9:30–16:00 ET. Outside the window the bot pauses even if the flag is `start`.
3. **Morning prep (Windows)** — refresh data, retrain, test, then start:

```powershell
.\run_daily.ps1
```

4. **Dashboard (optional)** — in another terminal: `streamlit run dashboard_app.py` for prices, manual trades, and scheduler edits.
5. **Signals** — set `MIN_SIGNAL_CONFIDENCE` (default `0.58`) so weak ML predictions become **hold** instead of churning sells. `MAX_DAILY_BUYS` caps new entries per day.
6. **Status file** — each tick updates `bot_status.json` (last signal, window state) for quick checks without reading logs.

Use **paper** keys until you are comfortable leaving the bot running unattended.

---

## How to Run the Code

### Trading bot (background loop)

```bash
python main.py
```

Control the bot with `bot_control.flag`: `start` | `stop` | `emergency_stop`.  
Trading windows are defined in `scheduler_config.json` (also editable in the dashboard **Scheduler** page).

### Streamlit dashboard

```bash
streamlit run dashboard_app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### Tests and backtest

```bash
python test_bot.py
python backtest.py
```

### Windows one-shot deploy helper

```powershell
.\deploy.ps1 -Dashboard
```

### Docker

```bash
docker compose up -d --build
```

Ensure a valid `.env` is present in the project root before starting containers.

---

## How the Bot Learns to Trade

1. **Build data:** `build_training_data.py` pulls ~2 years of 5-minute bars and computes 17 features
   (`bot/features.py`: RSI, MACD + histogram, 1/3/6/12-bar returns, volume ratio, ATR%, bar range,
   distance from SMA20/50, Bollinger position, time-of-day, news sentiment, FRED macro score).
   Each bar is labeled **up / flat / down** by whether the close 6 bars later moved more than 0.5×ATR.
2. **Train offline:** `train_model.py` fits XGBoost (or Random Forest), holds out the newest 20% of
   bars, and reports how often confident up/down calls were right vs. wrong at each threshold.
3. **Trade live:** every cycle the bot computes the *same* features from Alpaca bars, FinBERT-scored
   headlines, and live FRED macro data, then acts only when P(up) or P(down) clears the threshold.
4. **Manage risk:** allocation sizes the order; ATR trailing stops protect positions; `MAX_DAILY_BUYS` caps entries.
5. **Fallback:** if the model is unavailable, RSI/MACD rules still trade.
6. **Improve:** rerun steps 1–2 periodically (e.g. weekly) and restart — the bot does not auto-relearn.

**Sentiment:** `SENTIMENT_MODEL=finbert` uses the ProsusAI/finbert transformer (downloaded on first
run, ~440 MB, CPU is fine); set `vader` for a lightweight lexicon scorer.
**Macro:** `MACRO_SOURCE=fred` pulls VIX, high-yield spreads, the 10y–2y curve, 10y yield, the dollar
index and the S&P 500 from FRED (no API key), z-scores their 20-day changes and blends them into [-1, 1].

---

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APCA_API_BASE_URL` | paper URL | Paper recommended for safe testing |
| `TRADING_SYMBOLS` | `AAPL,TSLA` | Stocks and/or crypto (`BTC/USD`) |
| `USE_WEBSOCKET` | `false` | Live bar WebSocket |
| `ENABLE_NEWS` | `true` | Alpaca news sentiment |
| `SCHEDULER_CONFIG_FILE` | `scheduler_config.json` | Trading windows |
| `DISCORD_WEBHOOK_URL` | empty | Trade / failure alerts |
| `MAX_ALLOCATION_PCT` | `0.10` | Max buying-power fraction per symbol |
| `MIN_SIGNAL_CONFIDENCE` | *(from model.pkl)* | Min P(up)/P(down) to act; blank = trainer's recommendation |
| `MAX_DAILY_BUYS` | `20` | Cap new buy orders per UTC day (`0` = unlimited) |
| `BAR_TIMEFRAME_MIN` | `5` | Bar size for features; must match training |
| `ML_MODEL_TYPE` | `xgboost` | `xgboost` or `random_forest` |
| `SENTIMENT_MODEL` | `finbert` | `finbert` or `vader` |
| `MACRO_SOURCE` | `fred` | `fred` (live) or `csv` (static file) |
| `TRAIN_DATA_SOURCE` | `alpaca` | `alpaca` or `yfinance` |
| `TRAIN_LOOKBACK_DAYS` | `730` | History pulled by `build_training_data.py` |
| `REST_FALLBACK_INTERVAL_SEC` | `120` | REST poll when WebSocket down |

---

## Disclaimer

For educational purposes only. Markets are noisy; ML models overfit easily. Use **paper trading** until you validate live behavior. Not financial advice.
