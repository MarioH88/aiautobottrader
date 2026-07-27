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
python train_model.py
```

This reads `train_data.csv` and writes `model.pkl` (also gitignored — regenerate locally).

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

1. **Train offline:** `python train_model.py` fits Random Forest/XGBoost on `train_data.csv` (RSI, MACD, sentiment) and saves `model.pkl`
2. **Trade live:** each cycle computes the same features from market + news data and asks the model for buy/sell/hold
3. **Manage risk:** allocation sizes the order; ATR trailing stops protect positions
4. **Fallback:** if the model is unavailable, RSI/MACD rules still trade
5. **Improve:** update training data, retrain, restart — the bot does not auto-relearn after each trade

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
| `REST_FALLBACK_INTERVAL_SEC` | `120` | REST poll when WebSocket down |

---

## Disclaimer

For educational purposes only. Markets are noisy; ML models overfit easily. Use **paper trading** until you validate live behavior. Not financial advice.
