# AI AutoBot Trader

Automated algorithmic trading system that processes live market data, uses machine learning to predict short-term price direction, and executes trades via the Alpaca API. The bot combines **news sentiment** (Alpaca News + VADER), **macro mood** (Kaggle-style CSV), and **price action** (RSI/MACD/ATR) to dynamically size positions while managing risk with **broker-backed ATR trailing stop-losses**.

Supports **stocks and crypto** (e.g. `AAPL`, `BTC/USD`), a **dashboard scheduler** (including overnight windows), Discord alerts, and **Docker / Windows / Linux** deploy paths.

## How the Bot Learns to Trade

1. **Train offline:** `python train_model.py` fits Random Forest/XGBoost on `train_data.csv` (RSI, MACD, sentiment) and saves `model.pkl`
2. **Trade live:** each cycle computes the same features from market + news data and asks the model for buy/sell/hold
3. **Manage risk:** allocation sizes the order; ATR trailing stops protect positions
4. **Fallback:** if the model is unavailable, RSI/MACD rules still trade
5. **Improve:** update training data, retrain, restart — the bot does not auto-relearn after each trade

## Setup

```bash
cd aiautobottrader
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # add Alpaca keys (paper recommended)
python train_model.py
```

Windows one-shot:

```powershell
.\deploy.ps1 -Dashboard
```

Docker:

```bash
docker compose up -d --build
```

## Run

```bash
python main.py
python test_bot.py
python backtest.py
streamlit run dashboard_app.py
```

Control via `bot_control.flag`: `start` | `stop` | `emergency_stop`

Scheduler windows live in `scheduler_config.json` (editable in the dashboard **Scheduler** page). `AUTO_START` only writes the flag — `watch_flag()` always respects schedule windows and never overrides `emergency_stop`.

## Windows Task Scheduler (auto-restart)

1. Open **Task Scheduler** → Create Task
2. Trigger: At startup (or At log on)
3. Action: Start a program
   - Program: `C:\Users\<you>\...\aiautobottrader\.venv\Scripts\python.exe`
   - Arguments: `main.py`
   - Start in: `C:\Users\<you>\...\aiautobottrader`
4. Settings: **If the task fails, restart every 1 minute** (set retry count as needed)
5. Ensure `.env` exists in the Start-in folder

Optional CLI equivalent:

```powershell
schtasks /Create /TN "AIAutoBotTrader" /SC ONSTART /RL HIGHEST /TR "\"$PWD\.venv\Scripts\python.exe\" main.py" /WD "$PWD"
```

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

## Risk & Resilience

- Broker stop orders on buy + software ATR trail with cancel/replace
- Orphan order cancellation on disconnect, then stop re-arm
- REST fallback when WebSocket slots are exhausted
- Scheduler can pause trading outside configured hours (overnight supported)
- Fractional sizing for crypto and low buying-power accounts
- Discord webhook notifications on fills / failures

## Disclaimer

For educational purposes only. Markets are noisy; ML models overfit easily. Use **paper trading** until you validate live behavior. Not financial advice.
