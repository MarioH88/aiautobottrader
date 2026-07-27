# AI AutoBot Trader — Project Completion Report

**Author:** Mario Hayden  
**Date:** July 27, 2026  
**Overall Completion:** 100%

---

## 1. Project Description

**AI AutoBot Trader** is an automated algorithmic trading system that:

- Processes **live and historical OHLCV** bar data via Alpaca (stocks + crypto)
- Ingests **financial news** via Alpaca NewsClient and scores headlines with **VADER** (crypto alias-aware)
- Loads **macro sentiment** from a Kaggle-style CSV export
- Uses **machine learning** (Random Forest / XGBoost) plus RSI, MACD, and ATR
- Executes trades with **dynamic / fractional sizing**, **broker-backed ATR trailing stops**, and Discord alerts
- Provides a **Streamlit dashboard** (live prices, manual trade, multi-window scheduler)
- Supports **Docker Compose healthchecks**, Windows Task Scheduler, and deploy scripts

---

## 2. Feature Completion

| Feature / Task | Completion |
|----------------|------------|
| Modular async bot architecture | 100% |
| Alpaca REST OHLCV pipeline | 100% |
| WebSocket live streaming (stock + crypto) | 100% |
| Technical indicators (RSI/MACD/ATR) | 100% |
| News sentiment (VADER + crypto aliases) | 100% |
| Macro sentiment (Kaggle CSV) | 100% |
| ML training & inference | 100% |
| Order execution & allocation | 100% |
| ATR trailing stops (broker + software) | 100% |
| Auto-start trading (`AUTO_START`) | 100% |
| Trading window scheduler (overnight + multi-window) | 100% |
| Streamlit dashboard UI | 100% |
| Live market prices | 100% |
| Manual buy/sell UI | 100% |
| Backtesting (ATR exits + metrics + crypto) | 100% |
| Integration / unit tests | 100% |
| Crypto OHLCV | 100% |
| Production deployment (Docker / Windows / Linux) | 100% |
| Discord notifications | 100% |

**Overall: 100%**

---

## 3. How the Bot Learns to Trade

The system learns **offline**, then applies that knowledge while trading. It does **not** continuously retrain itself after every live trade.

### Training (learning step)

1. Run `python train_model.py`
2. The script reads historical rows from `train_data.csv`
3. Features used: **RSI**, **MACD**, and **sentiment_score** (news/macro mood)
4. Target label: short-term direction — **up (1)** vs **down (0)**
5. A **Random Forest** (default) or **XGBoost** classifier is fit and saved as `model.pkl`

### Live trading (using what it learned)

When `main.py` runs, each cycle:

1. Fetches latest OHLCV bars (stocks and/or crypto)
2. Computes RSI / MACD / ATR
3. Scores recent headlines with VADER and blends macro sentiment
4. Passes features into `model.pkl` for a buy / sell / hold signal
5. Allocation engine chooses position size (including fractional when needed)
6. Execution client places orders and manages ATR trailing stops

### Fallback and improvement

- If `model.pkl` is missing or fails, a **rule-based RSI/MACD** strategy is used
- To improve behavior: update/expand `train_data.csv`, re-run `train_model.py`, restart the bot
- Backtests (`python backtest.py`) validate the strategy on historical data before relying on live results

---

## 4. UI Screenshot

See `docs/report_assets/dashboard_ui_screenshot.png`

---

## 5. Code Screenshots / Listings

See `docs/report_assets/code_screenshot_main.png` and `main.py`, `bot/trading_bot.py`, `dashboard_app.py`

---

## 6. Challenges

1. **WebSocket connection limit** — REST polling + reconnect circuit breaker
2. **ML live vs training mismatch** — sentiment features + rule-based fallback
3. **Fractional order time-in-force** — DAY for fractional quantities
4. **Windows UTF-8 BOM flag file** — UTF-8-safe flag reader/writer
5. **pandas-ta on NumPy 2.x** — pure-pandas indicator fallback

---

## 7. Export to PDF

Open `docs/PROJECT_COMPLETION_REPORT.html` in a browser → **Print → Save as PDF**
