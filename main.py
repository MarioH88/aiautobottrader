
# --- Standard and third-party imports ---
import os
import time
import threading
from collections import defaultdict, deque
import pandas as pd
import schedule
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca.data.live import StockDataStream
from indicators import add_indicators
from ml_predict import load_sklearn_model, predict_action
from trade_log import log_trade
from datetime import datetime
from notify import send_email, send_discord


# --- NOTIFICATION CONFIG ---
# IMPORTANT: The previous Discord webhook URL was hardcoded and is now removed for security. 
# Please revoke and change your Discord webhook immediately if this code was ever public or shared.
# Store your webhook in the .env file as DISCORD_WEBHOOK_URL and load it securely below.
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

# --- RISK MANAGEMENT CONFIG ---
DAILY_PNL_LIMIT = 100.0  # Example: $100 max loss per day
MAX_DRAWDOWN = 0.10      # 10% max drawdown
MAX_TRADES_PER_DAY = 10

# --- ML MODEL CONFIG ---
ML_MODEL_PATH = 'model.pkl'  # Path to your trained model
model = None

# --- STATE VARIABLES ---
starting_equity = None
max_equity = None
trades_today = 0
EMERGENCY_STOP = False


# --- Load environment variables from .env file ---
load_dotenv()

# --- CONFIGURATION ---
USE_PAPER = True  # Set to False for live trading
PAPER_URL = 'https://paper-api.alpaca.markets'
LIVE_URL = 'https://api.alpaca.markets'

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = PAPER_URL if USE_PAPER else LIVE_URL

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# --- ML Model Load ---
if os.path.exists(ML_MODEL_PATH):
    model = load_sklearn_model(ML_MODEL_PATH)
    print('ML model loaded.')
else:
    print('ML model not found. Trading will use rule-based logic.')

# --- Real-time data config ---
TICKERS = ['AAPL', 'TSLA']
ohlcv_data = defaultdict(lambda: deque(maxlen=1000))  # Store up to 1000 bars per ticker

# --- Trading parameters ---
SYMBOL = 'AAPL'
SHORT_WINDOW = 20
LONG_WINDOW = 50
QTY = 1
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit

def get_account_equity():
    try:
        return float(api.get_account().equity)
    except Exception:
        return None

def get_moving_averages_and_rsi(symbol, short_window, long_window):
    # Use in-memory OHLCV if available, else fallback to REST
    if len(ohlcv_data[symbol]) >= long_window+15:
        df = pd.DataFrame(list(ohlcv_data[symbol]))
    else:
        barset = api.get_bars(symbol, TimeFrame.Minute, limit=long_window+15)
        df = barset.df[barset.df['symbol'] == symbol].copy()
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def get_position(symbol):
    try:
        position = api.get_position(symbol)
        return int(position.qty), float(position.avg_entry_price)
    except Exception:
        return 0, 0.0


def check_risk_and_emergency(equity, global_vars):
    """Check daily P&L, drawdown, and emergency stop."""
    starting_equity, max_equity, EMERGENCY_STOP = global_vars['starting_equity'], global_vars['max_equity'], global_vars['EMERGENCY_STOP']
    if starting_equity is None and equity is not None:
        starting_equity = equity
        max_equity = equity
    if equity is not None:
        if equity > max_equity:
            max_equity = equity
        daily_pnl = equity - starting_equity
        drawdown = (max_equity - equity) / max_equity if max_equity else 0
        if daily_pnl < -DAILY_PNL_LIMIT:
            print(f"Daily P&L limit reached: {daily_pnl:.2f}. Emergency stop triggered.")
            EMERGENCY_STOP = True
        if drawdown > MAX_DRAWDOWN:
            print(f"Max drawdown exceeded: {drawdown:.2%}. Emergency stop triggered.")
            EMERGENCY_STOP = True
    global_vars['starting_equity'] = starting_equity
    global_vars['max_equity'] = max_equity
    global_vars['EMERGENCY_STOP'] = EMERGENCY_STOP
    return EMERGENCY_STOP

def handle_emergency_stop():
    print("EMERGENCY STOP: Trading halted.")
    try:
        send_discord("EMERGENCY STOP: Trading halted.", DISCORD_WEBHOOK_URL)
    except Exception:
        pass

def handle_stop_loss_take_profit(entry_price, latest, trades_today):
    current_price = latest['close']
    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
    if current_price <= stop_loss_price:
        print(f"Stop-loss triggered at {current_price:.2f}. Selling {SYMBOL}.")
        api.submit_order(symbol=SYMBOL, qty=QTY, side='sell', type='market', time_in_force='gtc')
        log_trade(datetime.now(), SYMBOL, 'sell', current_price, QTY, 'stop-loss')
        try:
            send_discord(f"Stop-loss: Sold {SYMBOL} at {current_price:.2f}", DISCORD_WEBHOOK_URL)
        except Exception:
            pass
        trades_today += 1
        return True, trades_today
    elif current_price >= take_profit_price:
        print(f"Take-profit triggered at {current_price:.2f}. Selling {SYMBOL}.")
        api.submit_order(symbol=SYMBOL, qty=QTY, side='sell', type='market', time_in_force='gtc')
        log_trade(datetime.now(), SYMBOL, 'sell', current_price, QTY, 'take-profit')
        try:
            send_discord(f"Take-profit: Sold {SYMBOL} at {current_price:.2f}", DISCORD_WEBHOOK_URL)
        except Exception:
            pass
        trades_today += 1
        return True, trades_today
    return False, trades_today

def handle_ml_trading(latest, position, trades_today):
    features = latest[['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'bb_high', 'bb_low']].to_frame().T
    action = predict_action(model, features)
    if action == 1 and position == 0:
        print(f"ML model: Buy signal for {SYMBOL}. Placing order...")
        api.submit_order(symbol=SYMBOL, qty=QTY, side='buy', type='market', time_in_force='gtc')
        log_trade(datetime.now(), SYMBOL, 'buy', latest['close'], QTY, 'ml-buy')
        try:
            send_discord(f"ML Buy: Bought {SYMBOL} at {latest['close']:.2f}", DISCORD_WEBHOOK_URL)
        except Exception:
            pass
        trades_today += 1
    elif action == -1 and position > 0:
        print(f"ML model: Sell signal for {SYMBOL}. Placing order...")
        api.submit_order(symbol=SYMBOL, qty=QTY, side='sell', type='market', time_in_force='gtc')
        log_trade(datetime.now(), SYMBOL, 'sell', latest['close'], QTY, 'ml-sell')
        try:
            send_discord(f"ML Sell: Sold {SYMBOL} at {latest['close']:.2f}", DISCORD_WEBHOOK_URL)
        except Exception:
            pass
        trades_today += 1
    else:
        print("ML model: Hold signal. No trade.")
    return trades_today

def handle_rule_trading(prev, latest, position, trades_today):
    if (
        prev['short_ma'] < prev['long_ma'] and
        latest['short_ma'] > latest['long_ma'] and
        latest['rsi'] < 70 and
        position == 0
    ):
        print(f"Buy signal for {SYMBOL}. Placing order...")
        api.submit_order(symbol=SYMBOL, qty=QTY, side='buy', type='market', time_in_force='gtc')
        log_trade(datetime.now(), SYMBOL, 'buy', latest['close'], QTY, 'rule-buy')
        try:
            send_discord(f"Rule Buy: Bought {SYMBOL} at {latest['close']:.2f}", DISCORD_WEBHOOK_URL)
        except Exception:
            pass
        trades_today += 1
    elif (
        ((prev['short_ma'] > prev['long_ma'] and latest['short_ma'] < latest['long_ma']) or latest['rsi'] > 80)
        and position > 0
    ):
        print(f"Sell signal for {SYMBOL}. Placing order...")
        api.submit_order(symbol=SYMBOL, qty=QTY, side='sell', type='market', time_in_force='gtc')
        log_trade(datetime.now(), SYMBOL, 'sell', latest['close'], QTY, 'rule-sell')
        try:
            send_discord(f"Rule Sell: Sold {SYMBOL} at {latest['close']:.2f}", DISCORD_WEBHOOK_URL)
        except Exception:
            pass
        trades_today += 1
    else:
        print("No trade signal.")
    return trades_today

def trade_logic():
    global trades_today, EMERGENCY_STOP, starting_equity, max_equity
    df = get_moving_averages_and_rsi(SYMBOL, SHORT_WINDOW, LONG_WINDOW)
    if len(df) < LONG_WINDOW + 1:
        print("Not enough data to trade.")
        return
    df = add_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    position, entry_price = get_position(SYMBOL)

    # Risk management: check daily P&L and drawdown
    global_vars = {'starting_equity': starting_equity, 'max_equity': max_equity, 'EMERGENCY_STOP': EMERGENCY_STOP}
    equity = get_account_equity()
    if check_risk_and_emergency(equity, global_vars):
        EMERGENCY_STOP = True
        handle_emergency_stop()
        starting_equity = global_vars['starting_equity']
        max_equity = global_vars['max_equity']
        return
    starting_equity = global_vars['starting_equity']
    max_equity = global_vars['max_equity']
    EMERGENCY_STOP = global_vars['EMERGENCY_STOP']

    # Stop-loss and take-profit logic
    if position > 0:
        stop_triggered, trades_today = handle_stop_loss_take_profit(entry_price, latest, trades_today)
        if stop_triggered:
            return

    # ML-based trading decision
    if trades_today >= MAX_TRADES_PER_DAY:
        print("Max trades per day reached. No more trades today.")
        return
    if model is not None:
        trades_today = handle_ml_trading(latest, position, trades_today)
    else:
        trades_today = handle_rule_trading(prev, latest, position, trades_today)

def on_bar(bar):
    # Store OHLCV bar in memory
    ohlcv_data[bar.symbol].append({
        'timestamp': bar.timestamp,
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume,
        'symbol': bar.symbol
    })
    print(f"[BAR] {bar.symbol} {bar.timestamp} O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}")

def start_realtime_stream():
    stream = StockDataStream(API_KEY, API_SECRET, base_url=BASE_URL)
    for ticker in TICKERS:
        stream.subscribe_bars(on_bar, ticker)
    print(f"Subscribed to real-time bars for: {TICKERS}")
    stream.run()

if __name__ == "__main__":
    account = api.get_account()
    print(f"Account status: {account.status}")
    # Start real-time data in a separate thread if needed
    stream_thread = threading.Thread(target=start_realtime_stream, daemon=True)
    stream_thread.start()
    # Schedule trading logic every 5 minutes
    schedule.every(5).minutes.do(trade_logic)
    print("Automated trading started. Running every 5 minutes.")
    while True:
        schedule.run_pending()
        time.sleep(1)
        print("Waiting 1 minute before next check...")
        time.sleep(60)
