import time
import yfinance as yf
import backtrader as bt

# --- Scheduler Thread Function (define before sidebar/menu) ---
def run_bot_job():
    """Run the aggressive trading bot using a Backtrader SMA crossover strategy and Alpaca for live trading."""
    symbols = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'TSLA']
    best_symbol = None
    try:
        class SmaCross(bt.SignalStrategy):
            def __init__(self):
                sma1 = bt.ind.SMA(period=10)
                sma2 = bt.ind.SMA(period=30)
                self.signal_add(bt.SIGNAL_LONG, bt.ind.CrossOver(sma1, sma2))

        # Find first symbol with a buy signal
        for symbol in symbols:
            data = yf.download(symbol, period='30d', interval='15m')
            if data.empty:
                continue
            data_bt = bt.feeds.PandasData(dataname=data)
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SmaCross)
            cerebro.adddata(data_bt)
            cerebro.broker.set_cash(10000)
            results = cerebro.run()
            strat = results[0]
            # Try to get last signal from Backtrader
            last_signal = None
            if hasattr(strat, 'signals') and len(strat.signals) > 0:
                last_signal = strat.signals[-1]
            # Fallback: check if last close > SMA(10)
            if last_signal is None:
                try:
                    last_signal = 1 if data['Close'][-1] > data['Close'].rolling(10).mean()[-1] else 0
                except Exception:
                    last_signal = 0
            if last_signal == 1:
                best_symbol = symbol
                break
        # Place trade if a buy signal was found
        if best_symbol:
            acc = api.get_account()
            price = float(api.get_last_trade(best_symbol).price)
            cash = float(acc.cash)
            qty = int(cash // price)
            if qty > 0:
                api.submit_order(
                    symbol=best_symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Placed market buy order for {qty} shares of {best_symbol}.")
            else:
                print(f"Insufficient cash to buy {best_symbol}. Skipping trade.")
        else:
            print("No buy signal found for any symbol. Skipping trade.")
    except Exception as e:
        print(f"Trade error: {e}")
    st.session_state['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
import threading
# --- Scheduler Thread Function (define before sidebar/menu) ---
def scheduler_thread(target_dt):
    while st.session_state.get('scheduler_running', False):
        now = datetime.now()
        if now >= target_dt:
            run_bot_job()
            st.session_state['scheduler_running'] = False
            break
        time.sleep(1)
from dotenv import load_dotenv
import os
import pandas as pd
ALPACA_API_KEY = os.getenv("PKTPQ64STD8F9IN7P4FDD")
# --- Cleaned, single-version dashboard app ---
import streamlit as st
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import alpaca_trade_api as tradeapi
from bs4 import BeautifulSoup
# --- Additional Tools for AI Trading Bot (for future integration) ---
# Sentiment Analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None
# Hugging Face Transformers (advanced NLP)
try:
    from transformers import pipeline
except ImportError:
    pipeline = None
# Alternative Data
try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    import finnhub
except ImportError:
    finnhub = None
# Backtesting & Optimization
try:
    import backtrader as bt
except ImportError:
    bt = None
try:
    import optuna
except ImportError:
    optuna = None
# Logging
try:
    from loguru import logger
except ImportError:
    logger = None
# Visualization
try:
    import plotly.express as px
except ImportError:
    px = None
# Scheduler (advanced)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None
# Error Monitoring
try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None

# These imports are for future use and do not affect current app logic.

# --- Lottie Loader Utility (define before use) ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_trading = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t24tpvcu.json")

# --- Web Scraping Utility ---
def scrape_headlines(url, selector):
    """Scrape headlines or data from a given URL using a CSS selector."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [el.get_text(strip=True) for el in soup.select(selector)]
        return headlines
    except Exception as e:
        return [f"Error: {e}"]


load_dotenv()

# --- API Connection Status Helper ---
def check_api_connection():
    try:
        # Try to get account info to verify connection
        acc = api.get_account()
        if acc.status:
            return True, None
        else:
            return False, "API returned no status."
    except Exception as e:
        return False, str(e)

# --- Constants ---
CARD_DIV_OPEN = '<div class="card" style="background: #181818; border-radius: 16px; box-shadow: 0 4px 24px #0003; padding: 1.5em 1em; margin-bottom: 1.5em;">'
CARD_DIV_CLOSE = '</div>'
BOT_LOGS_LABEL = "Bot Logs"

# --- Alpaca API Setup ---
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# --- Data Functions ---
def get_positions():
    positions = api.list_positions()
    return pd.DataFrame([{
        'symbol': p.symbol,
        'qty': float(p.qty),
        'avg_entry_price': float(p.avg_entry_price),
        'market_value': float(p.market_value)
    } for p in positions])

def get_account_info():
    acc = api.get_account()
    return {
        'status': acc.status,
        'equity': float(acc.equity),
        'cash': float(acc.cash),
        'buying_power': float(acc.buying_power),
        'last_update': getattr(acc, 'last_equity_update', None)
    }

def get_recent_trades(max_trades=500):
    # Alpaca API max page_size is 100, so we need to paginate
    all_activities = []
    page_token = None
    fetched = 0
    while fetched < max_trades:
        page_size = min(100, max_trades - fetched)
        params = {'activity_types': 'FILL', 'direction': 'desc', 'page_size': page_size}
        if page_token:
            params['page_token'] = page_token
        resp = api.get_activities(**params)
        if not resp:
            break
        all_activities.extend(resp)
        fetched += len(resp)
        # If less than page_size returned, we're done
        if len(resp) < page_size:
            break
        # Try to get the next page token if available (Alpaca v2 pagination)
        try:
            page_token = resp[-1].id
        except Exception:
            break
    return pd.DataFrame([{
        'time': pd.to_datetime(a.transaction_time),
        'symbol': a.symbol,
        'side': a.side,
        'price': float(a.price),
        'qty': float(a.qty)
    } for a in all_activities])

def calculate_earnings(trades_df):
    if trades_df.empty:
        summary = pd.DataFrame({'Period': ['Today', 'This Month', 'This Year'], 'Earnings': [0, 0, 0]})
        daily = pd.DataFrame(columns=['date', 'amount'])
        monthly = pd.DataFrame(columns=['month', 'amount'])
        annual = pd.DataFrame(columns=['year', 'amount'])
        return summary, daily, monthly, annual
    trades_df = trades_df.copy()
    trades_df['amount'] = trades_df['price'] * trades_df['qty'] * trades_df['side'].map({'buy': -1, 'sell': 1})
    trades_df['date'] = trades_df['time'].dt.date
    trades_df['month'] = trades_df['time'].dt.to_period('M')
    trades_df['year'] = trades_df['time'].dt.year
    daily = trades_df.groupby('date')['amount'].sum().reset_index()
    monthly = trades_df.groupby('month')['amount'].sum().reset_index()
    annual = trades_df.groupby('year')['amount'].sum().reset_index()
    summary = pd.DataFrame({
        'Period': ['Today', 'This Month', 'This Year'],
        'Earnings': [
            daily[daily['date'] == pd.Timestamp.today().date()]['amount'].sum() if not daily.empty else 0,
            monthly[monthly['month'] == pd.Timestamp.today().to_period('M')]['amount'].sum() if not monthly.empty else 0,
            annual[annual['year'] == pd.Timestamp.today().year]['amount'].sum() if not annual.empty else 0
        ]
    })
    return summary, daily, monthly, annual

def get_bot_logs():
    # If you have a log file, read and return the last N lines here
    return []

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_trading = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t24tpvcu.json")

# --- Check API Connection ---
api_ok, api_error = check_api_connection()

st.set_page_config(page_title="AI Automated Trading Bot Dashboard", layout="wide")
# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
body {background-color: #111; color: #fff; font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;}
.stApp {background-color: #111;}
.metric-card {background: #181818; border-radius: 16px; padding: 1.2em 1em; margin-bottom: 1em; box-shadow: 0 2px 8px #0002;}
.stButton>button {background-color: #ff3333; color: #fff; border-radius: 8px; font-weight: bold;}
.red-text { color: #ff3333 !important; }
.green-text { color: #33ff33 !important; }
.status-badge {display: inline-block; padding: 0.25em 0.7em; border-radius: 8px; font-weight: bold; background: #ff3333; color: #fff; margin-left: 0.5em;}
.status-badge.green { background: #33ff33; color: #111; }
.icon {font-size: 1.2em; margin-right: 0.3em; vertical-align: middle;}
/* Card contrast improvements */
.dashboard-card {
  background: #181c20 !important;
  border-radius: 18px !important;
  box-shadow: 0 2px 12px #0006 !important;
  padding: 1.2em 1em !important;
  margin-bottom: 1em !important;
  border-left: 6px solid #007BFF !important;
  color: #f5f6fa !important;
}
.dashboard-card .card-title {
  color: #b0b8c1 !important;
  font-size: 1.1em !important;
  font-weight: 600 !important;
  margin-bottom: 0.2em !important;
}
.dashboard-card .card-value {
  color: #33ff99 !important;
  font-size: 1.5em !important;
  font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)




# --- Top Navigation Bar ---
st.markdown("""
<div style='display: flex; align-items: center; background: #F5F7FA; padding: 0.7em 1.5em; box-shadow: 0 2px 8px #0001; border-radius: 0 0 16px 16px;'>
  <img src='https://cdn-icons-png.flaticon.com/512/2721/2721269.png' style='height:40px; margin-right:1em;'>
  <span style='font-size: 1.7em; font-weight: 700; color: #007BFF; letter-spacing: 1px;'>AutoTrade AI</span>
</div>
""", unsafe_allow_html=True)


# --- Sidebar Navigation Menu (single version, all links) ---
MENU_DASHBOARD = "🏠 Dashboard"
MENU_SCHEDULER = "⏱ Scheduler"
MENU_LIVE_TRADES = "📊 Live Trades"
MENU_BOT_LOGS = "📝 Bot Logs"
MENU_REPORTS = "📤 Reports / Export"
MENU_ACCOUNT = "👤 Account Info"
MENU_SETTINGS = "⚙️ Settings"
MENU_LIST = [MENU_DASHBOARD, MENU_SCHEDULER, MENU_LIVE_TRADES, MENU_BOT_LOGS, MENU_REPORTS, MENU_ACCOUNT, MENU_SETTINGS]
with st.sidebar:
    menu = st.radio("Navigation", MENU_LIST, index=0)
    st.markdown("---")
    st.write("Made with ❤️ for investors.")
    st.markdown("---")


# --- Main Content Area (single version, all menu pages) ---

if menu == MENU_DASHBOARD:
    acc = get_account_info()
    st.markdown("<div style='margin-top:1.5em;'></div>", unsafe_allow_html=True)
    col1, col2, col4 = st.columns([1,1,1])
    with col1:
        st.markdown(f"""
        <div class='dashboard-card' title='View Equity Details'>
            <div class='card-title'>💰 Equity</div>
            <div class='card-value'>${acc['equity']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='dashboard-card' title='View Cash Details'>
            <div class='card-title'>💵 Cash</div>
            <div class='card-value'>${acc['cash']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        # Show last trade info if available, else show 'None'
        last_trade = None
        trades_df = get_recent_trades(max_trades=1)
        if not trades_df.empty:
            t = trades_df.iloc[0]
            last_trade = f"{t['side'].capitalize()} {t['qty']} {t['symbol']} @ ${t['price']:.2f}"
        else:
            last_trade = "None"
        st.markdown(
            f"""
            <div class='dashboard-card' title='Last Trade'>
                <div class='card-title'>📈 Last Trade</div>
                <div class='card-value'>{last_trade}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<div style='margin-bottom:2em;'></div>", unsafe_allow_html=True)
    # --- Live Trades Table on Dashboard ---
    st.subheader("Live Trades (Last 10)")
    trades_df = get_recent_trades(max_trades=10)
    st.dataframe(trades_df, use_container_width=True)
    st.caption("Showing last 10 trades. Table will auto-refresh if enabled.")



elif menu == MENU_SCHEDULER:
    st.header("⏱ Scheduler")
    if 'scheduler_running' not in st.session_state:
        st.session_state['scheduler_running'] = False
    if 'last_run' not in st.session_state:
        st.session_state['last_run'] = 'Never'
    if 'scheduled_time' not in st.session_state:
        st.session_state['scheduled_time'] = None
    st.markdown("<b>📅 Select Start Date</b>", unsafe_allow_html=True)
    start_date = st.date_input("", value=datetime.now().date(), key="start_calendar_date")
    st.markdown("<b>⏰ Select Start Time</b>", unsafe_allow_html=True)
    start_time = st.time_input("", value=datetime.now().time().replace(second=0, microsecond=0), key="start_calendar_time")
    start_dt = datetime.combine(start_date, start_time)
    st.markdown("<b>📅 Select End Date</b>", unsafe_allow_html=True)
    end_date = st.date_input("", value=datetime.now().date(), key="end_calendar_date")
    st.markdown("<b>⏰ Select End Time</b>", unsafe_allow_html=True)
    end_time = st.time_input("", value=datetime.now().time().replace(second=0, microsecond=0), key="end_calendar_time")
    end_dt = datetime.combine(end_date, end_time)
    st.write(f"Start: <b>{start_dt.strftime('%Y-%m-%d %H:%M:%S')}</b>", unsafe_allow_html=True)
    st.write(f"End: <b>{end_dt.strftime('%Y-%m-%d %H:%M:%S')}</b>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        if not st.session_state['scheduler_running']:
            if st.button("Schedule Bot Run"):
                st.session_state['scheduler_running'] = True
                st.session_state['scheduled_time'] = start_dt
                st.session_state['end_time'] = end_dt
                threading.Thread(target=scheduler_thread, args=(start_dt,), daemon=True).start()
        else:
            if st.button("Cancel Scheduled Run"):
                st.session_state['scheduler_running'] = False
                st.session_state['scheduled_time'] = None
                st.session_state['end_time'] = None
    with colB:
        st.write(f"Last run: {st.session_state['last_run']}")
    if st.session_state['scheduler_running']:
        st.success(f"Bot scheduled for {st.session_state['scheduled_time']} (End: {st.session_state['end_time']})")
    else:
        st.info("No scheduled run.")

elif menu == MENU_SETTINGS:
    st.header("⚙️ Settings")
    # All settings are now fully automatic and hidden from the user.
    st.session_state['theme'] = 'Dark'
    st.session_state['default_mode'] = 'Live'
    st.session_state['notifications'] = True
    st.info("All settings are fully automatic. The bot will always use the most aggressive, fully automated mode.")

