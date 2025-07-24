import os
import time
import random
import pandas as pd
import streamlit as st
from datetime import datetime
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from bs4 import BeautifulSoup
from scalping_breakout_strategy import detect_breakout, get_take_profit_stop_loss

# --- Constants ---
HTML_PARSER = 'html.parser'
def get_affordable_tickers(trending, cash, min_trade=1.0):
    affordable = []
    prices = {}
    for symbol in trending:
        try:
            price = float(api.get_latest_trade(symbol).price)
            if price >= min_trade and price <= cash:
                affordable.append(symbol)
                prices[symbol] = price
        except Exception:
            continue
    return affordable, prices

def get_breakout_candidates(affordable):
    candidates = []
    for symbol in affordable:
        try:
            bars = api.get_bars(symbol, '5Min', limit=30).df
            if bars.empty or len(bars) < 21:
                continue
            if detect_breakout(bars):
                candidates.append(symbol)
        except Exception as e:
            print(f"Breakout check failed for {symbol}: {e}")
            continue
    return candidates

def trade_breakout_candidates(candidates, cash):
    for symbol in candidates:
        try:
            price = float(api.get_latest_trade(symbol).price)
            qty = round(cash / price, 3)
            if qty < 0.01:
                continue
            tp, sl = get_take_profit_stop_loss(price, tp_pct=0.01, sl_pct=0.01)
            api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
            try:
                api.submit_order(symbol=symbol, qty=qty, side='sell', type='limit', time_in_force='gtc', limit_price=tp)
                api.submit_order(symbol=symbol, qty=qty, side='sell', type='stop', time_in_force='gtc', stop_price=sl)
                print(f"Scalping breakout: Bought {qty} {symbol} at {price:.2f}, TP {tp:.2f}, SL {sl:.2f}")
            except Exception as e:
                print(f"Could not place TP/SL for {symbol}: {e}")
        except Exception as e:
            print(f"Scalping trade failed for {symbol}: {e}")

def update_last_run():
    st.session_state['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def run_bot_job():
    try:
        trending = get_trending_tickers()
        if not trending:
            print("No trending tickers found. Skipping trade.")
            update_last_run()
            return
        acc = api.get_account()
        cash = float(acc.cash)
        affordable, prices = get_affordable_tickers(trending, cash)
        if not affordable:
            print("No affordable tickers found with current cash. Skipping trade.")
            update_last_run()
            return
        breakout_candidates = get_breakout_candidates(affordable)
        trade_breakout_candidates(breakout_candidates, cash)
        best_symbol = find_first_buy_signal(affordable)
        if best_symbol:
            place_aggressive_trade(best_symbol, price_override=prices.get(best_symbol))
        else:
            print("No buy signal found for affordable tickers.")
            update_last_run()
            return
        update_last_run()
    except Exception as e:
        print(f"Trade error: {e}")
        update_last_run()

# --- Trending Tickers Scraper ---

def get_robinhood_tickers():
    try:
        import robin_stocks.robinhood as r
        url = 'https://robinhood.com/collections/100-most-popular'
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, HTML_PARSER)
        tickers = []
        for a in soup.find_all('a', href=True):
            if '/stocks/' in a['href']:
                ticker = a['href'].split('/')[-1].upper()
                if ticker.isalpha() and len(ticker) <= 6:
                    tickers.append(ticker)
        return list(set(tickers))[:10] if tickers else None
    except Exception as e:
        print(f"Robinhood trending ticker error: {e}")
        return None

def get_webull_tickers():
    try:
        from webull import webull
        wb = webull()
        hot = wb.get_hot_stock_list()
        tickers = [item['ticker'] for item in hot if 'ticker' in item]
        return tickers[:10] if tickers else None
    except Exception as e:
        print(f"Webull trending ticker error: {e}")
        return None

def get_flatfile_tickers():
    csv_path = os.path.join(os.getcwd(), "trending_tickers.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            tickers = df['symbol'].dropna().astype(str).tolist()
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            if tickers:
                return tickers[:10]
        except Exception as e:
            print(f"Error reading trending_tickers.csv: {e}")
    return None

def get_fallback_tickers():
    fallback = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "V",
        "JPM", "PG", "MA", "HD", "LLY", "PEP", "ABBV", "COST", "AVGO", "MRK",
        "KO", "XOM", "WMT", "CVX", "BAC", "MCD", "DIS", "ADBE", "CSCO", "PFE",
        "TMO", "ABT", "CMCSA", "ACN", "DHR", "LIN", "NKE", "TXN", "NEE", "WFC"
    ]
    random.shuffle(fallback)
    return fallback

def get_polygon_tickers():
    polygon_key = os.getenv("POLYGON_API_KEY")
    if not polygon_key:
        return None
    try:
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/actives?apiKey={polygon_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 401 or resp.status_code == 403:
            print("Polygon API key is invalid, expired, or your plan does not allow access to this endpoint. Please check your Polygon.io subscription and API key.")
            return None
        resp.raise_for_status()
        data = resp.json()
        if 'tickers' in data and data['tickers']:
            return [item['ticker'] for item in data['tickers'] if 'ticker' in item][:10]
    except Exception as e:
        print(f"Polygon trending ticker error: {e}")
    return None

def get_yahoo_tickers():
    try:
        url = "https://finance.yahoo.com/most-active"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429:
            print("Trending ticker scrape error: 429 Too Many Requests. Using last cached tickers if available.")
            return None
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, HTML_PARSER)
        table = soup.find('table')
        if not table:
            tables = soup.find_all('table')
            if tables:
                table = tables[0]
        if not table:
            return None
        tickers = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if cols:
                ticker = cols[0].get_text(strip=True)
                if (ticker.replace('.', '').isalpha() and len(ticker) <= 6):
                    tickers.append(ticker)
        return tickers[:10] if tickers else None
    except Exception as e:
        print(f"Trending ticker scrape error: {e}")
        return None

def get_crypto_tickers():
    try:
        crypto_assets = api.list_assets(asset_class='crypto')
        return [a.symbol for a in crypto_assets if a.tradable]
    except Exception:
        return []

def get_trending_tickers():
    now = datetime.now()
    cache_key = 'trending_tickers_cache'
    cache_time_key = 'trending_tickers_cache_time'
    cache_duration = 15 * 60  # 15 minutes in seconds

    # Check cache first
    if (
        cache_key in st.session_state and
        cache_time_key in st.session_state and
        (now - st.session_state[cache_time_key]).total_seconds() < cache_duration
    ):
        tickers = st.session_state[cache_key]
        if tickers:
            return tickers

    # Try each source in order
    sources = [
        get_flatfile_tickers,
        get_webull_tickers,
        get_robinhood_tickers,
        get_polygon_tickers,
        get_yahoo_tickers,
        get_fallback_tickers
    ]
    us_tickers = None
    for src in sources:
        us_tickers = src()
        if us_tickers:
            break

    crypto_tickers = get_crypto_tickers()
    all_tickers = list(set((us_tickers or []) + (crypto_tickers or [])))
    st.session_state[cache_key] = all_tickers
    st.session_state[cache_time_key] = now
    return all_tickers


from prophet import Prophet
from textblob import TextBlob
import requests

def get_news_headlines(symbol):
    try:
        url = f'https://finance.yahoo.com/quote/{symbol}/news?p={symbol}'
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, HTML_PARSER)
        headlines = [el.get_text(strip=True) for el in soup.select('h3')]
        return headlines[:5]  # Use top 5 headlines
    except Exception:
        return []

def get_avg_sentiment(headlines):
    if not headlines:
        return 0
    sentiment_scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return sum(sentiment_scores) / len(sentiment_scores)

def prophet_predict_up(symbol):
    try:
        bars = api.get_bars(symbol, '15Min', limit=200).df
        if bars.empty:
            bars = api.get_bars(symbol, '5Min', limit=200).df
        if bars.empty:
            print(f"No Alpaca data for {symbol}, skipping.")
            return None, None, None
        df = bars.reset_index()[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        m = Prophet(daily_seasonality=False, weekly_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=1, freq='15min')
        forecast = m.predict(future)
        last_actual = df['y'].iloc[-1]
        next_pred = forecast['yhat'].iloc[-1]
        return last_actual, next_pred, None
    except Exception as e:
        print(f"Prophet prediction failed for {symbol}: {e}")
        return None, None, e

def find_first_buy_signal(symbols):
    for symbol in symbols:
        last_actual, next_pred, pred_err = prophet_predict_up(symbol)
        if pred_err or last_actual is None or next_pred is None:
            continue
        headlines = get_news_headlines(symbol)
        avg_sentiment = get_avg_sentiment(headlines)
        if next_pred > last_actual and avg_sentiment > 0:
            print(f"{symbol}: Prophet up, Sentiment {avg_sentiment:.2f} (POSITIVE) => BUY")
            return symbol
        else:
            print(f"{symbol}: Prophet {'up' if next_pred > last_actual else 'down'}, Sentiment {avg_sentiment:.2f}")
    return None


def place_aggressive_trade(symbol, price_override=None):
    def _place(symbol, qty):
        tif = 'day' if qty % 1 != 0 else 'gtc'
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force=tif
        )
        print(f"Placed market buy order for {qty} shares of {symbol} (TIF: {tif}).")

    if not symbol:
        print("No buy signal found for any symbol. Skipping trade.")
        return
    acc = api.get_account()
    cash = float(acc.cash)
    if price_override is not None:
        price = price_override
    else:
        price = float(api.get_latest_trade(symbol).price)
    # Fractional shares: use all available cash, round to 3 decimals
    qty = round(cash / price, 3)
    if qty >= 0.01:
        _place(symbol, qty)
    else:
        print(f"Insufficient cash to buy {symbol}. Skipping trade.")
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
# --- Cleaned, single-version dashboard app ---
import streamlit as st
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import alpaca_trade_api as tradeapi
from bs4 import BeautifulSoup

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
        soup = BeautifulSoup(response.text, HTML_PARSER)
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
HR_DIVIDER = "<hr style='margin:1em 0;'>"

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

# --- Recent Trades Helper ---
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
            page_token = None
            break
    if not all_activities:
        return pd.DataFrame(columns=['time', 'symbol', 'side', 'price', 'qty'])
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

# --- Modern Professional Dark Mode CSS ---
st.markdown("""
<style>
body, .stApp {
    background-color: #181a20 !important;
    color: #f5f6fa !important;
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
}
.stSidebar {
    background: #1e2128 !important;
    color: #f5f6fa !important;
}
.stButton>button {
    background-color: #007BFF !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    border: none !important;
    box-shadow: 0 2px 8px #0002 !important;
    transition: background 0.2s;
}
.stButton>button:hover {
    background-color: #0056b3 !important;
}
.dashboard-card {
    background: #23272f !important;
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
.stDataFrame, .stTable {
    background: #23272f !important;
    color: #f5f6fa !important;
    border-radius: 12px !important;
}
.stMarkdown, .stText, .stCaption {
    color: #b0b8c1 !important;
}
.stRadio > div {
    background: #23272f !important;
    border-radius: 10px !important;
    color: #f5f6fa !important;
}
.stExpanderHeader {
    color: #33ff99 !important;
}
.stSlider > div {
    color: #33ff99 !important;
}
.stAlert {
    background: #23272f !important;
    color: #ff3333 !important;
    border-radius: 10px !important;
}
.stFooter {
    color: #b0b8c1 !important;
    font-size: 0.9em !important;
    text-align: center !important;
    margin-top: 2em !important;
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

# --- Grouped and relabeled navigation ---
MENU_SYSTEM = "🔧 System"
MENU_DASHBOARD = "🏠 Dashboard"
MENU_SCHEDULER = "⏱ Scheduler"
MENU_SETTINGS = "⚙️ Settings"
MENU_TRADING = "📈 Trading"
MENU_LIVE_TRADES = "📊 Live Trades"
MENU_REPORTS = "📤 Reports & Exports"
MENU_BOT_LOGS = "📝 Bot Logs"
MENU_ACCOUNT = "👤 Account"
MENU_ACCOUNT_INFO = "👤 Account Info"

MENU_LIST = [
    MENU_SYSTEM,
    MENU_DASHBOARD,
    MENU_SCHEDULER,
    MENU_SETTINGS,
    MENU_TRADING,
    MENU_LIVE_TRADES,
    MENU_REPORTS,
    MENU_BOT_LOGS,
    MENU_ACCOUNT,
    MENU_ACCOUNT_INFO
]
with st.sidebar:
    acc = get_account_info()
    st.markdown("""
        <div style='line-height:1.7; margin-bottom:1em; padding:0.7em 0 0.2em 0;'>
            <b>💰 Equity:</b> ${equity}<br>
            <b>💵 Cash Available:</b> ${cash}
        </div>
    """.format(
        equity=f"{float(acc['equity']):.2f}",
        cash=f"{float(acc['cash']):.2f}"
    ), unsafe_allow_html=True)

    st.markdown("<b>Navigation</b>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)
    menu = st.radio(" ", [MENU_DASHBOARD, MENU_SCHEDULER, MENU_SETTINGS], index=0, key="sysnav")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("📈 <b>Trading</b>", unsafe_allow_html=True)
    menu2 = st.radio("  ", ["📊 Live Trades", "📤 Reports & Exports", "📝 Bot Logs"], index=0, key="tradenav")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("👤 <b>Account</b>", unsafe_allow_html=True)
    menu3 = st.radio("   ", ["👤 Account", "👤 Account Info"], index=0, key="accnav")
    st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)
    st.write("Made with ❤️ for investors.")
    st.caption("Powered by AutoTrade AI")

    # --- Bot Status Block at very bottom ---
    bot_status = '🟢 Running' if st.session_state.get('bot_running', False) else '🔴 Stopped'
    st.markdown("""
        <div style='line-height:1.7; margin-top:1.5em; padding:0.7em 0 0.2em 0; border-top:1px solid #333;'>
            <b>Bot Status:</b> {bot_status}
        </div>
    """.format(bot_status=bot_status), unsafe_allow_html=True)


# --- Main Content Area (single version, all menu pages) ---

if menu == MENU_DASHBOARD:

    # --- 📊 Live Trades Section (now at top) ---
    from trade_log import get_recent_trades_log
    st.markdown("<h3>📊 Live Trades (Last 10)</h3>", unsafe_allow_html=True)
    trades_df = get_recent_trades_log(10)
    trades_df = trades_df.rename(columns={
        'timestamp': 'Timestamp',
        'symbol': 'Symbol',
        'action': 'Action',
        'price': 'Price',
        'qty': 'Quantity',
        'reason': 'Reason'
    })
    trades_df = trades_df[['Timestamp', 'Symbol', 'Action', 'Price', 'Quantity', 'Reason']]
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    auto_refresh = st.session_state.get('auto_refresh', True)
    st.caption(f"[Auto-refresh: {'ON' if auto_refresh else 'OFF'}]")
    st.markdown(HR_DIVIDER, unsafe_allow_html=True)

    # --- Last Trade (below live trades) ---
    acc = get_account_info()
    trades_df_last = get_recent_trades(max_trades=1)
    if not trades_df_last.empty:
        t = trades_df_last.iloc[0]
        last_trade = f"{t['side'].capitalize()} {t['qty']} {t['symbol']} @ ${t['price']:.2f}"
    else:
        last_trade = "None"
    st.markdown(f"""
        <div style='line-height:1.7;'>
            <b>📈 Last Trade:</b> {last_trade}
        </div>
    """, unsafe_allow_html=True)
    st.markdown(HR_DIVIDER, unsafe_allow_html=True)

    # --- Trending Tickers (below last trade) ---
    trending_tickers = get_trending_tickers()
    st.markdown("<h3>🔥 Top 10 Trending Tickers</h3>", unsafe_allow_html=True)
    if trending_tickers:
        badge_html = "<div style='display: flex; flex-wrap: wrap; gap: 0.5em 1em; margin-bottom: 1em;'>"
        for i, ticker in enumerate(trending_tickers[:10]):
            badge_html += f"<span style='background: #23272f; color: #33ff99; border-radius: 12px; padding: 0.5em 1.2em; font-weight: 600; font-size: 1.1em; margin-bottom: 0.3em; display: inline-block;'>{ticker}</span>"
            if (i+1) % 4 == 0:
                badge_html += "<br>"
        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)
        if len(trending_tickers) > 10:
            with st.expander(f"Show {len(trending_tickers)-10} More Tickers", expanded=False):
                more_badge_html = "<div style='display: flex; flex-wrap: wrap; gap: 0.5em 1em; margin-bottom: 1em;'>"
                for i, ticker in enumerate(trending_tickers[10:]):
                    more_badge_html += f"<span style='background: #23272f; color: #33ff99; border-radius: 12px; padding: 0.5em 1.2em; font-weight: 600; font-size: 1.1em; margin-bottom: 0.3em; display: inline-block;'>{ticker}</span>"
                    if (i+1) % 4 == 0:
                        more_badge_html += "<br>"
                more_badge_html += "</div>"
                st.markdown(more_badge_html, unsafe_allow_html=True)
    else:
        st.info("No trending tickers found.")
    st.markdown(HR_DIVIDER, unsafe_allow_html=True)

elif menu == MENU_SCHEDULER:
    st.header(MENU_SCHEDULER)
    st.info("Manual Start/Stop is now available on the Dashboard. Scheduler is disabled.")

elif menu == MENU_SETTINGS:
    st.header("⚙️ Settings")
    # All settings are now fully automatic and hidden from the user.
    st.session_state['theme'] = 'Dark'
    st.session_state['default_mode'] = 'Live'
    st.session_state['notifications'] = True
    st.info("All settings are fully automatic. The bot will always use the most aggressive, fully automated mode.")

