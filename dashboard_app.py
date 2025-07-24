import os
import pandas as pd
import streamlit as st
from datetime import datetime
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from bs4 import BeautifulSoup

from scalping_breakout_strategy import detect_breakout, get_take_profit_stop_loss

# --- Global Constants ---
HR_DIVIDER = "<hr style='margin:1.5em 0;'>"
FLAG_FILE = "bot_flag.txt"
MENU_DASHBOARD = "🏠 Dashboard"
MENU_SCHEDULER = "⏰ Scheduler"
MENU_SETTINGS = "⚙️ Settings"

# --- Dummy implementations for missing functions (replace with real logic as needed) ---
def get_account_info():
    # Load credentials from .env and fetch live account info from Alpaca
    from dotenv import load_dotenv
    import os
    import alpaca_trade_api as tradeapi
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL")
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    account = api.get_account()
    return {
        'equity': float(account.equity),
        'cash': float(account.cash)
    }

def get_recent_trades():
    # Replace with actual trade log retrieval logic
    import pandas as pd
    data = [
        {'side': 'buy', 'qty': 10, 'symbol': 'AAPL', 'price': 195.23},
        {'side': 'sell', 'qty': 5, 'symbol': 'TSLA', 'price': 265.10}
    ]
    return pd.DataFrame(data)


# --- Constants ---
BADGE_DIV_OPEN = "<div style='display: flex; flex-wrap: wrap; gap: 0.5em 1em; margin-bottom: 1em;'>"
BADGE_DIV_CLOSE = "</div>"
HTML_PARSER = 'html.parser'

# --- Ticker fetch stub ---
def get_affordable_tickers():
    # Fetch trending tickers from Yahoo Finance (as an example)
    import requests
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        symbols = [item['symbol'] for item in data.get('finance', {}).get('result', [])[0].get('quotes', [])]
        return symbols
    except Exception:
        return []
HTML_PARSER = 'html.parser'

# --- Sidebar function (top-level, not nested) ---
def render_sidebar():
    # (Removed ticker block from sidebar)
    bot_running = st.session_state.get('bot_running', False)
    power_status = "ON" if bot_running else "OFF"
    if st.sidebar.button(f"{'🟢' if bot_running else '🔴'} Power: {power_status}", key="power_btn", help="Toggle bot power"):
        bot_running = not bot_running
        st.session_state['bot_running'] = bot_running
        with open(FLAG_FILE, "w") as f:
            f.write("start" if bot_running else "stop")

    acc = get_account_info()
    st.sidebar.markdown(f"""
<div style='line-height:1.7; margin-bottom:1em; padding:0.7em 0 0.2em 0;'>
    <b>Equity:</b> ${float(acc['equity']):.2f}<br>
    <b>Cash Available:</b> ${float(acc['cash']):.2f}
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("<b>Navigation</b>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)
    menu = st.sidebar.radio(" ", [MENU_DASHBOARD, MENU_SCHEDULER, MENU_SETTINGS], index=0, key="sysnav")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("📈 <b>Trading</b>", unsafe_allow_html=True)
    menu2 = st.sidebar.radio("  ", ["📊 Live Trades", "📤 Reports & Exports", "📝 Bot Logs"], index=0, key="tradenav")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("👤 <b>Account</b>", unsafe_allow_html=True)
    menu3 = st.sidebar.radio("   ", ["👤 Account", "👤 Account Info"], index=0, key="accnav")
    st.sidebar.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)
    st.sidebar.write("Made with ❤️ for investors.")
    st.sidebar.caption("Powered by AutoTrade AI")

    # --- Bot Status Block at very bottom ---
    bot_status = '🟢 Running' if st.session_state.get('bot_running', False) else '🔴 Stopped'
    st.sidebar.markdown(f"""
<div style='line-height:1.7; margin-top:1.5em; padding:0.7em 0 0.2em 0; border-top:1px solid #333;'>
    <b>Bot Status:</b> {bot_status}
</div>
""", unsafe_allow_html=True)
    return menu, menu2, menu3

menu, menu2, menu3 = render_sidebar()


# --- Main Content Area (single version, all menu pages) ---

if menu == MENU_DASHBOARD:

    # --- Dashboard Cards Section ---
    acc = get_account_info()
    trades_df_last = get_recent_trades()
    if not trades_df_last.empty:
        t = trades_df_last.iloc[0]
        last_trade = f"{t['side'].capitalize()} {t['qty']} {t['symbol']} @ ${t['price']:.2f}"
    else:
        last_trade = "No trades found."

    st.markdown("""
<div style='display: flex; gap: 2em; margin-bottom: 2em;'>
  <div style='background: #23272f; border-radius: 16px; box-shadow: 0 2px 8px #0002; padding: 1.5em 2em; min-width: 220px; text-align: left;'>
    <span style='font-size:1.3em;'>🪙 <b>Equity</b></span><br>
    <span style='color:#3cffb3; font-size:2em; font-weight:bold;'>${:.2f}</span>
  </div>
  <div style='background: #23272f; border-radius: 16px; box-shadow: 0 2px 8px #0002; padding: 1.5em 2em; min-width: 220px; text-align: left;'>
    <span style='font-size:1.3em;'>💵 <b>Cash</b></span><br>
    <span style='color:#3cffb3; font-size:2em; font-weight:bold;'>${:.2f}</span>
  </div>
  <div style='background: #23272f; border-radius: 16px; box-shadow: 0 2px 8px #0002; padding: 1.5em 2em; min-width: 220px; text-align: left;'>
    <span style='font-size:1.3em;'>📈 <b>Last Trade</b></span><br>
    <span style='color:#3cffb3; font-size:1.5em; font-weight:bold;'>{}</span>
  </div>
</div>
    """.format(float(acc['equity']), float(acc['cash']), last_trade), unsafe_allow_html=True)

    # --- Trending Tickers Table ---
    tickers = get_affordable_tickers()
    st.markdown("<h4 style='margin-top:0.5em;'>Top 10 Trending Tickers:</h4>", unsafe_allow_html=True)
    if tickers:
        # Display tickers in a 4-column table
        cols = 4
        rows = (len(tickers) + cols - 1) // cols
        table_html = "<table style='width:100%;background:#23272f;border-radius:10px;'><tr>"
        for i in range(cols):
            table_html += f"<th style='color:#ccc;padding:0.5em;'>Ticker {i+1}</th>"
        table_html += "</tr>"
        for r in range(rows):
            table_html += "<tr>"
            for c in range(cols):
                idx = r + rows * c
                if idx < len(tickers):
                    table_html += f"<td style='color:#fff;padding:0.5em;'>{tickers[idx]}</td>"
                else:
                    table_html += "<td></td>"
            table_html += "</tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No tickers found.")


    # --- 📊 Live Trades Section (Last 10) ---
    try:
        from trade_log import get_recent_trades_log
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
        st.markdown("<h3>📊 Live Trades (Last 10)</h3>", unsafe_allow_html=True)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"Live trades unavailable: {e}")

    st.markdown(HR_DIVIDER, unsafe_allow_html=True)

    # --- Last Trade ---
    acc = get_account_info()
    trades_df_last = get_recent_trades()
    if not trades_df_last.empty:
        t = trades_df_last.iloc[0]
        last_trade = f"{t['side'].capitalize()} {t['qty']} {t['symbol']} @ ${t['price']:.2f}"

