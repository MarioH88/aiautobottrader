CARD_DIV_OPEN = '<div class="card">'
CARD_DIV_CLOSE = '</div>'
BOT_LOGS_LABEL = "Bot Logs"
import alpaca_trade_api as tradeapi
import os

ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

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
        'last_update': acc.updated_at
    }

def get_recent_trades():
    activities = api.get_activities(activity_types='FILL', direction='desc', page_size=10)
    return pd.DataFrame([{
        'time': a.transaction_time,
        'symbol': a.symbol,
        'side': a.side,
        'price': float(a.price),
        'qty': float(a.qty)
    } for a in activities])

def get_bot_logs():
    # If you have a log file, read and return the last N lines here
    return []

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_trading = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t24tpvcu.json")

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")

# --- Header with animation ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st_lottie(lottie_trading, height=80, key="trading")
with col_title:
    st.title("AI Automated Trading Bot Dashboard")
    st.caption("Just Sit Back and Relax, Your Bot is Trading for You")

st.markdown("""
<style>
body {background-color: #111; color: #fff;}
.stApp {background-color: #111;}
.metric-card {background: #181818; border-radius: 12px; padding: 1.2em 1em; margin-bottom: 1em; box-shadow: 0 2px 8px #0002;}
.stButton>button {background-color: #ff3333; color: #fff; border-radius: 8px; font-weight: bold;}
.red-text { color: #ff3333 !important; }
.status-badge {display: inline-block; padding: 0.25em 0.7em; border-radius: 8px; font-weight: bold; background: #ff3333; color: #fff; margin-left: 0.5em;}
st.info("This dashboard is a template. Connect it to your live trading bot and logs for real-time monitoring.")


<style>
body {
    background-color: #111;
    color: #fff;
}
section.main > div {
    background: #181818;
    border-radius: 10px;
    padding: 2rem;
}
[data-testid="stHeader"] {background: #111;}
[data-testid="stSidebar"] {background: #181818;}
.red-text { color: #ff3333 !important; }
.green-text { color: #33ff33 !important; }
.status-badge {
    display: inline-block;
    padding: 0.25em 0.7em;
    border-radius: 8px;
    font-weight: bold;
    background: #ff3333;
    color: #fff;
    margin-left: 0.5em;
}
.status-badge.green { background: #33ff33; color: #111; }
</style>
""", unsafe_allow_html=True)


st.dataframe(get_recent_trades())
st.subheader(BOT_LOGS_LABEL)
st.code('\n'.join(get_bot_logs()))



st.subheader("Recent Trades")
st.dataframe(get_recent_trades())
st.subheader(BOT_LOGS_LABEL)
st.code('\n'.join(get_bot_logs()))

# --- Main Card Layout ---
main1, main2 = st.columns([2, 1])
with main1:
    with st.container():
        st.markdown(CARD_DIV_OPEN, unsafe_allow_html=True)
        st.subheader("Recent Trades")
        st.dataframe(get_recent_trades(), use_container_width=True)
        st.markdown(CARD_DIV_CLOSE, unsafe_allow_html=True)
    with st.container():
        st.markdown(CARD_DIV_OPEN, unsafe_allow_html=True)
        st.subheader("Current Positions")
        st.dataframe(get_positions(), use_container_width=True)
        st.markdown(CARD_DIV_CLOSE, unsafe_allow_html=True)
with main2:
    with st.container():
        st.markdown(CARD_DIV_OPEN, unsafe_allow_html=True)
        st.subheader("Account Info")
        acc = get_account_info()
        st.write(f"**Status:** {acc['status']}")
        st.write(f"**Equity:** ${acc['equity']:,}")
        st.write(f"**Cash:** ${acc['cash']:,}")
        st.write(f"**Buying Power:** ${acc['buying_power']:,}")
        st.write(f"**Last Update:** {acc['last_update']}")
        st.markdown(CARD_DIV_CLOSE, unsafe_allow_html=True)
    with st.container():
        st.markdown(CARD_DIV_OPEN, unsafe_allow_html=True)
        st.subheader(BOT_LOGS_LABEL)
        st.code('\n'.join(get_bot_logs()))
        st.markdown(CARD_DIV_CLOSE, unsafe_allow_html=True)




# --- Sidebar Navigation (optional, can be expanded) ---
with st.sidebar:
    st.header("Navigation")
    st.write("Select a section above to view details.")
    st.markdown("---")
    st.write("Made with ❤️ for investors.")

st.info("This dashboard is a template. Connect it to your live trading bot and logs for real-time monitoring.")
