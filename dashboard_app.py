
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
    st.title("Trading Bot Dashboard")
    st.caption("Professional Automated Trading Monitoring")

st.markdown("""
<style>
body {background-color: #111; color: #fff;}
.stApp {background-color: #111;}
.metric-card {background: #181818; border-radius: 12px; padding: 1.2em 1em; margin-bottom: 1em; box-shadow: 0 2px 8px #0002;}
.stButton>button {background-color: #ff3333; color: #fff; border-radius: 8px; font-weight: bold;}
.red-text { color: #ff3333 !important; }
.green-text { color: #33ff33 !important; }
.status-badge {display: inline-block; padding: 0.25em 0.7em; border-radius: 8px; font-weight: bold; background: #ff3333; color: #fff; margin-left: 0.5em;}
st.info("This dashboard is a template. Connect it to your live trading bot and logs for real-time monitoring.")

def get_positions():
    # Replace with real Alpaca API call
    return pd.DataFrame([
        {'symbol': 'AAPL', 'qty': 10, 'avg_entry_price': 180.5, 'market_value': 1820.0},
        {'symbol': 'TSLA', 'qty': 2, 'avg_entry_price': 700.0, 'market_value': 1450.0}
    ])

def get_recent_trades():
    # Replace with real trade log reading
    return pd.DataFrame([
        {'time': '2025-07-23 10:00', 'symbol': 'AAPL', 'side': 'buy', 'price': 180.5, 'qty': 10},
        {'time': '2025-07-23 09:30', 'symbol': 'TSLA', 'side': 'sell', 'price': 725.0, 'qty': 1}
    ])

def get_bot_logs():
    # Replace with reading from your log file
    return [
        '2025-07-23 10:00: Bought 10 AAPL at $180.5',
        '2025-07-23 09:30: Sold 1 TSLA at $725.0',
        '2025-07-23 09:00: Bot started.'
    ]

st.markdown("""
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

st.subheader("Recent Trades")
st.dataframe(get_recent_trades())
st.subheader("Bot Logs")
st.code('\n'.join(get_bot_logs()))

# --- Main Card Layout ---
main1, main2 = st.columns([2, 1])
with main1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent Trades")
        st.dataframe(get_recent_trades(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Current Positions")
        st.dataframe(get_positions(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
with main2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Account Info")
        st.write(f"**Status:** {acc['status']}")
        st.write(f"**Equity:** ${acc['equity']:,}")
        st.write(f"**Cash:** ${acc['cash']:,}")
        st.write(f"**Buying Power:** ${acc['buying_power']:,}")
        st.write(f"**Last Update:** {acc['last_update']}")
        st.markdown('</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Bot Logs")
        st.code('\n'.join(get_bot_logs()))
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Account":
elif page == "Positions":
elif page == "Trades":
elif page == "Logs":

# --- Sidebar Navigation (optional, can be expanded) ---
with st.sidebar:
    st.header("Navigation")
    st.write("Select a section above to view details.")
    st.markdown("---")
    st.write("Made with ❤️ for investors.")

st.info("This dashboard is a template. Connect it to your live trading bot and logs for real-time monitoring.")
