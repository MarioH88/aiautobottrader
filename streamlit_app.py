
import streamlit as st
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Trading Bot Prediction", layout="centered")
st.title(":crystal_ball: Trading Bot Prediction")

# Lottie animation for visual appeal
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_trading = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t24tpvcu.json")
st_lottie(lottie_trading, height=180, key="trading")

st.markdown("""
<style>
.main {
    background-color: #181818;
    color: #fff;
}
.stButton>button {
    background-color: #ff3333;
    color: #fff;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("Enter your indicator values and the amount you want to trade below.")

# Metrics row for quick stats (demo values)
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Direction", "Buy", "+85%", delta_color="normal")
col2.metric("Confidence", "92%", "+7%", delta_color="normal")
col3.metric("Risk Level", "Low", "-2%", delta_color="inverse")

# Stylish input form
with st.form("trade_form"):
    st.subheader(":chart_with_upwards_trend: Trade Input")
    indicator1 = st.number_input("SMA Value", min_value=0.0, value=50.0)
    indicator2 = st.number_input("RSI Value", min_value=0.0, max_value=100.0, value=30.0)
    amount = st.number_input("Amount to Trade ($)", min_value=0.0, value=1000.0)
    submitted = st.form_submit_button("Predict & Trade")
    if submitted:
        st.success(f"Prediction: Buy | Confidence: 92% | Amount: ${amount}")

