"""Demo Streamlit stub — use dashboard_app.py for the production UI."""

import streamlit as st

st.set_page_config(page_title="AI AutoBot Trader", layout="wide")
st.title("AI AutoBot Trader")
st.info(
    "This is a legacy demo page. Launch the full dashboard with:\n\n"
    "`streamlit run dashboard_app.py`"
)
st.code("streamlit run dashboard_app.py", language="bash")
