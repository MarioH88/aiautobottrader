import os
import logging
from datetime import timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

from bot.dashboard_api import (
    DashboardAlpaca,
    snapshot_last_price,
    snapshot_prev_close,
    snapshot_quote,
    snapshot_trade_time,
)
from bot.flag_file import read_flag, write_flag
from bot.scheduler import (
    DEFAULT_SCHEDULER_CONFIG,
    describe_active_window,
    is_trading_window_active,
    load_scheduler_config,
    save_scheduler_config,
    validate_hhmm,
)
from bot.symbols import normalize_symbol

FLAG_FILE = "bot_control.flag"
SCHEDULER_FILE = "scheduler_config.json"
MENU_DASHBOARD = "Dashboard"
MENU_SCHEDULER = "Scheduler"
MENU_SETTINGS = "Settings"
THEME_LIGHT = "light"
THEME_DARK = "dark"
POPULAR_SYMBOLS = [
    "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD", "NFLX", "SPY", "QQQ",
    "BTC/USD", "ETH/USD",
]


def get_configured_symbols() -> list[str]:
    """Return symbols the auto-bot watches from .env."""
    raw = os.getenv("TRADING_SYMBOLS", "AAPL,TSLA")
    return [normalize_symbol(symbol) for symbol in raw.split(",") if symbol.strip()]


def get_tradeable_symbols() -> list[str]:
    """Combine bot symbols, popular tickers, and trending names for the UI."""
    symbols = set(get_configured_symbols())
    symbols.update(POPULAR_SYMBOLS)
    symbols.update(normalize_symbol(s) for s in get_trending_tickers())
    return sorted(symbols)


def _theme_css(theme: str) -> str:
    """Return modern CSS overrides for the selected theme."""
    if theme == THEME_DARK:
        tokens = {
            "bg": "#070b12",
            "bg2": "#0d1320",
            "surface": "#121a2b",
            "surface2": "#182236",
            "border": "#243049",
            "text": "#f1f5f9",
            "muted": "#94a3b8",
            "accent": "#818cf8",
            "accent2": "#6366f1",
            "success": "#34d399",
            "danger": "#f87171",
            "shadow": "0 8px 32px rgba(0,0,0,0.35)",
            "hero": "linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #0f172a 100%)",
        }
    else:
        tokens = {
            "bg": "#f4f7fb",
            "bg2": "#eef2f8",
            "surface": "#ffffff",
            "surface2": "#f8fafc",
            "border": "#e2e8f0",
            "text": "#0f172a",
            "muted": "#64748b",
            "accent": "#4f46e5",
            "accent2": "#6366f1",
            "success": "#059669",
            "danger": "#dc2626",
            "shadow": "0 8px 30px rgba(15,23,42,0.08)",
            "hero": "linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #ffffff 100%)",
        }

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: radial-gradient(circle at top right, {tokens['bg2']} 0%, {tokens['bg']} 45%);
            color: {tokens['text']};
        }}

        .block-container {{
            padding-top: 1.5rem;
            max-width: 1200px;
        }}

        [data-testid="stSidebar"] {{
            background: {tokens['surface']};
            border-right: 1px solid {tokens['border']};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem;
        }}

        .sidebar-brand {{
            padding: 1rem 0.25rem 0.5rem;
        }}
        .sidebar-brand h2 {{
            margin: 0;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: {tokens['text']} !important;
        }}
        .sidebar-brand p {{
            margin: 0.25rem 0 0;
            font-size: 0.78rem;
            color: {tokens['muted']} !important;
        }}

        .hero-banner {{
            background: {tokens['hero']};
            border: 1px solid {tokens['border']};
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: {tokens['shadow']};
        }}
        .hero-banner h1 {{
            margin: 0;
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: {tokens['text']} !important;
        }}
        .hero-banner p {{
            margin: 0.35rem 0 0;
            color: {tokens['muted']} !important;
            font-size: 0.95rem;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {tokens['surface']};
            border-color: {tokens['border']} !important;
            border-radius: 16px !important;
            box-shadow: {tokens['shadow']};
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.75rem;
        }}
        .section-title {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {tokens['muted']} !important;
            margin: 0 0 0.75rem 0;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid {tokens['border']};
            background: {tokens['surface2']};
            color: {tokens['text']} !important;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        .ticker-chip {{
            display: inline-block;
            padding: 0.45rem 0.8rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            background: {tokens['surface2']};
            border: 1px solid {tokens['border']};
            color: {tokens['text']} !important;
        }}

        [data-testid="stMetric"] {{
            background: {tokens['surface']};
            border: 1px solid {tokens['border']};
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: {tokens['shadow']};
        }}
        [data-testid="stMetricLabel"] {{
            color: {tokens['muted']} !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        [data-testid="stMetricValue"] {{
            color: {tokens['text']} !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {tokens['border']};
            border-radius: 12px;
            overflow: hidden;
        }}

        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid {tokens['border']};
            transition: all 0.15s ease;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {tokens['accent2']} 0%, {tokens['accent']} 100%);
            border: none;
            color: white !important;
        }}

        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
            border-radius: 10px !important;
        }}

        hr {{
            border: none;
            border-top: 1px solid {tokens['border']} !important;
            margin: 1rem 0 !important;
        }}

        h1, h2, h3, h4, label, p, span, .stCaption {{
            color: {tokens['text']};
        }}
        .stCaption {{
            color: {tokens['muted']} !important;
        }}
    </style>
    """


def _status_meta(status: str) -> tuple[str, str, str]:
    """Return label, dot color, and pill class for bot status."""
    mapping = {
        "start": ("Live", "#34d399", "running"),
        "stop": ("Stopped", "#94a3b8", "stopped"),
        "paused_by_schedule": ("Paused by Schedule", "#fbbf24", "idle"),
        "emergency_stop": ("Emergency Halt", "#f87171", "halt"),
    }
    return mapping.get(status, ("Not Configured", "#fbbf24", "idle"))


def resolve_bot_display_status() -> str:
    """Combine flag + scheduler into a dashboard-facing status."""
    flag = read_bot_status()
    schedule = load_scheduler_config(SCHEDULER_FILE)
    if flag == "emergency_stop":
        return "emergency_stop"
    if schedule.get("enabled") and not is_trading_window_active(schedule) and flag == "start":
        return "paused_by_schedule"
    return flag


def render_hero(title: str, subtitle: str) -> None:
    """Render a modern page hero banner."""
    status = resolve_bot_display_status()
    label, dot_color, _ = _status_meta(status)
    st.markdown(
        f"""
        <div class="hero-banner">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;">
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
                <span class="status-pill">
                    <span class="status-dot" style="background:{dot_color};"></span>
                    {label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_in_card(title: str):
    """Return a bordered container with a section title."""
    container = st.container(border=True)
    container.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
    return container


def init_theme() -> str:
    """Initialize theme preference in session state."""
    if "theme" not in st.session_state:
        st.session_state.theme = THEME_LIGHT
    return st.session_state.theme


def apply_theme() -> None:
    """Inject CSS for the active theme."""
    st.markdown(_theme_css(st.session_state.theme), unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Sidebar control to switch between light and dark mode."""
    st.sidebar.markdown("---")
    st.sidebar.caption("Appearance")
    choice = st.sidebar.radio(
        "Theme",
        options=[THEME_LIGHT, THEME_DARK],
        format_func=lambda value: "Light" if value == THEME_LIGHT else "Dark",
        horizontal=True,
        index=0 if st.session_state.theme == THEME_LIGHT else 1,
    )
    st.session_state.theme = choice


def init_popup_state() -> None:
    """Track whether the author popup was dismissed this session."""
    if "popup_dismissed" not in st.session_state:
        st.session_state.popup_dismissed = False


@st.dialog("AI AutoBot Trader")
def author_popup() -> None:
    """Startup popup — author: Mario Hayden."""
    st.markdown("### Welcome")
    st.caption("Your trading dashboard is ready.")
    st.markdown("---")
    st.markdown("Author: **Mario Hayden**")
    if st.button("OK", type="primary", use_container_width=True):
        st.session_state.popup_dismissed = True
        st.rerun()


def show_startup_popup() -> None:
    """Show the author popup once when the app runs."""
    if not st.session_state.popup_dismissed:
        author_popup()


@st.cache_resource
def init_alpaca_api():
    try:
        api = DashboardAlpaca()
        _ = api.get_account()
        return api
    except Exception as e:
        st.error(f"Failed to connect to Alpaca API: {e}")
        logger.exception("Alpaca init failed")
        return None


def get_account_info():
    api = init_alpaca_api()
    if not api:
        return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0, "status": "disconnected"}
    try:
        acc = api.get_account()
        return {
            "equity": float(acc.equity),
            "cash": float(acc.cash),
            "buying_power": float(acc.buying_power),
            "status": str(acc.status),
        }
    except Exception as e:
        st.warning(f"Account info unavailable: {e}")
        return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0, "status": "error"}


def get_positions():
    api = init_alpaca_api()
    if not api:
        return []
    try:
        return api.list_positions()
    except Exception as e:
        st.warning(f"Positions unavailable: {e}")
        return []


def get_position_qty(symbol: str) -> float:
    """Return held quantity for a symbol, or 0."""
    symbol = normalize_symbol(symbol)
    for position in get_positions():
        if normalize_symbol(position.symbol) == symbol:
            return float(position.qty)
    return 0.0


def fetch_live_prices(symbols: list[str]) -> pd.DataFrame:
    """Fetch latest trade, bid/ask, and day change from Alpaca snapshots."""
    api = init_alpaca_api()
    if not api:
        return pd.DataFrame()
    unique = sorted({normalize_symbol(symbol) for symbol in symbols if symbol and str(symbol).strip()})
    if not unique:
        return pd.DataFrame()

    rows: list[dict] = []
    try:
        snapshots = api.get_snapshots(unique)
        for symbol in unique:
            snap = snapshots.get(symbol)
            if snap is None:
                continue
            price = snapshot_last_price(snap)
            bid, ask = snapshot_quote(snap)
            prev_close = snapshot_prev_close(snap)
            change_pct = None
            if price is not None and prev_close:
                change_pct = ((price - prev_close) / prev_close) * 100
            updated = snapshot_trade_time(snap)
            if updated is not None:
                try:
                    updated = updated.astimezone(timezone.utc).strftime("%H:%M:%S UTC")
                except Exception:
                    updated = str(updated)
            rows.append(
                {
                    "Symbol": symbol,
                    "Last": price,
                    "Bid": bid,
                    "Ask": ask,
                    "Change %": change_pct,
                    "Updated": updated or "—",
                }
            )
    except Exception as exc:
        logger.exception("Live price fetch failed")
        st.warning(f"Live prices unavailable: {exc}")
    return pd.DataFrame(rows)


def _price_watchlist() -> list[str]:
    """Build the symbol list for the live prices panel."""
    symbols = set(get_configured_symbols())
    symbols.update(POPULAR_SYMBOLS[:8])
    symbols.update(normalize_symbol(s) for s in get_trending_tickers()[:5])
    for position in get_positions():
        symbols.add(normalize_symbol(position.symbol))
    selected = st.session_state.get("manual_symbol")
    if selected:
        symbols.add(normalize_symbol(str(selected)))
    custom = st.session_state.get("custom_symbol_input", "")
    if custom.strip():
        symbols.add(normalize_symbol(custom))
    return sorted(symbols)


@st.fragment(run_every=timedelta(seconds=10))
def render_live_prices() -> None:
    """Auto-refreshing live market price table."""
    card = render_in_card("Live Market Prices")
    with card:
        st.caption("Updates every 10 seconds via Alpaca Market Data")
        frame = fetch_live_prices(_price_watchlist())
        if frame.empty:
            st.info("No live prices available right now.")
            return

        display = frame.copy()
        display["Last"] = display["Last"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
        display["Bid"] = display["Bid"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
        display["Ask"] = display["Ask"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
        display["Change %"] = display["Change %"].map(
            lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)


def show_symbol_live_price(symbol: str) -> None:
    """Show a large live price readout for one symbol."""
    frame = fetch_live_prices([symbol])
    if frame.empty or frame.iloc[0]["Last"] is None:
        st.caption("Live price unavailable")
        return
    row = frame.iloc[0]
    price = row["Last"]
    change = row["Change %"]
    bid, ask = row["Bid"], row["Ask"]
    c1, c2 = st.columns(2)
    c1.metric("Live Price", f"${price:,.2f}")
    if change is not None and pd.notna(change):
        c2.metric("Day Change", f"{change:+.2f}%")
    elif bid and ask:
        c2.metric("Bid / Ask", f"${bid:,.2f} / ${ask:,.2f}")
    else:
        c2.metric("Updated", row["Updated"])


@st.fragment(run_every=timedelta(seconds=10))
def render_selected_symbol_price(symbol: str) -> None:
    """Refresh live price for the manually selected symbol."""
    show_symbol_live_price(symbol)


def submit_manual_order(symbol: str, side: str, qty: float) -> tuple[bool, str]:
    """Submit a market buy or sell order via Alpaca."""
    api = init_alpaca_api()
    if not api:
        return False, "Alpaca API not connected"
    symbol = normalize_symbol(symbol)
    if not symbol:
        return False, "Enter a valid symbol"
    if qty <= 0:
        return False, "Quantity must be greater than 0"
    try:
        api.submit_market_order(symbol=symbol, side=side, qty=qty)
        return True, f"{side.upper()} order submitted: {qty} of {symbol}"
    except Exception as exc:
        logger.exception("Manual order failed")
        return False, str(exc)


def get_recent_trades():
    try:
        from trade_log import get_recent_trades_log
        df = get_recent_trades_log(10)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "price", "qty", "reason"])


def get_trending_tickers():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        js = r.json()
        quotes = js.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
        return symbols[:10] if symbols else ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
    except Exception:
        return ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]


def read_bot_status():
    return read_flag(FLAG_FILE) or "not_configured"


def write_bot_status(status: str):
    if write_flag(FLAG_FILE, status):
        return True
    st.error("Failed to write bot flag")
    return False


def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <h2>AutoBot Trader</h2>
            <p>AI-powered execution desk</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status = resolve_bot_display_status()
    label, dot_color, _ = _status_meta(status)
    st.sidebar.markdown(
        f'<span class="status-pill"><span class="status-dot" style="background:{dot_color};"></span>{label}</span>',
        unsafe_allow_html=True,
    )

    running = status == "start"
    power_label = "Pause Bot" if running or status == "paused_by_schedule" else "Start Bot"
    if st.sidebar.button(power_label, type="primary", use_container_width=True):
        running = not running
        st.session_state["bot_running"] = running
        write_bot_status("start" if running else "stop")

    acc = get_account_info()
    st.sidebar.markdown("##### Portfolio")
    st.sidebar.metric("Equity", f"${acc['equity']:,.2f}")
    st.sidebar.metric("Cash", f"${acc['cash']:,.2f}")
    st.sidebar.metric("Buying Power", f"${acc['buying_power']:,.2f}")

    st.sidebar.markdown("---")
    render_theme_toggle()
    menu = st.sidebar.radio(
        "Navigation",
        [MENU_DASHBOARD, MENU_SCHEDULER, MENU_SETTINGS],
        index=0,
        format_func=lambda item: {"Dashboard": "Dashboard", "Scheduler": "Scheduler", "Settings": "Settings"}[item],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Author: Mario Hayden")
    return menu


def render_dashboard():
    render_hero("Trading Dashboard", "Monitor positions, place orders, and control the auto-bot.")

    acc = get_account_info()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio", f"${acc['equity']:,.2f}")
    k2.metric("Cash", f"${acc['cash']:,.2f}")
    k3.metric("Buying Power", f"${acc['buying_power']:,.2f}")
    k4.metric("Account", acc["status"].upper())

    render_live_prices()

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        card = render_in_card("Bot Controls")
        with card:
            b1, b2, b3 = st.columns(3)
            if b1.button("Start", type="primary", use_container_width=True):
                write_bot_status("start")
                st.session_state["bot_running"] = True
                st.toast("Bot started", icon="✅")
            if b2.button("Stop", use_container_width=True):
                write_bot_status("stop")
                st.session_state["bot_running"] = False
                st.toast("Bot stopped", icon="⏸️")
            if b3.button("Emergency", use_container_width=True):
                write_bot_status("emergency_stop")
                st.session_state["bot_running"] = False
                st.toast("Emergency stop activated", icon="🛑")

        configured = get_configured_symbols()
        card = render_in_card("Manual Trade")
        with card:
            st.caption(f"Auto-bot watches: {', '.join(configured)}")
            trade_symbols = get_tradeable_symbols()
            symbol = st.selectbox("Symbol", trade_symbols, key="manual_symbol", label_visibility="collapsed")
            custom_symbol = st.text_input(
                "Custom symbol",
                placeholder="Type any ticker, e.g. MSFT",
                key="custom_symbol_input",
            )
            if custom_symbol.strip():
                symbol = normalize_symbol(custom_symbol)

            render_selected_symbol_price(symbol)

            held_qty = get_position_qty(symbol)
            q1, q2, q3, q4 = st.columns([1.2, 1, 1, 1])
            with q1:
                default_qty = max(held_qty, 1.0) if held_qty > 0 else 1.0
                qty = st.number_input("Qty", min_value=0.01, value=float(default_qty), step=1.0)
                if held_qty > 0:
                    st.caption(f"Holding {held_qty:g} shares")
            with q2:
                if st.button("Buy", type="primary", use_container_width=True):
                    ok, msg = submit_manual_order(symbol, "buy", qty)
                    st.toast(msg, icon="✅" if ok else "⚠️")
                    if not ok:
                        st.error(msg)
            with q3:
                if st.button("Sell", use_container_width=True):
                    sell_qty = held_qty if held_qty > 0 else qty
                    ok, msg = submit_manual_order(symbol, "sell", sell_qty)
                    st.toast(msg, icon="✅" if ok else "⚠️")
                    if not ok:
                        st.error(msg)
            with q4:
                if st.button("Sell All", use_container_width=True):
                    if held_qty <= 0:
                        st.warning(f"No {symbol} position")
                    else:
                        ok, msg = submit_manual_order(symbol, "sell", held_qty)
                        st.toast(msg, icon="✅" if ok else "⚠️")
                        if not ok:
                            st.error(msg)

        card = render_in_card("Trending")
        with card:
            chips = "".join(f'<span class="ticker-chip">{tk}</span>' for tk in get_trending_tickers()[:10])
            st.markdown(chips, unsafe_allow_html=True)

    with right:
        card = render_in_card("Open Positions")
        with card:
            positions = get_positions()
            if positions:
                rows = []
                for p in positions:
                    pl = float(p.unrealized_pl)
                    rows.append({
                        "Symbol": p.symbol,
                        "Qty": float(p.qty),
                        "Entry": f"${float(p.avg_entry_price):.2f}",
                        "Price": f"${float(p.current_price):.2f}",
                        "Value": f"${float(p.market_value):,.2f}",
                        "P&L": f"${pl:,.2f}",
                        "P&L %": f"{float(p.unrealized_plpc) * 100:.2f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions yet.")

        card = render_in_card("Recent Trades")
        with card:
            trades = get_recent_trades()
            if not trades.empty:
                if "timestamp" in trades.columns:
                    trades["timestamp"] = pd.to_datetime(trades["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(trades.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("No recent trades logged.")

def render_settings():
    render_hero("Settings", "Customize appearance and review bot configuration.")

    card = render_in_card("Appearance")
    with card:
        st.write(f"Active theme: **{st.session_state.theme.title()}**")
        st.caption("Use the sidebar Appearance toggle to switch light or dark mode.")

    card = render_in_card("Auto-Bot Symbols")
    with card:
        symbols = get_configured_symbols()
        st.write(", ".join(f"`{s}`" for s in symbols))
        st.code(f"TRADING_SYMBOLS={','.join(symbols)}", language="bash")
        st.caption("Update `.env` and restart `python main.py` to change auto-traded symbols.")
        st.caption("Crypto examples: `BTC/USD`, `ETH/USD` (or `BTCUSD`).")


def render_scheduler():
    render_hero("Scheduler", "Automated run windows that pause the bot outside market hours.")

    config = load_scheduler_config(SCHEDULER_FILE)
    card = render_in_card("Status")
    with card:
        st.write(describe_active_window(config))
        display = resolve_bot_display_status()
        if display == "paused_by_schedule":
            st.warning("Bot flag is start, but scheduler is holding it paused.")
        elif config.get("enabled"):
            if is_trading_window_active(config):
                st.success("Currently inside an active trading window.")
            else:
                st.warning("Currently outside trading windows — bot will stay paused.")

    card = render_in_card("Configure Windows")
    with card:
        enabled = st.toggle("Enable scheduler", value=bool(config.get("enabled", False)))
        timezone_name = st.text_input(
            "Timezone",
            value=str(config.get("timezone", "America/New_York")),
            help="IANA timezone, e.g. America/New_York or America/Chicago",
        )
        windows = list(config.get("windows") or DEFAULT_SCHEDULER_CONFIG["windows"])
        day_options = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        edited: list[dict] = []
        for idx, window in enumerate(windows):
            st.markdown(f"**Window {idx + 1}**")
            selected_days = st.multiselect(
                f"Active days #{idx + 1}",
                options=day_options,
                default=[d for d in window.get("days", day_options[:5]) if d in day_options],
                key=f"sched_days_{idx}",
            )
            c1, c2, c3 = st.columns([1, 1, 0.6])
            start_time = c1.text_input(
                f"Start #{idx + 1}",
                value=str(window.get("start", "09:30")),
                key=f"sched_start_{idx}",
            )
            end_time = c2.text_input(
                f"End #{idx + 1}",
                value=str(window.get("end", "16:00")),
                key=f"sched_end_{idx}",
                help="Overnight windows supported (e.g. 22:00 -> 06:00)",
            )
            remove = c3.checkbox("Remove", key=f"sched_remove_{idx}")
            if not remove:
                edited.append(
                    {
                        "days": selected_days or ["mon", "tue", "wed", "thu", "fri"],
                        "start": start_time.strip() or "09:30",
                        "end": end_time.strip() or "16:00",
                    }
                )

        if st.button("Add window"):
            edited.append({"days": ["mon", "tue", "wed", "thu", "fri"], "start": "09:30", "end": "16:00"})
            st.session_state["pending_scheduler_windows"] = edited
            st.rerun()

        pending = st.session_state.pop("pending_scheduler_windows", None)
        if pending is not None:
            edited = pending

        if st.button("Save schedule", type="primary"):
            invalid = [
                w for w in edited
                if not validate_hhmm(w["start"]) or not validate_hhmm(w["end"])
            ]
            if invalid:
                st.error("Each window needs valid HH:MM start/end times.")
            elif not edited:
                st.error("Keep at least one trading window, or disable the scheduler.")
            else:
                payload = {
                    "enabled": enabled,
                    "timezone": timezone_name.strip() or "America/New_York",
                    "windows": edited,
                }
                if save_scheduler_config(SCHEDULER_FILE, payload):
                    st.success("Scheduler saved. The bot watcher will apply it within a few seconds.")
                    st.toast("Scheduler updated", icon="✅")
                else:
                    st.error("Failed to save scheduler_config.json")

        st.caption(f"Config file: `{SCHEDULER_FILE}`")


def main():
    st.set_page_config(
        page_title="AI AutoBot Trader Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_theme()
    init_popup_state()
    apply_theme()
    api = init_alpaca_api()
    if not api:
        st.stop()
    show_startup_popup()
    menu = render_sidebar()
    apply_theme()
    if menu == MENU_DASHBOARD:
        render_dashboard()
    elif menu == MENU_SCHEDULER:
        render_scheduler()
    elif menu == MENU_SETTINGS:
        render_settings()


if __name__ == "__main__":
    main()
