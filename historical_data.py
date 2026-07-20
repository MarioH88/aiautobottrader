import os
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

_TIMEFRAMES = {
    '1Min': TimeFrame.Minute,
    '5Min': TimeFrame(5, TimeFrame.Minute),
    '1D': TimeFrame.Day,
}


def fetch_alpaca_history(symbol, timeframe='1Min', limit=1000):
    api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    tf = _TIMEFRAMES.get(timeframe, TimeFrame.Minute)
    bars = api.get_bars(symbol, tf, limit=limit)
    df = bars.df.copy()
    if df.empty:
        return df
    if 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()
    else:
        df['symbol'] = symbol
    return df.reset_index()


def fetch_yahoo_history(symbol, interval='1m', period='5d'):
    data = yf.download(symbol, interval=interval, period=period, progress=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns={col: col.lower() for col in data.columns})
    return data.reset_index()


if __name__ == "__main__":
    print("Alpaca 1min bars:")
    print(fetch_alpaca_history('AAPL', '1Min', 10))
    print("Yahoo daily bars:")
    print(fetch_yahoo_history('AAPL', '1d', '1mo').tail())
