import os
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

def fetch_alpaca_history(symbol, timeframe='1Min', limit=1000):
    api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    tf = {'1Min': TimeFrame.Minute, '5Min': TimeFrame(5, TimeFrame.Unit.Minute), '1D': TimeFrame.Day}[timeframe]
    bars = api.get_bars(symbol, tf, limit=limit)
    df = bars.df[ bars.df['symbol'] == symbol ].copy()
    return df

def fetch_yahoo_history(symbol, interval='1m', period='5d'):
    data = yf.download(symbol, interval=interval, period=period)
    return data

if __name__ == "__main__":
    # Example usage
    print("Alpaca 1min bars:")
    print(fetch_alpaca_history('AAPL', '1Min', 10))
    print("Yahoo 1min bars:")
    print(fetch_yahoo_history('AAPL', '1m', '1d'))
