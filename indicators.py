import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Simple Moving Average
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    # RSI
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    # MACD
    macd = ta.trend.macd(df['close'])
    macd_signal = ta.trend.macd_signal(df['close'])
    macd_hist = macd - macd_signal
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    data = yf.download('AAPL', period='1mo', interval='1d')
    data = data.rename(columns={col: col.lower() for col in data.columns})
    data = add_indicators(data)
    print(data.tail())
