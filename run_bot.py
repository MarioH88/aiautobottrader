import time
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from indicators import add_indicators
from scalping_breakout_strategy import detect_breakout, get_take_profit_stop_loss
from trade_log import log_trade

SYMBOL = "AAPL"  # Change to your preferred symbol
QTY = 1

def fetch_latest_data(api, symbol, limit=50):
    barset = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit)
    df = barset.df
    # If 'symbol' column exists, filter by symbol; else, use as is
    if 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()
    df = df.reset_index()
    return df

def place_order(api, symbol, qty, side, reason=None):
    order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
    print(f"Order placed: {side} {qty} {symbol}")
    log_trade(datetime.now(), symbol, side, 0, qty, reason)  # price 0 for now
    return order

def main():
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    while True:
        with open("bot_flag.txt") as f:
            flag = f.read().strip()
        if flag != "start":
            print("Bot is stopped.")
            time.sleep(10)
            continue

        print("Bot is running...")
        try:
            df = fetch_latest_data(api, SYMBOL, limit=50)
            df = add_indicators(df)
            # --- Log indicator values for debugging ---
            print("Latest indicators:")
            print(df.tail(3)[['close','sma_20','ema_20','rsi_14','macd','macd_signal','macd_hist','bb_high','bb_low']])

            # Get account info for cash and position
            account = api.get_account()
            cash = float(account.cash)
            positions = {p.symbol: int(float(p.qty)) for p in api.list_positions()}
            shares_held = positions.get(SYMBOL, 0)

            # --- Aggressive strategy: buy if close > ema_20 or rsi_14 > 45, sell if close < sma_20 or rsi_14 < 55 ---
            last_row = df.iloc[-1]
            buy_signal = False
            sell_signal = False
            if last_row['close'] > last_row['ema_20'] or last_row['rsi_14'] > 45:
                buy_signal = True
                print(f"Buy signal: close={last_row['close']} > ema_20={last_row['ema_20']} or rsi_14={last_row['rsi_14']} > 45")
            if last_row['close'] < last_row['sma_20'] or last_row['rsi_14'] < 55:
                sell_signal = True
                print(f"Sell signal: close={last_row['close']} < sma_20={last_row['sma_20']} or rsi_14={last_row['rsi_14']} < 55")

            # Buy as long as you have enough cash for at least 1 share
            if buy_signal:
                est_price = last_row['close']
                max_qty = int(cash // est_price)
                if max_qty >= 1:
                    print(f"Placing BUY order for {max_qty} shares...")
                    place_order(api, SYMBOL, max_qty, 'buy', reason='aggressive_buy')
                else:
                    print("Not enough cash to buy.")

            # Sell as long as you have shares
            if sell_signal and shares_held > 0:
                print(f"Placing SELL order for {shares_held} shares...")
                place_order(api, SYMBOL, shares_held, 'sell', reason='aggressive_sell')
            elif sell_signal:
                print("No shares to sell.")
        except Exception as e:
            print(f"Error in trading loop: {e}")
        time.sleep(60)  # Run every minute

if __name__ == "__main__":
    main()