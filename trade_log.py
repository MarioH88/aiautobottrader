import pandas as pd
import os
from datetime import datetime

LOG_FILE = 'trade_log.csv'

# Log a trade to CSV

def log_trade(timestamp, symbol, action, price, qty, reason=None):
    log_exists = os.path.isfile(LOG_FILE)
    row = {
        'timestamp': timestamp,
        'symbol': symbol,
        'action': action,
        'price': price,
        'qty': qty,
        'reason': reason or ''
    }
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode='a', header=not log_exists, index=False)

if __name__ == "__main__":
    # Example usage
    log_trade(datetime.now(), 'AAPL', 'buy', 200.0, 1, 'test buy')
    print(pd.read_csv(LOG_FILE).tail())
