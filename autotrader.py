"""
Automated Trading Bot with Dynamic Strategy Selection
- Loads API keys from .env
- Defines all user strategies as modular functions
- Selects and runs the best strategy based on market conditions
- Executes trades via Alpaca API
"""
import os
import time
import requests
from dotenv import load_dotenv

# Load API keys securely
load_dotenv()
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL')

HEADERS = {
    'APCA-API-KEY-ID': APCA_API_KEY_ID,
    'APCA-API-SECRET-KEY': APCA_API_SECRET_KEY
}

# --- Strategy Functions ---
import pandas as pd
from twitter_sentiment import get_twitter_sentiment
from reddit_sentiment import get_reddit_sentiment

def sector_rotation_strategy(factor_df, num_sectors=2):
    """Select top sectors by composite factor score and return buy signals."""
    factor_df['composite'] = factor_df[['growth', 'value_factor', 'quality', 'momentum', 'low_volatility']].sum(axis=1)
    sector_scores = factor_df.groupby('sector')['composite'].mean().sort_values(ascending=False)
    selected_sectors = sector_scores.head(num_sectors).index.tolist()
    selected = factor_df[factor_df['sector'].isin(selected_sectors)]
    weights = 1 / len(selected) if len(selected) > 0 else 0
    signals = [{'symbol': row['symbol'], 'weight': weights} for _, row in selected.iterrows()]
    return signals

def sentiment_strategy(symbol):
    """Combine Twitter and Reddit sentiment for a symbol."""
    twitter = get_twitter_sentiment(symbol,
        os.getenv('TWITTER_API_KEY'),
        os.getenv('TWITTER_API_KEY_SECRET'),
        os.getenv('TWITTER_ACCESS_TOKEN'),
        os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
    reddit = get_reddit_sentiment(symbol,
        os.getenv('REDDIT_CLIENT_ID'),
        os.getenv('REDDIT_CLIENT_SECRET'),
        os.getenv('REDDIT_USER_AGENT'))
    total_mentions = twitter['mentions'] + reddit['mentions']
    bullish = twitter['bullish'] + reddit['bullish']
    bearish = twitter['bearish'] + reddit['bearish']
    sentiment_score = bullish - bearish
    if total_mentions > 50 and sentiment_score > 10:
        return 'buy'
    elif total_mentions > 50 and sentiment_score < -10:
        return 'sell'
    return None
def breakout_strategy(market_data):
    """Buy when price breaks key resistance with volume spike"""
    # Placeholder logic
    if market_data['price'] > market_data['resistance'] and market_data['volume'] > market_data['avg_volume'] * 2:
        return 'buy'
    return None

def momentum_strategy(market_data):
    """Ride stocks with strong upward trends and positive momentum indicators"""
    if market_data['momentum'] > 0.8:
        return 'buy'
    return None

def mean_reversion_strategy(market_data):
    """Buy undervalued stocks below moving average; sell when they revert"""
    if market_data['price'] < market_data['moving_avg'] * 0.97:
        return 'buy'
    elif market_data['price'] > market_data['moving_avg'] * 1.03:
        return 'sell'
    return None

def swing_trading_strategy(market_data):
    """Enter trades at support/resistance levels; hold for several days to weeks"""
    if market_data['price'] <= market_data['support']:
        return 'buy'
    elif market_data['price'] >= market_data['resistance']:
        return 'sell'
    return None

def news_based_strategy(market_data):
    """Trade based on earnings releases, product launches, or macroeconomic news"""
    if market_data.get('news_event') == 'positive':
        return 'buy'
    elif market_data.get('news_event') == 'negative':
        return 'sell'
    return None

def volume_spike_strategy(market_data):
    """Enter trades when unusual volume suggests institutional activity"""
    if market_data['volume'] > market_data['avg_volume'] * 3:
        return 'buy'
    return None

def gap_and_go_strategy(market_data):
    """Trade stocks that gap up at open and show continued strength"""
    if market_data['gap_up'] and market_data['momentum'] > 0.7:
        return 'buy'
    return None

def sentiment_technical_combo_strategy(market_data):
    """Trade only when both sentiment and technicals align"""
    if market_data['rsi'] < 30 and market_data['sentiment'] == 'bullish':
        return 'buy'
    return None

# --- Strategy Selector ---
def select_strategy(market_data, factor_df=None):
    """Select the best strategy based on market conditions and user's custom logic."""
    # Try sector rotation first if factor data is available
    if factor_df is not None:
        signals = sector_rotation_strategy(factor_df)
        if signals:
            return 'sector_rotation_strategy', signals
    # Try sentiment strategy
    sentiment_action = sentiment_strategy(market_data['symbol'])
    if sentiment_action:
        return 'sentiment_strategy', [{'symbol': market_data['symbol'], 'weight': 1, 'action': sentiment_action}]
    # Fallback to built-in strategies
    strategies = [
        breakout_strategy,
        momentum_strategy,
        mean_reversion_strategy,
        swing_trading_strategy,
        news_based_strategy,
        volume_spike_strategy,
        gap_and_go_strategy,
        sentiment_technical_combo_strategy
    ]
    for strategy in strategies:
        action = strategy(market_data)
        if action:
            return strategy.__name__, [{'symbol': market_data['symbol'], 'weight': 1, 'action': action}]
    return None, None

# --- Trade Execution ---
def execute_trade(symbol, action, qty=1):
    """Send order to Alpaca"""
    endpoint = f"{APCA_API_BASE_URL}/v2/orders"
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": action,
        "type": "market",
        "time_in_force": "gtc"
    }
    response = requests.post(endpoint, json=order, headers=HEADERS)
    if response.status_code == 200:
        print(f"Order executed: {action} {qty} {symbol}")
    else:
        print(f"Order failed: {response.text}")

# --- Main Trading Loop ---
def get_market_data(symbol):
    """Fetch market data for symbol (placeholder)"""
    # Replace with real data fetching logic
    return {
        'price': 100,
        'resistance': 105,
        'support': 95,
        'volume': 50000,
        'avg_volume': 20000,
        'momentum': 0.85,
        'moving_avg': 102,
        'news_event': None,
        'gap_up': True,
        'rsi': 28,
        'sentiment': 'bullish'
    }

def main():
    symbol = 'AAPL'  # Example symbol
    # Example factor data for sector rotation
    factor_df = pd.DataFrame([
        {'symbol': 'AAPL', 'sector': 'Tech', 'growth': 0.8, 'value_factor': 0.7, 'quality': 0.9, 'momentum': 0.85, 'low_volatility': 0.6},
        {'symbol': 'MSFT', 'sector': 'Tech', 'growth': 0.7, 'value_factor': 0.8, 'quality': 0.85, 'momentum': 0.8, 'low_volatility': 0.7},
        {'symbol': 'JPM', 'sector': 'Finance', 'growth': 0.6, 'value_factor': 0.9, 'quality': 0.8, 'momentum': 0.7, 'low_volatility': 0.8},
        {'symbol': 'GS', 'sector': 'Finance', 'growth': 0.65, 'value_factor': 0.85, 'quality': 0.75, 'momentum': 0.65, 'low_volatility': 0.75}
    ])
    while True:
        market_data = get_market_data(symbol)
        strategy, actions = select_strategy(market_data, factor_df)
        if strategy and actions:
            print(f"Selected strategy: {strategy}, Actions: {actions}")
            for act in actions:
                execute_trade(act['symbol'], act.get('action', 'buy'), qty=int(100 * act['weight']))
        else:
            print("No actionable strategy found.")
        time.sleep(60)  # Run every minute

if __name__ == "__main__":
    main()
