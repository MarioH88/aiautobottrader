import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from historical_data import fetch_alpaca_history, fetch_yahoo_history
from indicators import add_indicators
from ml_predict import load_sklearn_model, predict_action
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _decide_action(row: pd.Series, features: pd.DataFrame, model: Optional[Any], position: int, rsi_buy: float, rsi_sell: float) -> int:
    """Decide trading action based on ML model or rule-based logic."""
    if model is not None:
        return predict_action(model, features)
    rsi = row.get('rsi_14', 50)
    if rsi < rsi_buy and position == 0:
        return 1
    elif rsi > rsi_sell and position > 0:
        return -1
    return 0

def _execute_trade(action: int, row: pd.Series, position: int, entry_price: float, trades: List[Dict], slippage: float, commission: float) -> (int, float):
    """Update position, entry price, and trades list based on action."""
    if action == 1 and position == 0:
        position = 1
        entry_price = row['close'] * (1 + slippage) + commission
        trades.append({
            'timestamp': row.name,
            'action': 'buy',
            'price': row['close'],
            'slippage': slippage,
            'commission': commission
        })
    elif action == -1 and position > 0:
        position = 0
        sell_price = row['close'] * (1 - slippage) - commission
        pnl = sell_price - entry_price
        trades.append({
            'timestamp': row.name,
            'action': 'sell',
            'price': row['close'],
            'pnl': pnl,
            'slippage': slippage,
            'commission': commission
        })
    return position, entry_price

def backtest(
    symbol: str,
    model_path: Optional[str] = None,
    use_yahoo: bool = False,
    limit: int = 1000,
    rsi_buy: float = 30,
    rsi_sell: float = 70,
    slippage: float = 0.0005,
    commission: float = 0.0,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Backtest a trading strategy on historical data for a single symbol.
    Args:
        symbol (str): Ticker symbol to backtest.
        model_path (str, optional): Path to ML model. If None, use rule-based logic.
        use_yahoo (bool): If True, use Yahoo Finance data. Else, use Alpaca.
        limit (int): Number of bars to fetch (Alpaca only).
        rsi_buy (float): RSI threshold for buy signal (rule-based).
        rsi_sell (float): RSI threshold for sell signal (rule-based).
    Returns:
        pd.DataFrame: Trade log with buy/sell actions and PnL.
    """
    # Fetch historical data
    try:
        if use_yahoo:
            df = fetch_yahoo_history(symbol, interval='1m', period='5d')
            df = df.rename(columns={col: col.lower() for col in df.columns})
        else:
            df = fetch_alpaca_history(symbol, '1Min', limit)
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        logging.error(f"No historical data returned for symbol: {symbol}")
        return pd.DataFrame()

    try:
        df = add_indicators(df)
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        return pd.DataFrame()
    model = load_sklearn_model(model_path) if model_path else None
    position = 0
    entry_price = 0.0
    trades: List[Dict] = []

    iterator = range(1, len(df))
    if show_progress and len(df) > 100:
        iterator = tqdm(iterator, desc=f"Backtesting {symbol}")

    for i in iterator:
        row = df.iloc[i]
        # Defensive: skip if any required indicator is missing or NaN
        try:
            features = row[['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'bb_high', 'bb_low']].to_frame().T
            if features.isnull().any().any():
                continue
        except KeyError:
            continue
        action = _decide_action(row, features, model, position, rsi_buy, rsi_sell)
        position, entry_price = _execute_trade(action, row, position, entry_price, trades, slippage, commission)
        # No-op if no trade, so position stays the same
    return pd.DataFrame(trades)

from typing import Tuple

def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics for a trade log DataFrame."""
    if trades.empty:
        return {"total_pnl": 0.0, "num_trades": 0, "win_rate": 0.0, "max_drawdown": 0.0}
    trades = trades.copy()
    trades['pnl'] = trades.get('pnl', 0.0)
    total_pnl = trades['pnl'].sum() if 'pnl' in trades else 0.0
    num_trades = len(trades) // 2  # buy/sell pairs
    wins = trades[trades['pnl'] > 0].shape[0] if 'pnl' in trades else 0
    win_rate = wins / num_trades if num_trades > 0 else 0.0
    # Max drawdown calculation
    equity_curve = trades['pnl'].cumsum() if 'pnl' in trades else pd.Series([0])
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown
    }

def save_trades(trades: pd.DataFrame, filename: str = "trades.csv") -> None:
    """Save trades DataFrame to CSV."""
    if not trades.empty:
        trades.to_csv(filename, index=False)
        logging.info(f"Trades saved to {filename}")

if __name__ == "__main__":
    # Example usage: backtest AAPL with rule-based logic on Yahoo data
    trades = backtest(
        'AAPL',
        model_path=None,
        use_yahoo=True,
        limit=500,
        slippage=0.0005,
        commission=0.01,
        show_progress=True
    )
    logging.info("Last 5 trades:\n%s", trades.tail())
    # Performance summary
    summary = summarize_trades(trades)
    logging.info("Performance Summary: %s", summary)
    # Save to CSV
    save_trades(trades, filename="trades_AAPL.csv")
