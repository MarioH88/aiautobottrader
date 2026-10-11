"""Historical backtest with ATR trailing stops and performance metrics."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from bot.features import FEATURE_COLUMNS, build_features
from historical_data import fetch_alpaca_history, fetch_yahoo_history
from ml_predict import load_sklearn_model, predict_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _decide_action(
    row: pd.Series,
    features: pd.DataFrame,
    model: Optional[Any],
    position: int,
    rsi_buy: float,
    rsi_sell: float,
) -> int:
    """Decide trading action based on ML model or rule-based logic."""
    if model is not None:
        pred = predict_action(model, features)
        if pred == 1 and position == 0:
            return 1
        if pred == -1 and position > 0:
            return -1
        return 0
    rsi = row.get("rsi", row.get("rsi_14", 50))
    macd = row.get("macd", 0.0)
    if rsi < rsi_buy and macd >= 0 and position == 0:
        return 1
    if rsi > rsi_sell and macd <= 0 and position > 0:
        return -1
    return 0


def _execute_trade(
    action: int,
    row: pd.Series,
    position: int,
    entry_price: float,
    stop_price: float,
    trades: List[Dict],
    slippage: float,
    commission: float,
    atr_multiplier: float,
) -> Tuple[int, float, float]:
    """Update position, entry/stop, and trades list based on action."""
    close = float(row["close"])
    atr = float(row.get("atr", 0.0) or 0.0)
    if action == 1 and position == 0:
        position = 1
        entry_price = close * (1 + slippage) + commission
        static_stop = entry_price * 0.98
        atr_stop = entry_price - (atr_multiplier * atr) if atr > 0 else static_stop
        stop_price = min(static_stop, atr_stop)
        trades.append(
            {
                "timestamp": row.name if hasattr(row, "name") else row.get("timestamp"),
                "action": "buy",
                "price": close,
                "slippage": slippage,
                "commission": commission,
                "stop": stop_price,
            }
        )
    elif action == -1 and position > 0:
        position = 0
        sell_price = close * (1 - slippage) - commission
        pnl = sell_price - entry_price
        trades.append(
            {
                "timestamp": row.name if hasattr(row, "name") else row.get("timestamp"),
                "action": "sell",
                "price": close,
                "pnl": pnl,
                "slippage": slippage,
                "commission": commission,
                "reason": "signal",
            }
        )
        entry_price = 0.0
        stop_price = 0.0
    return position, entry_price, stop_price


def backtest(
    symbol: str,
    model_path: Optional[str] = None,
    use_yahoo: bool = False,
    limit: int = 1000,
    rsi_buy: float = 30,
    rsi_sell: float = 70,
    slippage: float = 0.0005,
    commission: float = 0.0,
    atr_multiplier: float = 2.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Backtest a trading strategy with ATR trailing stops."""
    try:
        if use_yahoo:
            df = fetch_yahoo_history(symbol, interval="5m", period="59d")
            if df.empty:
                logging.warning("Yahoo data empty for %s, falling back to Alpaca", symbol)
                df = fetch_alpaca_history(symbol, "5Min", limit)
        else:
            df = fetch_alpaca_history(symbol, "5Min", limit)
    except Exception as exc:
        logging.error("Error fetching data for %s: %s", symbol, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        logging.error("No historical data returned for symbol: %s", symbol)
        return pd.DataFrame()

    try:
        # Same feature code the live bot and trainer use, so the backtest is apples-to-apples.
        df = build_features(df.sort_values("timestamp"), sentiment_score=0.0, macro_sentiment=0.0)
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    except Exception as exc:
        logging.error("Error adding indicators: %s", exc)
        return pd.DataFrame()

    model = load_sklearn_model(model_path) if model_path else None
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    trades: List[Dict] = []

    iterator = range(1, len(df))
    if show_progress and len(df) > 100:
        iterator = tqdm(iterator, desc=f"Backtesting {symbol}")

    for i in iterator:
        row = df.iloc[i]
        try:
            features = row[FEATURE_COLUMNS].to_frame().T.astype(float)
            if features.isnull().any().any():
                continue
        except KeyError:
            continue

        # ATR trailing stop check while in a position
        if position > 0:
            close = float(row["close"])
            atr = float(row.get("atr", 0.0) or 0.0)
            if close > entry_price and atr > 0:
                trailed = close - (atr_multiplier * atr)
                stop_price = max(stop_price, trailed)
            if close <= stop_price:
                position, entry_price, stop_price = _execute_trade(
                    -1,
                    row,
                    position,
                    entry_price,
                    stop_price,
                    trades,
                    slippage,
                    commission,
                    atr_multiplier,
                )
                trades[-1]["reason"] = "atr-trailing-stop"
                continue

        action = _decide_action(row, features, model, position, rsi_buy, rsi_sell)
        position, entry_price, stop_price = _execute_trade(
            action,
            row,
            position,
            entry_price,
            stop_price,
            trades,
            slippage,
            commission,
            atr_multiplier,
        )
    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics for a trade log DataFrame."""
    if trades.empty:
        return {
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "sharpe": 0.0,
        }

    sells = trades[trades["action"] == "sell"].copy() if "action" in trades.columns else trades.copy()
    if "pnl" not in sells.columns:
        sells["pnl"] = 0.0
    sells = sells.dropna(subset=["pnl"])
    pnls = sells["pnl"].astype(float)
    total_pnl = float(pnls.sum()) if not pnls.empty else 0.0
    num_trades = int(len(sells))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / num_trades) if num_trades else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    equity = pnls.cumsum()
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    sharpe = 0.0
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = float((pnls.mean() / pnls.std()) * (num_trades ** 0.5))

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
    }


def save_trades(trades: pd.DataFrame, filename: str = "trades.csv") -> None:
    """Save trades DataFrame to CSV."""
    if not trades.empty:
        trades.to_csv(filename, index=False)
        logging.info("Trades saved to %s", filename)


if __name__ == "__main__":
    for symbol in ("AAPL", "BTC/USD"):
        trades = backtest(
            symbol,
            model_path="model.pkl" if os.path.exists("model.pkl") else None,
            use_yahoo=False,
            limit=500,
            slippage=0.0005,
            commission=0.01,
            show_progress=True,
        )
        logging.info("%s last trades:\n%s", symbol, trades.tail())
        summary = summarize_trades(trades)
        logging.info("%s Performance Summary: %s", symbol, summary)
        safe_name = symbol.replace("/", "_")
        save_trades(trades, filename=f"trades_{safe_name}.csv")
