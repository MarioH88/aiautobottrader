import numpy as np
import pandas as pd

def detect_breakout(bars, resistance_lookback=20, volume_multiplier=2.0):
    """
    Detect breakout: price breaks above recent resistance and volume spike.
    Returns True if breakout detected, else False.
    """
    if len(bars) < resistance_lookback + 1:
        return False
    recent_high = bars['high'][-(resistance_lookback+1):-1].max()
    last_close = bars['close'].iloc[-1]
    last_vol = bars['volume'].iloc[-1]
    avg_vol = bars['volume'][-(resistance_lookback+1):-1].mean()
    breakout = last_close > recent_high and last_vol > avg_vol * volume_multiplier
    return breakout

def get_take_profit_stop_loss(entry_price, tp_pct=0.01, sl_pct=0.01):
    """
    Returns (take_profit_price, stop_loss_price)
    """
    take_profit = entry_price * (1 + tp_pct)
    stop_loss = entry_price * (1 - sl_pct)
    return take_profit, stop_loss
