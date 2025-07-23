import backtrader as bt
import yfinance as yf

class SmaRsiMacdStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.sma_period)
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.p.macd1, period_me2=self.p.macd2, period_signal=self.p.macdsig)

    def next(self):
        if not self.position:
            if self.rsi < 30 and self.datas[0].close[0] > self.sma[0] and self.macd.macd[0] > self.macd.signal[0]:
                self.buy()
        else:
            if self.rsi > 70 or self.datas[0].close[0] < self.sma[0] or self.macd.macd[0] < self.macd.signal[0]:
                self.sell()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaRsiMacdStrategy)
    # Download historical data from Yahoo Finance
    data = bt.feeds.PandasData(dataname=yf.download('AAPL', '2023-01-01', '2023-12-31', interval='1d'))
    cerebro.adddata(data)
    cerebro.broker.set_cash(10000)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()
