"""
SMA Crossover Backtest — backtesting.py
"""

import pandas as pd
import yfinance as yf
from backtesting import Strategy
from backtesting.lib import FractionalBacktest
from backtesting.lib import crossover


# ── CONFIG ────────────────────────────────────────────────────────────────────

SYMBOL      = "AAPL"
START_DATE  = "2023-01-01"
END_DATE    = "2026-01-01"

SHORT_WIN   = 20        # Short SMA window (days)
LONG_WIN    = 50        # Long SMA window (days)

CASH        = 10_000    # Starting capital ($)
COMMISSION  = 0.002     # Commission per trade (0.2%)


# ── DATA ─────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = None
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ── STRATEGY ─────────────────────────────────────────────────────────────────

class SMACrossover(Strategy):
    short_win = SHORT_WIN
    long_win  = LONG_WIN

    def init(self):
        c = self.data.Close
        self.sma_short = self.I(lambda x: pd.Series(x).rolling(self.short_win).mean(), c)
        self.sma_long  = self.I(lambda x: pd.Series(x).rolling(self.long_win).mean(),  c)

    def next(self):
        if crossover(self.sma_short, self.sma_long):   # Golden Cross → BUY
            self.buy()
        elif crossover(self.sma_long, self.sma_short): # Death Cross  → EXIT
            self.position.close()


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_data(SYMBOL, START_DATE, END_DATE)

    bt    = FractionalBacktest(df, SMACrossover, cash=CASH, commission=COMMISSION, finalize_trades=True)
    stats = bt.run()

    print(stats)
    bt.plot()