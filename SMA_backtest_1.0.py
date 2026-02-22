"""
Moving Average Crossover Strategy, yfinance
=======================================================
Quant Finance Project
-----------------------
Strategy: Dual SMA Crossover (Golden Cross / Death Cross)
- BUY  when short-term SMA crosses ABOVE long-term SMA
- SELL when short-term SMA crosses BELOW long-term SMA
"""

import pandas as pd
import numpy as np
import yfinance as yf

# STRATEGY PARAMETERS
SYMBOL     = "AAPL"          # Ticker to trade
SHORT_WIN  = 20              # Short SMA window (days)
LONG_WIN   = 50              # Long SMA window (days)
START_DATE = "2020-01-01"
END_DATE   = "2024-01-01"


# FETCH DATA FROM YFINANCE
def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Yahoo Finance.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    No API key required.
    auto_adjust=True applies splits/dividends automatically.
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance may return MultiIndex columns — flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = None

    return df[["Open", "High", "Low", "Close", "Volume"]]


#COMPUTE SIGNALS
def add_signals(df: pd.DataFrame, short_win: int, long_win: int) -> pd.DataFrame:
    """
    Add SMA columns and a crossover signal.
    Signal:  1 = Long (Golden Cross), -1 = Short/Flat (Death Cross)
    Position: lagged by 1 bar to avoid look-ahead bias.
    """
    df = df.copy()

    df[f"SMA_{short_win}"] = df["Close"].rolling(short_win).mean()
    df[f"SMA_{long_win}"]  = df["Close"].rolling(long_win).mean()

    # Raw signal — 1 when short MA above long MA
    df["Signal"] = np.where(
        df[f"SMA_{short_win}"] > df[f"SMA_{long_win}"], 1, -1
    )

    # Detect crossover events (change in signal)
    df["Crossover"] = df["Signal"].diff().fillna(0)

    # IMPORTANT: shift signal by 1 to execute on the NEXT bar open
    # This prevents look-ahead bias — a critical concern in quant backtesting
    df["Position"] = df["Signal"].shift(1)

    df.dropna(inplace=True)
    return df

# STEP 3: STRATEGY CLASS FOR backtesting.py
try:
    from backtesting import Strategy
    from backtesting.lib import crossover

    class SMACrossover(Strategy):
        """
        Drop-in Strategy class for backtesting.py
        Usage:
            from backtesting import Backtest
            bt = Backtest(df, SMACrossover, cash=10_000, commission=0.002)
            stats = bt.run()
            bt.plot()
        """
        short_window = SHORT_WIN
        long_window  = LONG_WIN

        def init(self):
            close = self.data.Close
            self.sma_short = self.I(lambda x: pd.Series(x).rolling(self.short_window).mean(), close)
            self.sma_long  = self.I(lambda x: pd.Series(x).rolling(self.long_window).mean(),  close)

        def next(self):
            if crossover(self.sma_short, self.sma_long):
                self.buy()   # Golden Cross → Go Long

            elif crossover(self.sma_long, self.sma_short):
                self.position.close()  # Death Cross → Exit

except ImportError:
    SMACrossover = None
    print("backtesting.py not installed — pip install backtesting")


# STEP 4: SIMPLE VECTORISED BACKTEST 
def vectorised_backtest(df: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    """
    Fast vectorised backtest — useful for parameter sweeps.

    Returns daily portfolio value. Strategy goes long when signal=1,
    stays in cash (or short) when signal=-1.

    Strategy Return = Position × Daily Return
    """
    df = df.copy()
    df["Daily_Return"]    = df["Close"].pct_change()
    df["Strategy_Return"] = df["Position"] * df["Daily_Return"]

    df["Equity_Curve"]    = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    df["Buy_Hold_Curve"]  = initial_capital * (1 + df["Daily_Return"]).cumprod()

    return df


def print_stats(df: pd.DataFrame):
    """Print key performance metrics — standard quant tearsheet basics."""
    strat  = df["Strategy_Return"].dropna()
    bh     = df["Daily_Return"].dropna()

    trading_days = 252

    def sharpe(returns):
        return (returns.mean() / returns.std()) * np.sqrt(trading_days)

    def max_drawdown(equity):
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        return drawdown.min()

    def cagr(equity, n_years):
        return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1

    n_years = len(df) / trading_days

    print("\n" + "="*45)
    print(f"  STRATEGY PERFORMANCE — {SYMBOL}")
    print("="*45)
    print(f"  Sharpe Ratio (Strategy):  {sharpe(strat):.2f}")
    print(f"  Sharpe Ratio (Buy&Hold):  {sharpe(bh):.2f}")
    print(f"  CAGR (Strategy):          {cagr(df['Equity_Curve'], n_years)*100:.1f}%")
    print(f"  CAGR (Buy & Hold):        {cagr(df['Buy_Hold_Curve'], n_years)*100:.1f}%")
    print(f"  Max Drawdown (Strategy):  {max_drawdown(df['Equity_Curve'])*100:.1f}%")
    print(f"  Max Drawdown (Buy&Hold):  {max_drawdown(df['Buy_Hold_Curve'])*100:.1f}%")
    print(f"  Total Trades:             {int((df['Crossover'].abs() > 0).sum())}")
    print("="*45 + "\n")


# MAIN
if __name__ == "__main__":
    print(f"Fetching {SYMBOL} data from Yahoo Finance...")
    df = fetch_data(SYMBOL, START_DATE, END_DATE)
    print(f"Fetched {len(df)} rows.\n")

    df = add_signals(df, SHORT_WIN, LONG_WIN)
    df = vectorised_backtest(df)

    print_stats(df)

    # Save for further analysis
    df.to_csv(f"{SYMBOL}_ma_backtest.csv")
    print(f"Results saved to {SYMBOL}_ma_backtest.csv")

    # Optional: run with backtesting.py
    if SMACrossover:
        from backtesting import Backtest
        raw_df = fetch_data(SYMBOL, START_DATE, END_DATE)
        bt = Backtest(raw_df, SMACrossover, cash=10_000, commission=0.002)
        stats = bt.run()
        print(stats)
        bt.plot()