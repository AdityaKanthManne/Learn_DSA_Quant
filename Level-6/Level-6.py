import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class RealTimeAnalytics:
    def __init__(self, window):
        self.window = window
        self.prices = collections.deque(maxlen=window)
        self.returns = collections.deque(maxlen=window)
        self.sum_prices = 0.0
        self.sum_returns = 0.0

    def update(self, price):
        # Manage price queue
        if len(self.prices) == self.window:
            self.sum_prices -= self.prices[0]
        self.prices.append(price)
        self.sum_prices += price

        # Compute return if possible
        if len(self.prices) > 1:
            ret = (self.prices[-1] / self.prices[-2]) - 1
            if len(self.returns) == self.window:
                self.sum_returns -= self.returns[0]
            self.returns.append(ret)
            self.sum_returns += ret

    def mean_price(self):
        return self.sum_prices / len(self.prices) if self.prices else np.nan

    def volatility(self):
        if len(self.returns) < 2:
            return np.nan
        return np.std(self.returns, ddof=1) * np.sqrt(252)  # annualized

    def sharpe(self, risk_free=0.02):
        if len(self.returns) < 2:
            return np.nan
        excess = np.array(self.returns) - (risk_free / 252)
        mean_ret = np.mean(excess)
        vol = np.std(excess, ddof=1)
        return np.sqrt(252) * mean_ret / vol if vol != 0 else np.nan


def process_stream(ticker="AAPL", window=20, period="1y"):
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        raise SystemExit("No data returned.")
    prices = data["Close"]

    stream = RealTimeAnalytics(window)
    ma, vol, sharpe = [], [], []

    for p in prices:
        stream.update(p)
        ma.append(stream.mean_price())
        vol.append(stream.volatility())
        sharpe.append(stream.sharpe())

    df = pd.DataFrame({
        "Price": prices.values,
        "MA": ma,
        "Volatility": vol,
        "Sharpe": sharpe
    }, index=prices.index)

    # Plot prices and indicators
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(df.index, df["Price"], label="Price", alpha=0.8)
    axs[0].plot(df.index, df["MA"], label=f"{window}-Day MA", linewidth=2)
    axs[0].set_title(f"{ticker} Price Stream + Moving Average")
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(df.index, df["Volatility"], color="orange", label="Rolling Volatility")
    axs[1].set_title("Rolling Volatility (Queue-Based)"); axs[1].legend(); axs[1].grid(True)

    axs[2].plot(df.index, df["Sharpe"], color="purple", label="Rolling Sharpe Ratio")
    axs[2].axhline(0, color="black", linewidth=0.8)
    axs[2].set_title("Rolling Sharpe Ratio"); axs[2].legend(); axs[2].grid(True)

    plt.tight_layout(); plt.show()
    return df


if __name__ == "__main__":
    df = process_stream("AAPL", window=30)
    print(df.tail())
