"""
Data fetching and local caching for backtesting.

Uses yfinance to download OHLCV data, then caches it as CSV so the demo
works offline during the lecture if network is unavailable.
"""

from __future__ import annotations

import os
import pandas as pd
import yfinance as yf


def fetch_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and normalize column names."""
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    # Flatten MultiIndex columns if present (yfinance >=0.2 returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


def load_or_fetch(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: str = ".cache",
) -> pd.DataFrame:
    """
    Load data from local CSV cache if available, otherwise fetch from Yahoo Finance.

    The cache file is named like: .cache/AAPL_2020-01-01_2024-01-01_1d.csv
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}_{interval}.csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col="date", parse_dates=True)
        return df

    df = fetch_data(ticker, start, end, interval)
    df.to_csv(cache_file)
    return df
