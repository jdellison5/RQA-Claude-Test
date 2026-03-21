"""
Data fetching and local caching for backtesting.

Uses yfinance to download OHLCV data, then caches it as CSV so the demo
works offline during the lecture if network is unavailable.

Pass start=None and end=None to fetch the maximum available history for a
ticker (equivalent to yfinance's period="max").  Omitting end alone defaults
it to today's date so the backtest always runs to the most recent close.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and normalise column names.

    start / end : YYYY-MM-DD strings, or None.
        - Both None  → fetch the full available history (period="max").
        - start only → fetch from start through today.
        - end only   → fetch from the earliest available date through end.
    """
    if start is None and end is None:
        df = yf.download(ticker, period="max", interval=interval, auto_adjust=True, progress=False)
        desc = f"{ticker} (full history)"
    else:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        desc = f"{ticker} {start} → {end}"

    if df.empty:
        raise ValueError(f"No data returned for {desc}")

    # Flatten MultiIndex columns if present (yfinance >=0.2 returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


def load_or_fetch(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: str = ".cache",
) -> pd.DataFrame:
    """
    Load data from local CSV cache if available, otherwise fetch from Yahoo Finance.

    Cache file naming:
        start=None, end=None  → .cache/AAPL_max_2026-03-21_1d.csv  (keyed to today)
        start provided        → .cache/AAPL_2020-01-01_2026-03-21_1d.csv
    Keying the "max" case to today's date ensures the cache refreshes daily
    so the backtest always reflects the latest available close.
    """
    today = date.today().isoformat()
    cache_start = start if start is not None else "max"
    cache_end = end if end is not None else today

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{cache_start}_{cache_end}_{interval}.csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col="date", parse_dates=True)
        return df

    df = fetch_data(ticker, start, end, interval)
    df.to_csv(cache_file)
    return df
