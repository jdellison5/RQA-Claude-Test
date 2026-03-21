"""
Shared pytest fixtures.

spy_data: real SPY daily OHLCV via yfinance (session-scoped, cached to .cache/).
Synthetic fixtures remain for unit tests that need controlled/known data shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data import load_or_fetch
from backtester.portfolio import Portfolio
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion


def _make_price_series(
    n: int,
    start_price: float,
    drift: float,
    volatility: float,
    seed: int = 42,
) -> pd.Series:
    np.random.seed(seed)
    returns = np.random.normal(drift, volatility, size=n)
    prices = start_price * np.exp(np.cumsum(returns))
    dates = pd.date_range(start="2020-01-02", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def _make_ohlcv(close_series: pd.Series) -> pd.DataFrame:
    """Wrap a close price series in a simple OHLCV DataFrame."""
    df = pd.DataFrame(index=close_series.index)
    df["close"] = close_series.values
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df["close"] * 1.005
    df["low"] = df["close"] * 0.995
    df["volume"] = 1_000_000
    df.index.name = "date"
    return df


@pytest.fixture(scope="session")
def spy_data() -> pd.DataFrame:
    """
    Real SPY daily OHLCV from yfinance, 2020-01-01 to 2024-01-01, cached locally.

    Falls back to synthetic trending data if the network is unavailable (e.g. in
    sandboxed environments). CI runners have full network access and will always
    use real data.
    """
    try:
        return load_or_fetch("SPY", "2020-01-01", "2024-01-01")
    except Exception:
        import warnings
        warnings.warn(
            "yfinance unavailable — spy_data fixture falling back to synthetic data. "
            "Real SPY data will be used in CI.",
            stacklevel=2,
        )
        close = _make_price_series(n=1005, start_price=320.0, drift=0.0004, volatility=0.012, seed=7)
        return _make_ohlcv(close)


@pytest.fixture
def synthetic_trending_data() -> pd.DataFrame:
    """200 bars of upward-trending price with noise. Deterministic via seed=42."""
    close = _make_price_series(n=200, start_price=100.0, drift=0.001, volatility=0.01, seed=42)
    return _make_ohlcv(close)


@pytest.fixture
def synthetic_oscillating_data() -> pd.DataFrame:
    """200 bars of mean-reverting price ideal for RSI testing."""
    np.random.seed(99)
    t = np.linspace(0, 4 * np.pi, 200)
    prices = 100 + 15 * np.sin(t) + np.random.normal(0, 1, 200)
    dates = pd.date_range(start="2020-01-02", periods=200, freq="B")
    close = pd.Series(prices, index=dates)
    return _make_ohlcv(close)


@pytest.fixture
def flat_equity_curve() -> pd.Series:
    dates = pd.date_range(start="2020-01-02", periods=100, freq="B")
    return pd.Series(10_000.0, index=dates)


@pytest.fixture
def rising_equity_curve() -> pd.Series:
    """Equity that doubles over 252 trading days (100% CAGR)."""
    dates = pd.date_range(start="2020-01-02", periods=252, freq="B")
    values = np.linspace(10_000, 20_000, 252)
    return pd.Series(values, index=dates)


@pytest.fixture
def drawdown_equity_curve() -> pd.Series:
    """Equity that peaks at 12000, drops to 9000, then recovers."""
    dates = pd.date_range(start="2020-01-02", periods=30, freq="B")
    values = [10000, 11000, 12000, 11500, 11000, 10500, 10000, 9500, 9000,
              9200, 9400, 9600, 9800, 10000, 10200, 10500, 11000, 11500,
              12000, 12200, 12500, 12800, 13000, 13200, 13500, 13800, 14000,
              14200, 14500, 14800]
    return pd.Series(values, index=dates, dtype=float)


@pytest.fixture
def sample_portfolio() -> Portfolio:
    return Portfolio(initial_capital=10_000.0)


@pytest.fixture
def ma_strategy() -> MovingAverageCrossover:
    return MovingAverageCrossover(fast_window=10, slow_window=20)


@pytest.fixture
def rsi_strategy() -> RSIMeanReversion:
    return RSIMeanReversion(period=2, oversold=25, overbought=75)
