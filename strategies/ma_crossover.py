"""
Moving Average Crossover strategy.

Generates a buy signal (1) when the fast SMA crosses above the slow SMA
(golden cross), and a sell signal (0) when it crosses below (death cross).

Default parameters (fast=20, slow=50) work well on daily AAPL/SPY data
and produce a clean narrative for classroom demos.

Look-ahead bias prevention: the raw crossover signal is shifted by 1 bar,
so we only act on a signal once the bar that generated it has fully closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    def __init__(self, fast_window: int = 20, slow_window: int = 50) -> None:
        super().__init__(name=f"MA_Crossover({fast_window},{slow_window})")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add SMA columns and a look-ahead-bias-free signal column.

        Columns added:
            sma_fast  : fast simple moving average
            sma_slow  : slow simple moving average
            signal    : 1 = long, 0 = flat (shifted by 1 bar)
        """
        df = data.copy()
        df["sma_fast"] = df["close"].rolling(window=self.fast_window).mean()
        df["sma_slow"] = df["close"].rolling(window=self.slow_window).mean()

        # Raw signal: 1 when fast > slow, else 0
        raw_signal = np.where(df["sma_fast"] > df["sma_slow"], 1, 0)

        # Shift by 1 bar to prevent look-ahead bias:
        # We can only act on today's close after the market closes,
        # so the signal takes effect at the NEXT bar's open (approximated as next close).
        df["signal"] = pd.Series(raw_signal, index=df.index).shift(1).fillna(0).astype(int)

        return df

    def __repr__(self) -> str:
        return f"MovingAverageCrossover(fast={self.fast_window}, slow={self.slow_window})"
