"""
RSI Mean Reversion strategy.

Buys when RSI crosses up through the oversold threshold (default 30),
sells when RSI crosses down through the overbought threshold (default 70).

Uses the `ta` library (pip install ta) which is pure Python and compatible
with Python 3.11. Do NOT use pandas-ta — it does not support Python 3.11.

Look-ahead bias prevention: signal is shifted by 1 bar before returning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from strategies.base import BaseStrategy


class RSIMeanReversion(BaseStrategy):
    def __init__(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ) -> None:
        super().__init__(name=f"RSI_MeanReversion({period},{oversold},{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI column and a look-ahead-bias-free signal column.

        Columns added:
            rsi    : RSI oscillator (0-100)
            signal : 1 = long, 0 = flat (shifted by 1 bar)

        Signal logic:
            - Enter long when RSI crosses UP through oversold level
              (RSI was below oversold yesterday, is above today)
            - Exit long when RSI crosses DOWN through overbought level
              (RSI was above overbought yesterday, is below today)
        """
        df = data.copy()

        rsi_indicator = RSIIndicator(close=df["close"], window=self.period)
        df["rsi"] = rsi_indicator.rsi()

        # Detect crossings
        rsi_prev = df["rsi"].shift(1)

        buy_cross = (rsi_prev < self.oversold) & (df["rsi"] >= self.oversold)
        sell_cross = (rsi_prev > self.overbought) & (df["rsi"] <= self.overbought)

        # Build signal using forward-fill: enter on buy cross, exit on sell cross
        raw_signal = pd.Series(np.nan, index=df.index)
        raw_signal[buy_cross] = 1
        raw_signal[sell_cross] = 0
        raw_signal = raw_signal.ffill().fillna(0).astype(int)

        # Shift by 1 bar to prevent look-ahead bias
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)

        return df

    def __repr__(self) -> str:
        return (
            f"RSIMeanReversion(period={self.period}, "
            f"oversold={self.oversold}, overbought={self.overbought})"
        )
