"""
RSI Mean Reversion strategy.

Buys when RSI falls below the oversold threshold (default 25),
sells when RSI rises above the overbought threshold (default 75).

Defaults to RSI(2) — a short-period mean-reversion approach.

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
        period: int = 2,
        oversold: int = 25,
        overbought: int = 75,
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
            - Enter long when RSI drops below oversold level
            - Exit long when RSI rises above overbought level
            - Otherwise maintain current position (forward-filled)
        """
        df = data.copy()

        rsi_indicator = RSIIndicator(close=df["close"], window=self.period)
        df["rsi"] = rsi_indicator.rsi()

        # Level-based entries and exits
        buy_signal = df["rsi"] < self.oversold
        sell_signal = df["rsi"] > self.overbought

        # Build signal using forward-fill: enter on oversold, exit on overbought
        raw_signal = pd.Series(np.nan, index=df.index)
        raw_signal[buy_signal] = 1
        raw_signal[sell_signal] = 0
        raw_signal = raw_signal.ffill().fillna(0).astype(int)

        # Shift by 1 bar to prevent look-ahead bias
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)

        return df

    def __repr__(self) -> str:
        return (
            f"RSIMeanReversion(period={self.period}, "
            f"oversold={self.oversold}, overbought={self.overbought})"
        )
