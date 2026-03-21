"""
252-Day High Breakout trend-following strategy with monthly rebalancing.

Entry : go long when the closing price breaches the prior 252-day high.
Exit  : exit when the closing price falls more than `trailing_stop` (default 5%)
        below the rolling 252-day high (the "highest high" trailing stop).

Rebalancing : signals are evaluated daily, but position changes only execute
              at the last trading day of each calendar month.  If a buy or sell
              condition triggers mid-month, the signal is logged (stored in the
              `raw_signal` column) and acted on at the next month-end close.

Look-ahead bias prevention: all indicators use close.shift(1) so today's close
is never used to compute today's signal.  The final signal is then shifted an
additional bar so the trade executes on the open of the following bar.

Columns added by generate_signals():
    high_252         : rolling 252-day high (lagged by 1 bar — no look-ahead)
    trail_stop_level : high_252 * (1 - trailing_stop)
    raw_signal       : the un-snapped daily signal (1 = long, 0 = flat)
    signal           : monthly-rebalanced, look-ahead-bias-free signal
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class BreakoutTrendFollowing(BaseStrategy):
    def __init__(
        self,
        breakout_period: int = 252,
        trailing_stop: float = 0.05,
    ) -> None:
        super().__init__(
            name=f"Breakout_TrendFollowing({breakout_period},{trailing_stop:.0%})"
        )
        self.breakout_period = breakout_period
        self.trailing_stop = trailing_stop

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator and signal columns, applying monthly rebalancing.

        Signal logic (daily):
            - Enter long when close > prior 252-day high (new breakout)
            - Exit when close < prior 252-day high * (1 - trailing_stop)
            - Otherwise hold current position

        Monthly snapping:
            The raw daily signal is only "executed" at the last trading day of
            each calendar month.  Between month-ends the position is frozen.
        """
        df = data.copy()

        # --- Indicators (all lagged 1 bar — no look-ahead) ---
        high_252 = df["close"].shift(1).rolling(window=self.breakout_period).max()
        df["high_252"] = high_252
        df["trail_stop_level"] = high_252 * (1 - self.trailing_stop)

        # --- Daily raw signal ---
        entry = df["close"] > high_252
        stop = df["close"] < df["trail_stop_level"]

        raw_signal = pd.Series(np.nan, index=df.index)
        raw_signal[entry] = 1
        raw_signal[stop] = 0
        df["raw_signal"] = raw_signal.ffill().fillna(0).astype(int)

        # --- Monthly snapping ---
        # A bar is the last trading day of its month when the next bar falls in
        # a different calendar month (or there is no next bar).
        next_bar_month = df.index.to_series().shift(-1).dt.month
        is_month_end = (df.index.to_series().dt.month != next_bar_month) | next_bar_month.isna()

        # At month-end bars, take the current raw signal; elsewhere carry forward
        monthly_signal = df["raw_signal"].where(is_month_end, other=np.nan)
        monthly_signal = monthly_signal.ffill().fillna(0).astype(int)

        # Shift by 1 bar to prevent look-ahead bias
        df["signal"] = monthly_signal.shift(1).fillna(0).astype(int)

        return df

    def __repr__(self) -> str:
        return (
            f"BreakoutTrendFollowing(breakout_period={self.breakout_period}, "
            f"trailing_stop={self.trailing_stop:.0%})"
        )
