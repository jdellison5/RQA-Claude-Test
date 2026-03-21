"""
Portfolio and position tracking for backtesting.

Tracks a single long position at a time. Records all completed trades
and maintains a daily equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    direction: str = "long"
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    def close(self, exit_date: pd.Timestamp, exit_price: float) -> None:
        self.exit_date = exit_date
        self.exit_price = exit_price
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.shares
        else:
            self.pnl = (self.entry_price - exit_price) * self.shares


class Portfolio:
    """
    Tracks cash, open position, and equity over time.

    Supports one open position at a time (long only by default).
    """

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.current_position: Optional[Trade] = None
        self._trades: list[Trade] = []
        self._equity_records: list[tuple[pd.Timestamp, float]] = []

    def open_position(
        self,
        date: pd.Timestamp,
        price: float,
        direction: str = "long",
    ) -> None:
        """Open a position using all available cash."""
        if self.current_position is not None:
            raise ValueError("Cannot open a new position while one is already open.")
        shares = self.cash / price
        self.current_position = Trade(
            entry_date=date,
            entry_price=price,
            shares=shares,
            direction=direction,
        )
        self.cash = 0.0

    def close_position(self, date: pd.Timestamp, price: float) -> None:
        """Close the current open position."""
        if self.current_position is None:
            raise ValueError("No open position to close.")
        trade = self.current_position
        trade.close(exit_date=date, exit_price=price)
        self.cash = trade.entry_price * trade.shares + trade.pnl  # recover cost basis + pnl
        self._trades.append(trade)
        self.current_position = None

    def mark_to_market(self, date: pd.Timestamp, current_price: float) -> None:
        """Record the current equity value at this date."""
        if self.current_position is not None:
            position_value = self.current_position.shares * current_price
            equity = self.cash + position_value
        else:
            equity = self.cash
        self._equity_records.append((date, equity))

    def equity_curve(self) -> pd.Series:
        """Return a DatetimeIndex Series of portfolio equity over time."""
        if not self._equity_records:
            return pd.Series(dtype=float)
        dates, values = zip(*self._equity_records)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

    def trade_log(self) -> pd.DataFrame:
        """Return a DataFrame of all completed trades."""
        if not self._trades:
            return pd.DataFrame(
                columns=["entry_date", "exit_date", "entry_price", "exit_price", "shares", "direction", "pnl"]
            )
        return pd.DataFrame(
            [
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "direction": t.direction,
                    "pnl": t.pnl,
                }
                for t in self._trades
            ]
        )
