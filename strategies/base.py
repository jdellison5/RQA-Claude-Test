"""
Abstract base class for all trading strategies.

Every strategy must implement generate_signals(), which receives an OHLCV
DataFrame and returns the same DataFrame enriched with at minimum a 'signal'
column where:
    1  = go long (buy)
    0  = stay flat / close any open long position

CRITICAL: Strategies MUST shift signals by one bar (.shift(1)) to prevent
look-ahead bias. You cannot trade on a signal computed from today's closing
price — you'd only know that signal after the market closes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Receive OHLCV DataFrame, return it enriched with a 'signal' column.

        The signal column must be shifted by 1 bar to avoid look-ahead bias.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
