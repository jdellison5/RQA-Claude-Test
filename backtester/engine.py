"""
Core backtesting engine.

Separates signal generation (done once, upfront) from execution (bar-by-bar).
This separation is intentional: it prevents look-ahead bias and mirrors how
a live trading system would work.

IMPORTANT: Strategies are responsible for shifting their own signals by one
bar (using .shift(1)) so we never act on a signal that used today's close price.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.portfolio import Portfolio
from strategies.base import BaseStrategy
from backtester import metrics as metrics_module


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    metrics: dict
    data: pd.DataFrame  # original OHLCV data enriched with signal columns


class Backtester:
    """
    Runs a strategy against historical price data bar by bar.

    Usage:
        bt = Backtester(strategy=MovingAverageCrossover(), initial_capital=10_000)
        result = bt.run(data)
    """

    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10_000.0) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest.

        Steps:
          1. Call strategy.generate_signals() once to produce signal column.
          2. Iterate bar-by-bar, executing trades based on prior-bar signal.
          3. Compute metrics and return a BacktestResult.
        """
        enriched = self.strategy.generate_signals(data.copy())
        portfolio = Portfolio(initial_capital=self.initial_capital)

        for date, row in enriched.iterrows():
            signal = row.get("signal", 0)
            price = row["close"]

            # Mark equity at the start of each bar (before any trade)
            portfolio.mark_to_market(date, price)

            # Execute based on signal (signal is already shifted in the strategy)
            if signal == 1 and portfolio.current_position is None:
                portfolio.open_position(date, price, direction="long")
            elif signal == 0 and portfolio.current_position is not None:
                portfolio.close_position(date, price)

        # Close any open position at end of data
        if portfolio.current_position is not None:
            last_date = enriched.index[-1]
            last_price = enriched.iloc[-1]["close"]
            portfolio.close_position(last_date, last_price)

        equity_curve = portfolio.equity_curve()
        trade_log = portfolio.trade_log()
        result = BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics={},
            data=enriched,
        )
        result.metrics = metrics_module.summary_stats(result)
        return result
