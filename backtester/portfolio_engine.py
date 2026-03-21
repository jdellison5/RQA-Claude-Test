"""
Portfolio backtester: runs multiple strategies and blends their equity curves
with periodic (monthly) rebalancing.

How it works
------------
1. Each constituent strategy is run independently via the standard Backtester
   with initial_capital=1.0 so returns are normalised.
2. The daily return series of every strategy are blended according to the
   target weights, starting from initial_capital.
3. At the last trading day of each calendar month the allocations are reset
   to target weights (equal weight by default).  Between rebalancings the
   allocations drift freely with each strategy's daily returns.
4. The result is returned as a BacktestResult so it can be fed directly into
   the existing dashboard/metrics pipeline.

Trade log
---------
The combined trade log concatenates each strategy's individual trades with a
"strategy" label column appended.  Win rate and profit factor computed from
this log are valid (they are ratios unaffected by the capital normalisation).

Usage
-----
    from backtester.portfolio_engine import PortfolioBacktester
    from strategies.ma_crossover import MovingAverageCrossover
    from strategies.rsi_mean_reversion import RSIMeanReversion
    from strategies.breakout_trend import BreakoutTrendFollowing

    pb = PortfolioBacktester(
        strategies=[MovingAverageCrossover(), RSIMeanReversion(), BreakoutTrendFollowing()],
        initial_capital=10_000.0,
    )
    result = pb.run(data)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtester.engine import Backtester, BacktestResult
from backtester import metrics as metrics_module
from strategies.base import BaseStrategy


class PortfolioBacktester:
    """
    Multi-strategy portfolio backtester with monthly rebalancing.

    Parameters
    ----------
    strategies:
        List of strategy instances to include in the portfolio.
    weights:
        Target allocation weights, one per strategy.  Will be normalised to
        sum to 1.  Defaults to equal weight.
    strategy_labels:
        Display names for each strategy.  Defaults to strategy.name.
    initial_capital:
        Starting portfolio value in dollars.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        weights: Optional[list[float]] = None,
        strategy_labels: Optional[list[str]] = None,
        initial_capital: float = 10_000.0,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required.")
        n = len(strategies)

        raw_weights = weights if weights is not None else [1.0] * n
        if len(raw_weights) != n:
            raise ValueError(
                f"len(weights)={len(raw_weights)} must match len(strategies)={n}"
            )
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")

        self.strategies = strategies
        self.weights = [w / total for w in raw_weights]  # normalised
        self.strategy_labels = (
            strategy_labels if strategy_labels is not None
            else [s.name for s in strategies]
        )
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run the portfolio backtest.

        Steps:
          1. Run each constituent strategy with capital=1.0 to get normalised
             equity curves.
          2. Convert equity curves to daily return series.
          3. Simulate the blended portfolio bar-by-bar, rebalancing at each
             month-end back to target weights.
          4. Assemble BacktestResult with combined equity curve and trade log.
        """
        sub_results = self._run_sub_strategies(data)
        equity_curve = self._blend_equity_curves(sub_results, data)
        trade_log = self._combine_trade_logs(sub_results)
        enriched_data = self._build_enriched_data(data, sub_results)

        result = BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics={},
            data=enriched_data,
        )
        result.metrics = metrics_module.summary_stats(result)
        return result

    def __repr__(self) -> str:
        labels = ", ".join(self.strategy_labels)
        weights = ", ".join(f"{w:.0%}" for w in self.weights)
        return f"PortfolioBacktester([{labels}], weights=[{weights}])"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_sub_strategies(self, data: pd.DataFrame) -> list[BacktestResult]:
        """Run each strategy with capital=1.0 for normalised returns."""
        results = []
        for strategy in self.strategies:
            bt = Backtester(strategy=strategy, initial_capital=1.0)
            results.append(bt.run(data))
        return results

    def _blend_equity_curves(
        self,
        sub_results: list[BacktestResult],
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Build the combined portfolio equity curve with monthly rebalancing.

        At each bar: apply each strategy's return to its current allocation.
        At month-end: rebalance allocations back to target weights.
        """
        # Align equity curves into a DataFrame of normalised daily returns
        returns_df = pd.DataFrame(
            {i: r.equity_curve.pct_change().fillna(0.0) for i, r in enumerate(sub_results)}
        )

        # Detect month-end bars: a bar is month-end when the following bar
        # belongs to a different calendar month (or there is no following bar)
        idx_series = returns_df.index.to_series()
        next_month = idx_series.shift(-1).dt.month
        is_month_end = (idx_series.dt.month != next_month) | next_month.isna()

        # Simulate bar by bar
        allocations = np.array(self.weights) * self.initial_capital
        equity_records: list[tuple[pd.Timestamp, float]] = []

        for date, ret_row in returns_df.iterrows():
            allocations = allocations * (1.0 + ret_row.values)
            total = float(allocations.sum())
            equity_records.append((date, total))

            if is_month_end[date]:
                allocations = np.array(self.weights) * total

        dates, values = zip(*equity_records)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

    def _combine_trade_logs(self, sub_results: list[BacktestResult]) -> pd.DataFrame:
        """Concatenate each strategy's trade log, adding a 'strategy' column."""
        frames = []
        for label, result in zip(self.strategy_labels, sub_results):
            tl = result.trade_log.copy()
            if not tl.empty:
                tl["strategy"] = label
                frames.append(tl)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "entry_date", "exit_date", "entry_price", "exit_price",
                    "shares", "direction", "pnl", "strategy",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def _build_enriched_data(
        self,
        data: pd.DataFrame,
        sub_results: list[BacktestResult],
    ) -> pd.DataFrame:
        """
        Attach each strategy's signal column to the OHLCV data.

        Adds columns:
            signal_<label>  : each strategy's individual signal
            signal          : mean of all strategy signals (fractional allocation)
        """
        enriched = data.copy()
        sig_cols = []
        for label, result in zip(self.strategy_labels, sub_results):
            col = f"signal_{label}"
            enriched[col] = result.data["signal"]
            sig_cols.append(col)
        enriched["signal"] = enriched[sig_cols].mean(axis=1)
        return enriched
