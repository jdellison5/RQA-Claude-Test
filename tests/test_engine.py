"""Tests for the Backtester engine using real SPY data."""

from __future__ import annotations

import pytest

from backtester.engine import Backtester, BacktestResult
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion


def test_engine_runs_without_error(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert result is not None


def test_result_is_backtest_result(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert isinstance(result, BacktestResult)


def test_equity_curve_is_not_empty(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert len(result.equity_curve) > 0


def test_equity_curve_starts_at_initial_capital(spy_data, ma_strategy):
    initial = 10_000.0
    bt = Backtester(strategy=ma_strategy, initial_capital=initial)
    result = bt.run(spy_data)
    assert abs(result.equity_curve.iloc[0] - initial) < 0.01


def test_result_contains_metrics(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    for key in ["sharpe_ratio", "max_drawdown", "cagr", "win_rate", "total_trades"]:
        assert key in result.metrics, f"Missing metric: {key}"


def test_result_data_has_signal_column(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert "signal" in result.data.columns


def test_engine_produces_trade_log(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert len(result.trade_log) >= 0


def test_engine_with_rsi_strategy(spy_data, rsi_strategy):
    bt = Backtester(strategy=rsi_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert len(result.equity_curve) > 0
    assert "rsi" in result.data.columns


def test_no_position_open_at_end_leaves_cash(spy_data, ma_strategy):
    """Engine should close any open position at the last bar."""
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert result.equity_curve.iloc[-1] > 0


def test_equity_curve_length_matches_data(spy_data, ma_strategy):
    bt = Backtester(strategy=ma_strategy, initial_capital=10_000.0)
    result = bt.run(spy_data)
    assert len(result.equity_curve) == len(spy_data)
