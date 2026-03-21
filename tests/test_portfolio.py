"""Tests for Portfolio and Trade logic."""

from __future__ import annotations

import pytest
import pandas as pd

from backtester.portfolio import Portfolio


def test_initial_equity_equals_capital(sample_portfolio):
    ts = pd.Timestamp("2020-01-02")
    sample_portfolio.mark_to_market(ts, 100.0)
    eq = sample_portfolio.equity_curve()
    assert abs(eq.iloc[0] - 10_000.0) < 0.01


def test_open_position_zeros_cash(sample_portfolio):
    ts = pd.Timestamp("2020-01-02")
    sample_portfolio.open_position(ts, price=100.0)
    assert sample_portfolio.cash == 0.0


def test_open_position_correct_shares(sample_portfolio):
    ts = pd.Timestamp("2020-01-02")
    sample_portfolio.open_position(ts, price=100.0)
    assert abs(sample_portfolio.current_position.shares - 100.0) < 0.001


def test_pnl_calculation_on_profitable_trade(sample_portfolio):
    t1 = pd.Timestamp("2020-01-02")
    t2 = pd.Timestamp("2020-01-03")
    sample_portfolio.open_position(t1, price=100.0)   # buy 100 shares
    sample_portfolio.close_position(t2, price=110.0)  # sell at 110
    log = sample_portfolio.trade_log()
    assert len(log) == 1
    assert abs(log.iloc[0]["pnl"] - 1_000.0) < 0.01  # 100 shares * $10 gain


def test_pnl_calculation_on_losing_trade(sample_portfolio):
    t1 = pd.Timestamp("2020-01-02")
    t2 = pd.Timestamp("2020-01-03")
    sample_portfolio.open_position(t1, price=100.0)
    sample_portfolio.close_position(t2, price=90.0)
    log = sample_portfolio.trade_log()
    assert log.iloc[0]["pnl"] < 0
    assert abs(log.iloc[0]["pnl"] - (-1_000.0)) < 0.01


def test_cannot_open_two_positions(sample_portfolio):
    t1 = pd.Timestamp("2020-01-02")
    sample_portfolio.open_position(t1, price=100.0)
    with pytest.raises(ValueError, match="already open"):
        sample_portfolio.open_position(t1, price=105.0)


def test_cannot_close_without_open_position(sample_portfolio):
    t1 = pd.Timestamp("2020-01-02")
    with pytest.raises(ValueError, match="No open position"):
        sample_portfolio.close_position(t1, price=100.0)


def test_equity_curve_reflects_mark_to_market(sample_portfolio):
    t1 = pd.Timestamp("2020-01-02")
    t2 = pd.Timestamp("2020-01-03")
    sample_portfolio.open_position(t1, price=100.0)  # 100 shares
    sample_portfolio.mark_to_market(t1, 100.0)
    sample_portfolio.mark_to_market(t2, 105.0)       # equity = 100 * 105 = 10500
    eq = sample_portfolio.equity_curve()
    assert abs(eq.iloc[-1] - 10_500.0) < 0.01


def test_trade_log_empty_initially():
    p = Portfolio(initial_capital=5_000.0)
    log = p.trade_log()
    assert len(log) == 0
