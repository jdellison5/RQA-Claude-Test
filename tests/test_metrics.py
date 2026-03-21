"""Tests for performance metrics with known expected values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import metrics


def test_sharpe_of_flat_equity_is_zero(flat_equity_curve):
    assert metrics.sharpe_ratio(flat_equity_curve) == 0.0


def test_sharpe_of_single_bar_is_zero():
    eq = pd.Series([10_000.0])
    assert metrics.sharpe_ratio(eq) == 0.0


def test_sharpe_of_monotonically_increasing_equity(rising_equity_curve):
    sharpe = metrics.sharpe_ratio(rising_equity_curve)
    # Monotonically increasing equity has very high Sharpe
    assert sharpe > 2.0


def test_max_drawdown_known_series(drawdown_equity_curve):
    # Peak is 12000 (index 2 and 18+), trough is 9000 (index 8)
    # Max drawdown = (9000 - 12000) / 12000 = -0.25
    dd = metrics.max_drawdown(drawdown_equity_curve)
    assert abs(dd - (-0.25)) < 0.01


def test_max_drawdown_is_never_positive(rising_equity_curve):
    dd = metrics.max_drawdown(rising_equity_curve)
    assert dd <= 0.0


def test_max_drawdown_flat_is_zero(flat_equity_curve):
    assert metrics.max_drawdown(flat_equity_curve) == 0.0


def test_cagr_approximate_for_doubling(rising_equity_curve):
    # rising_equity_curve goes from 10000 to 20000 over 252 trading days
    # Expected CAGR = (20000/10000)^(252/252) - 1 = 1.0 = 100%
    result = metrics.cagr(rising_equity_curve)
    assert abs(result - 1.0) < 0.05  # within 5 percentage points


def test_cagr_single_bar_is_zero():
    eq = pd.Series([10_000.0])
    assert metrics.cagr(eq) == 0.0


def test_win_rate_all_winners():
    log = pd.DataFrame({"pnl": [100.0, 200.0, 50.0]})
    assert metrics.win_rate(log) == 1.0


def test_win_rate_all_losers():
    log = pd.DataFrame({"pnl": [-100.0, -200.0]})
    assert metrics.win_rate(log) == 0.0


def test_win_rate_mixed():
    log = pd.DataFrame({"pnl": [100.0, -50.0, 75.0, -25.0]})
    assert abs(metrics.win_rate(log) - 0.5) < 0.01


def test_win_rate_empty_log():
    log = pd.DataFrame(columns=["pnl"])
    assert metrics.win_rate(log) == 0.0


def test_profit_factor_all_profit():
    log = pd.DataFrame({"pnl": [100.0, 200.0]})
    assert metrics.profit_factor(log) == float("inf")


def test_profit_factor_breakeven():
    log = pd.DataFrame({"pnl": [100.0, -100.0]})
    assert abs(metrics.profit_factor(log) - 1.0) < 0.001


def test_profit_factor_two_to_one():
    log = pd.DataFrame({"pnl": [200.0, -100.0]})
    assert abs(metrics.profit_factor(log) - 2.0) < 0.001


def test_max_drawdown_duration_known():
    # 3-bar drawdown: bars 1,2,3 are below peak at bar 0
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([10000, 9500, 9000, 9200, 10500], index=dates, dtype=float)
    duration = metrics.max_drawdown_duration(eq)
    assert duration == 3
