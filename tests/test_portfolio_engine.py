"""Tests for the PortfolioBacktester (multi-strategy portfolio with monthly rebalancing)."""

from __future__ import annotations

import numpy as np
import pytest

from backtester.engine import BacktestResult
from backtester.portfolio_engine import PortfolioBacktester
from strategies.breakout_trend import BreakoutTrendFollowing
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_strategies():
    return [
        MovingAverageCrossover(fast_window=10, slow_window=20),
        RSIMeanReversion(period=2, oversold=25, overbought=75),
        BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05),
    ]


@pytest.fixture
def two_strategies():
    return [
        MovingAverageCrossover(fast_window=10, slow_window=20),
        RSIMeanReversion(period=2, oversold=25, overbought=75),
    ]


# ---------------------------------------------------------------------------
# Construction / weight normalisation
# ---------------------------------------------------------------------------

def test_default_weights_are_equal(three_strategies):
    pb = PortfolioBacktester(strategies=three_strategies)
    assert len(pb.weights) == 3
    for w in pb.weights:
        assert abs(w - 1 / 3) < 1e-9


def test_custom_weights_are_normalised(two_strategies):
    pb = PortfolioBacktester(strategies=two_strategies, weights=[1.0, 3.0])
    assert abs(pb.weights[0] - 0.25) < 1e-9
    assert abs(pb.weights[1] - 0.75) < 1e-9
    assert abs(sum(pb.weights) - 1.0) < 1e-9


def test_mismatched_weights_raises(two_strategies):
    with pytest.raises(ValueError, match=r"len\(weights\)"):
        PortfolioBacktester(strategies=two_strategies, weights=[0.5, 0.3, 0.2])


def test_empty_strategies_raises():
    with pytest.raises(ValueError, match="At least one strategy"):
        PortfolioBacktester(strategies=[])


def test_zero_weights_raises(two_strategies):
    with pytest.raises(ValueError, match="Weights must sum"):
        PortfolioBacktester(strategies=two_strategies, weights=[0.0, 0.0])


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_run_returns_backtest_result(spy_data, three_strategies):
    pb = PortfolioBacktester(strategies=three_strategies, initial_capital=10_000.0)
    result = pb.run(spy_data)
    assert isinstance(result, BacktestResult)


def test_equity_curve_length_matches_data(spy_data, three_strategies):
    pb = PortfolioBacktester(strategies=three_strategies, initial_capital=10_000.0)
    result = pb.run(spy_data)
    assert len(result.equity_curve) == len(spy_data)


def test_equity_curve_starts_near_initial_capital(spy_data, two_strategies):
    initial = 10_000.0
    pb = PortfolioBacktester(strategies=two_strategies, initial_capital=initial)
    result = pb.run(spy_data)
    # First bar: no strategy has a signal yet, all allocations stay at initial weight
    assert abs(result.equity_curve.iloc[0] - initial) < initial * 0.01


def test_metrics_are_populated(spy_data, three_strategies):
    pb = PortfolioBacktester(strategies=three_strategies)
    result = pb.run(spy_data)
    for key in ("cagr", "sharpe_ratio", "max_drawdown", "total_trades"):
        assert key in result.metrics, f"Missing metric: {key}"


def test_equity_curve_values_are_positive(spy_data, three_strategies):
    pb = PortfolioBacktester(strategies=three_strategies, initial_capital=10_000.0)
    result = pb.run(spy_data)
    assert (result.equity_curve > 0).all()


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def test_trade_log_has_strategy_column(spy_data, two_strategies):
    pb = PortfolioBacktester(strategies=two_strategies)
    result = pb.run(spy_data)
    if not result.trade_log.empty:
        assert "strategy" in result.trade_log.columns


def test_trade_log_contains_all_strategy_labels(spy_data, two_strategies):
    labels = ["Strategy A", "Strategy B"]
    pb = PortfolioBacktester(
        strategies=two_strategies, strategy_labels=labels
    )
    result = pb.run(spy_data)
    if not result.trade_log.empty:
        found = set(result.trade_log["strategy"].unique())
        # At least one strategy should appear; all present strategies are labelled
        assert found.issubset(set(labels))


# ---------------------------------------------------------------------------
# Enriched data columns
# ---------------------------------------------------------------------------

def test_enriched_data_has_individual_signal_columns(spy_data, two_strategies):
    labels = ["MA", "RSI"]
    pb = PortfolioBacktester(strategies=two_strategies, strategy_labels=labels)
    result = pb.run(spy_data)
    assert "signal_MA" in result.data.columns
    assert "signal_RSI" in result.data.columns


def test_enriched_data_has_blended_signal_column(spy_data, two_strategies):
    pb = PortfolioBacktester(strategies=two_strategies)
    result = pb.run(spy_data)
    assert "signal" in result.data.columns


def test_blended_signal_is_mean_of_individual_signals(spy_data, two_strategies):
    labels = ["MA", "RSI"]
    pb = PortfolioBacktester(strategies=two_strategies, strategy_labels=labels)
    result = pb.run(spy_data)
    expected = result.data[["signal_MA", "signal_RSI"]].mean(axis=1)
    np.testing.assert_allclose(result.data["signal"].values, expected.values, atol=1e-9)


# ---------------------------------------------------------------------------
# Monthly rebalancing
# ---------------------------------------------------------------------------

def test_monthly_rebalancing_keeps_equity_positive(spy_data, three_strategies):
    """Equity must stay positive throughout; rebalancing must not create negative values."""
    pb = PortfolioBacktester(strategies=three_strategies, initial_capital=10_000.0)
    result = pb.run(spy_data)
    assert (result.equity_curve > 0).all()


def test_single_strategy_portfolio_matches_standalone(spy_data):
    """
    A portfolio with one strategy and 100% weight should produce an equity
    curve that tracks the standalone Backtester result (same return profile).
    """
    from backtester.engine import Backtester

    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
    capital = 10_000.0

    standalone_result = Backtester(strategy=strategy, initial_capital=capital).run(spy_data)
    pb = PortfolioBacktester(strategies=[strategy], initial_capital=capital)
    portfolio_result = pb.run(spy_data)

    standalone_returns = standalone_result.equity_curve.pct_change().fillna(0)
    portfolio_returns = portfolio_result.equity_curve.pct_change().fillna(0)

    np.testing.assert_allclose(
        standalone_returns.values,
        portfolio_returns.values,
        atol=1e-6,
        err_msg="Single-strategy portfolio returns deviate from standalone backtester",
    )


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_repr_contains_strategy_labels(two_strategies):
    labels = ["Alpha", "Beta"]
    pb = PortfolioBacktester(strategies=two_strategies, strategy_labels=labels)
    r = repr(pb)
    assert "Alpha" in r
    assert "Beta" in r
