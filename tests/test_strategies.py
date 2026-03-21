"""Tests for strategy signal generation using real SPY data."""

from __future__ import annotations

import numpy as np
import pytest

from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion


def test_ma_crossover_generates_signal_column(spy_data):
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
    result = strategy.generate_signals(spy_data)
    assert "signal" in result.columns


def test_ma_crossover_adds_sma_columns(spy_data):
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
    result = strategy.generate_signals(spy_data)
    assert "sma_fast" in result.columns
    assert "sma_slow" in result.columns


def test_ma_crossover_signal_values_are_valid(spy_data):
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
    result = strategy.generate_signals(spy_data)
    unique_signals = set(result["signal"].unique())
    assert unique_signals.issubset({0, 1}), f"Unexpected signal values: {unique_signals}"


def test_ma_crossover_no_lookahead_bias(spy_data):
    """
    Verify look-ahead bias prevention: the signal at bar t must not depend
    on the closing price at bar t.

    We check this by confirming that when the fast SMA first crosses the slow SMA
    at bar t, the signal at bar t is still 0 (the signal takes effect at bar t+1).
    """
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
    result = strategy.generate_signals(spy_data)

    fast = result["sma_fast"]
    slow = result["sma_slow"]
    golden_cross_bars = result.index[
        (fast > slow) & (fast.shift(1) <= slow.shift(1))
    ]

    if len(golden_cross_bars) > 0:
        first_cross = golden_cross_bars[0]
        cross_loc = result.index.get_loc(first_cross)
        signal_at_cross = result.iloc[cross_loc]["signal"]
        assert signal_at_cross == 0, (
            f"Look-ahead bias detected: signal={signal_at_cross} at cross bar {first_cross}. "
            "Signal should be 0 at the cross bar and 1 at the next bar."
        )


def test_ma_crossover_does_not_modify_original(spy_data):
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
    original_cols = list(spy_data.columns)
    strategy.generate_signals(spy_data)
    assert list(spy_data.columns) == original_cols


def test_rsi_generates_signal_and_rsi_columns(spy_data):
    strategy = RSIMeanReversion(period=2, oversold=25, overbought=75)
    result = strategy.generate_signals(spy_data)
    assert "rsi" in result.columns
    assert "signal" in result.columns


def test_rsi_values_bounded_0_to_100(spy_data):
    strategy = RSIMeanReversion(period=2, oversold=25, overbought=75)
    result = strategy.generate_signals(spy_data)
    rsi = result["rsi"].dropna()
    assert (rsi >= 0).all(), "RSI has values below 0"
    assert (rsi <= 100).all(), "RSI has values above 100"


def test_rsi_signal_values_are_valid(spy_data):
    strategy = RSIMeanReversion(period=2, oversold=25, overbought=75)
    result = strategy.generate_signals(spy_data)
    unique_signals = set(result["signal"].unique())
    assert unique_signals.issubset({0, 1}), f"Unexpected signal values: {unique_signals}"


def test_rsi_no_lookahead_bias(spy_data):
    """
    Verify look-ahead bias prevention: signal must not use today's close.

    We check the FIRST bar where RSI2 drops below the oversold level.
    At that bar, signal must be 0. The 1 only appears at bar t+1.
    """
    strategy = RSIMeanReversion(period=2, oversold=25, overbought=75)
    result = strategy.generate_signals(spy_data)

    rsi = result["rsi"]

    all_buy_bars = result.index[rsi < 25]

    if len(all_buy_bars) == 0:
        pytest.skip("No RSI2 oversold bars in SPY data for this period — adjust parameters")

    first_buy = all_buy_bars[0]
    first_buy_loc = result.index.get_loc(first_buy)

    signal_at_bar = result.iloc[first_buy_loc]["signal"]
    assert signal_at_bar == 0, (
        f"Look-ahead bias detected: signal={signal_at_bar} at first RSI2 oversold bar {first_buy}. "
        "Signal must be 0 at the bar where the condition triggers (acted on next bar)."
    )

    if first_buy_loc + 1 < len(result):
        signal_next = result.iloc[first_buy_loc + 1]["signal"]
        assert signal_next == 1, (
            f"Expected signal=1 at bar after first oversold bar, got {signal_next}"
        )


def test_strategy_repr_includes_parameters():
    strategy = MovingAverageCrossover(fast_window=5, slow_window=15)
    r = repr(strategy)
    assert "5" in r
    assert "15" in r


def test_rsi_strategy_repr_includes_parameters():
    strategy = RSIMeanReversion(period=7, oversold=25, overbought=75)
    r = repr(strategy)
    assert "7" in r
    assert "25" in r
    assert "75" in r
