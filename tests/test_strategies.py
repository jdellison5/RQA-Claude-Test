"""Tests for strategy signal generation using real SPY data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.breakout_trend import BreakoutTrendFollowing
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
    assert "sma_trend" in result.columns
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

    We check the FIRST bar where RSI2 drops below oversold AND price > SMA trend filter.
    At that bar, signal must be 0. The 1 only appears at bar t+1.
    """
    strategy = RSIMeanReversion(period=2, oversold=25, overbought=75)
    result = strategy.generate_signals(spy_data)

    rsi = result["rsi"]
    above_trend = result["close"] > result["sma_trend"]

    # Find bars where both conditions for a buy are met
    all_buy_bars = result.index[(rsi < 25) & above_trend]

    if len(all_buy_bars) == 0:
        pytest.skip("No RSI2 oversold bars above trend SMA in SPY data — adjust parameters")

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
    strategy = RSIMeanReversion(period=7, oversold=25, overbought=75, trend_sma=200)
    r = repr(strategy)
    assert "7" in r
    assert "25" in r
    assert "75" in r
    assert "200" in r


def test_rsi_trend_filter_blocks_longs_below_sma(spy_data):
    """Signal must be 0 on any bar where close < sma_trend (after the shift)."""
    strategy = RSIMeanReversion(period=2, oversold=25, overbought=75, trend_sma=200)
    result = strategy.generate_signals(spy_data).dropna(subset=["sma_trend"])

    below_trend = result["close"] < result["sma_trend"]
    # signal at bar t reflects conditions at bar t-1, so compare to previous bar's trend state
    prev_below_trend = below_trend.shift(1).fillna(False)
    signals_when_prev_below = result.loc[prev_below_trend, "signal"]
    assert (signals_when_prev_below == 0).all(), (
        "Trend filter violated: signal=1 found on a bar where previous close was below SMA"
    )


# ---------------------------------------------------------------------------
# BreakoutTrendFollowing tests
# ---------------------------------------------------------------------------

def test_breakout_generates_expected_columns(spy_data):
    strategy = BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05)
    result = strategy.generate_signals(spy_data)
    for col in ("high_252", "trail_stop_level", "raw_signal", "signal"):
        assert col in result.columns, f"Missing column: {col}"


def test_breakout_signal_values_are_valid(spy_data):
    strategy = BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05)
    result = strategy.generate_signals(spy_data)
    unique = set(result["signal"].unique())
    assert unique.issubset({0, 1}), f"Unexpected signal values: {unique}"


def test_breakout_trail_stop_level_correct(spy_data):
    """trail_stop_level must equal high_252 * 0.95 everywhere."""
    strategy = BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05)
    result = strategy.generate_signals(spy_data).dropna(subset=["high_252"])
    expected = result["high_252"] * 0.95
    pd.testing.assert_series_equal(result["trail_stop_level"], expected, check_names=False)


def test_breakout_no_lookahead_bias(spy_data):
    """
    At the first bar where a breakout triggers, signal must still be 0.
    The 1 can only appear at the following bar.
    """
    strategy = BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05)
    result = strategy.generate_signals(spy_data).dropna(subset=["high_252"])

    breakout_bars = result.index[result["close"] > result["high_252"]]
    if len(breakout_bars) == 0:
        pytest.skip("No 252-day breakout in synthetic data — adjust parameters")

    first = breakout_bars[0]
    loc = result.index.get_loc(first)

    assert result.iloc[loc]["signal"] == 0, (
        f"Look-ahead bias: signal=1 at breakout bar {first}, expected 0"
    )
    if loc + 1 < len(result):
        # Next bar should reflect the month-end snap — may be 1 if first is month-end
        # Just verify it's a valid value (the key assertion is the one above)
        assert result.iloc[loc + 1]["signal"] in (0, 1)


def test_breakout_signal_changes_only_at_month_end(spy_data):
    """
    Signal transitions (0→1 or 1→0) must only occur on bars that immediately
    follow a month-end bar, because of the 1-bar shift applied after monthly snapping.
    """
    strategy = BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05)
    result = strategy.generate_signals(spy_data)

    sig = result["signal"]
    changed = sig != sig.shift(1)
    change_locs = result.index[changed & sig.shift(1).notna()]

    if len(change_locs) == 0:
        pytest.skip("No signal changes in data — strategy never triggered")

    # The bar *before* each change bar must have been a month-end
    idx_series = result.index.to_series()
    for bar in change_locs:
        loc = result.index.get_loc(bar)
        if loc == 0:
            continue  # first bar initialises from NaN — skip
        prev_bar = result.index[loc - 1]
        prev_month = idx_series.iloc[loc - 1].month
        curr_month = idx_series.iloc[loc].month
        # prev_bar is month-end when the following bar (current bar) is in a new month
        assert prev_month != curr_month or loc == 1, (
            f"Signal changed at {bar} but previous bar {prev_bar} was not a month-end"
        )


def test_breakout_repr_includes_parameters():
    strategy = BreakoutTrendFollowing(breakout_period=126, trailing_stop=0.08)
    r = repr(strategy)
    assert "126" in r
    assert "8%" in r
