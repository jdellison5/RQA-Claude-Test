"""
Performance metrics for backtesting results.

All metrics use standard finance definitions:
  - Sharpe ratio: annualized excess return divided by annualized volatility
  - Max drawdown: largest peak-to-trough decline in equity
  - CAGR: compound annual growth rate
  - Win rate: fraction of trades with positive PnL
  - Profit factor: gross profit divided by gross loss
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio.

    Formula: sqrt(N) * mean(daily_returns - rfr/N) / std(daily_returns)
    where N = trading days per year (252).
    """
    if len(equity_curve) < 2:
        return 0.0
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown as a negative fraction.

    Formula: min((equity - running_max) / running_max)
    Returns a value <= 0. E.g. -0.25 means a 25% peak-to-trough drop.
    """
    if len(equity_curve) < 2:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Longest drawdown duration in trading days (bars).

    Counts the maximum number of consecutive bars below the previous equity peak.
    """
    if len(equity_curve) < 2:
        return 0
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max

    max_duration = 0
    current_duration = 0
    for is_under in underwater:
        if is_under:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    Formula: (final / initial) ^ (periods_per_year / n_periods) - 1
    """
    if len(equity_curve) < 2:
        return 0.0
    n = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -1.0
    return float(total_return ** (periods_per_year / n) - 1)


def win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of completed trades with positive PnL."""
    if trade_log.empty or "pnl" not in trade_log.columns:
        return 0.0
    completed = trade_log.dropna(subset=["pnl"])
    if completed.empty:
        return 0.0
    return float((completed["pnl"] > 0).mean())


def profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Gross profit divided by gross loss.

    Values > 1 are profitable. Returns inf if there are no losing trades.
    """
    if trade_log.empty or "pnl" not in trade_log.columns:
        return 0.0
    completed = trade_log.dropna(subset=["pnl"])
    gross_profit = completed.loc[completed["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(completed.loc[completed["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def summary_stats(result) -> dict:
    """Compute all metrics for a BacktestResult and return as a dict."""
    ec = result.equity_curve
    tl = result.trade_log
    return {
        "sharpe_ratio": sharpe_ratio(ec),
        "max_drawdown": max_drawdown(ec),
        "max_drawdown_duration_days": max_drawdown_duration(ec),
        "cagr": cagr(ec),
        "win_rate": win_rate(tl),
        "profit_factor": profit_factor(tl),
        "total_trades": len(tl),
        "final_equity": float(ec.iloc[-1]) if len(ec) > 0 else 0.0,
        "total_return": float(ec.iloc[-1] / ec.iloc[0] - 1) if len(ec) > 1 else 0.0,
    }
