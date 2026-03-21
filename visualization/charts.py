"""
Visualization for backtest results.

All charts use matplotlib only (no Plotly dependency).
Charts can be saved to disk via save_path parameter.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _buy_hold_equity(data: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Compute a buy-and-hold equity curve for benchmark comparison."""
    close = data["close"]
    shares = initial_capital / close.iloc[0]
    return (close * shares).rename("buy_and_hold")


def plot_price_with_signals(
    result,
    ticker: str = "",
    strategy_name: str = "",
    initial_capital: float = 10_000.0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    2-panel chart:
      Panel 1: Close price + SMA lines (if present) + buy/sell markers
      Panel 2: Strategy equity curve vs buy-and-hold benchmark
    """
    data = result.data
    equity = result.equity_curve
    trade_log = result.trade_log

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    title = f"{ticker} — {strategy_name}" if ticker or strategy_name else "Backtest Results"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Panel 1: Price + signals ---
    ax1.plot(data.index, data["close"], color="steelblue", linewidth=1.2, label="Close Price")

    if "sma_fast" in data.columns:
        ax1.plot(data.index, data["sma_fast"], color="orange", linewidth=1, linestyle="--", label=f"SMA Fast")
    if "sma_slow" in data.columns:
        ax1.plot(data.index, data["sma_slow"], color="red", linewidth=1, linestyle="--", label=f"SMA Slow")

    if not trade_log.empty:
        buys = trade_log.dropna(subset=["entry_date"])
        sells = trade_log.dropna(subset=["exit_date"])
        for _, trade in buys.iterrows():
            ax1.scatter(trade["entry_date"], trade["entry_price"], marker="^", color="green", s=80, zorder=5)
        for _, trade in sells.iterrows():
            ax1.scatter(trade["exit_date"], trade["exit_price"], marker="v", color="red", s=80, zorder=5)

    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Equity curve vs benchmark ---
    bh = _buy_hold_equity(data, initial_capital)
    ax2.plot(equity.index, equity.values, color="steelblue", linewidth=1.5, label="Strategy")
    ax2.plot(bh.index, bh.values, color="gray", linewidth=1.2, linestyle="--", label="Buy & Hold")
    ax2.axhline(initial_capital, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")

    return fig


def plot_rsi_panel(
    result,
    ticker: str = "",
    strategy_name: str = "",
    initial_capital: float = 10_000.0,
    oversold: int = 30,
    overbought: int = 70,
    save_path: str | None = None,
) -> plt.Figure:
    """
    3-panel chart for RSI strategy:
      Panel 1: Price with buy/sell markers
      Panel 2: RSI oscillator with overbought/oversold bands
      Panel 3: Equity curve vs benchmark
    """
    data = result.data
    equity = result.equity_curve
    trade_log = result.trade_log

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    title = f"{ticker} — {strategy_name}" if ticker or strategy_name else "RSI Strategy Results"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Price
    ax1.plot(data.index, data["close"], color="steelblue", linewidth=1.2)
    if not trade_log.empty:
        for _, trade in trade_log.dropna(subset=["entry_date"]).iterrows():
            ax1.scatter(trade["entry_date"], trade["entry_price"], marker="^", color="green", s=80, zorder=5)
        for _, trade in trade_log.dropna(subset=["exit_date"]).iterrows():
            ax1.scatter(trade["exit_date"], trade["exit_price"], marker="v", color="red", s=80, zorder=5)
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: RSI
    if "rsi" in data.columns:
        ax2.plot(data.index, data["rsi"], color="purple", linewidth=1.2, label="RSI")
        ax2.axhline(overbought, color="red", linewidth=1, linestyle="--", label=f"Overbought ({overbought})")
        ax2.axhline(oversold, color="green", linewidth=1, linestyle="--", label=f"Oversold ({oversold})")
        ax2.fill_between(data.index, oversold, data["rsi"].clip(upper=oversold), alpha=0.1, color="green")
        ax2.fill_between(data.index, overbought, data["rsi"].clip(lower=overbought), alpha=0.1, color="red")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

    # Panel 3: Equity
    bh = _buy_hold_equity(data, initial_capital)
    ax3.plot(equity.index, equity.values, color="steelblue", linewidth=1.5, label="Strategy")
    ax3.plot(bh.index, bh.values, color="gray", linewidth=1.2, linestyle="--", label="Buy & Hold")
    ax3.axhline(initial_capital, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")

    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    save_path: str | None = None,
) -> plt.Figure:
    """Underwater equity chart showing drawdown periods."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100  # in percent

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")

    return fig


def print_summary_table(metrics: dict, ticker: str = "", strategy_name: str = "") -> None:
    """Print a formatted metrics table to the console."""
    header = f"  {ticker} | {strategy_name}  " if ticker or strategy_name else "  Backtest Results  "
    sep = "=" * max(len(header), 42)

    print(f"\n{sep}")
    print(header)
    print(sep)

    rows = [
        ("CAGR",                    f"{metrics.get('cagr', 0):.2%}"),
        ("Total Return",            f"{metrics.get('total_return', 0):.2%}"),
        ("Sharpe Ratio",            f"{metrics.get('sharpe_ratio', 0):.2f}"),
        ("Max Drawdown",            f"{metrics.get('max_drawdown', 0):.2%}"),
        ("Max DD Duration (days)",  f"{metrics.get('max_drawdown_duration_days', 0)}"),
        ("Total Trades",            f"{metrics.get('total_trades', 0)}"),
        ("Win Rate",                f"{metrics.get('win_rate', 0):.2%}"),
        ("Profit Factor",           f"{metrics.get('profit_factor', 0):.2f}"),
        ("Final Equity",            f"${metrics.get('final_equity', 0):,.2f}"),
    ]

    for label, value in rows:
        print(f"  {label:<28} {value:>10}")

    print(sep + "\n")
