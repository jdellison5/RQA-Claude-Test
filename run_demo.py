"""
Backtesting Demo Entry Point
============================

Runs two strategies (Moving Average Crossover and RSI Mean Reversion)
on a ticker of your choice, prints a metrics summary table, and saves
all charts to the outputs/ directory.

Usage:
    python run_demo.py
    python run_demo.py --ticker MSFT --start 2021-01-01 --end 2025-01-01
    python run_demo.py --ticker SPY --sweep-rsi

Examples shown during University of Richmond guest lecture (April 1, 2025).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; charts are saved to disk

from backtester.data import load_or_fetch
from backtester.engine import Backtester
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from visualization.charts import (
    plot_price_with_signals,
    plot_rsi_panel,
    plot_drawdown,
    print_summary_table,
)


def run_strategy(ticker: str, start: str, end: str, strategy, output_dir: str, chart_func, chart_kwargs: dict):
    print(f"\nFetching data for {ticker} ({start} → {end}) ...")
    data = load_or_fetch(ticker, start, end)
    print(f"  Loaded {len(data)} bars.")

    bt = Backtester(strategy=strategy, initial_capital=10_000.0)
    result = bt.run(data)

    print_summary_table(result.metrics, ticker=ticker, strategy_name=strategy.name)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = strategy.name.replace("(", "").replace(")", "").replace(",", "_")
    chart_path = os.path.join(output_dir, f"{ticker}_{safe_name}.png")

    fig = chart_func(result, ticker=ticker, strategy_name=strategy.name, **chart_kwargs)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved: {chart_path}")

    # Also save a drawdown chart
    dd_path = os.path.join(output_dir, f"{ticker}_{safe_name}_drawdown.png")
    dd_fig = plot_drawdown(result.equity_curve, title=f"{ticker} | {strategy.name} — Drawdown")
    dd_fig.savefig(dd_path, dpi=150, bbox_inches="tight")
    print(f"  Drawdown chart saved: {dd_path}")

    import matplotlib.pyplot as plt
    plt.close("all")

    return result


def run_rsi_sweep(ticker: str, start: str, end: str, output_dir: str):
    """Parameter sweep: compare RSI strategies with different oversold thresholds."""
    print(f"\n{'='*50}")
    print("RSI Oversold Level Parameter Sweep")
    print(f"{'='*50}")

    import pandas as pd
    data = load_or_fetch(ticker, start, end)
    rows = []

    for oversold_level in [25, 30, 35]:
        strategy = RSIMeanReversion(period=14, oversold=oversold_level, overbought=70)
        bt = Backtester(strategy=strategy, initial_capital=10_000.0)
        result = bt.run(data)
        m = result.metrics
        rows.append({
            "Oversold Level": oversold_level,
            "CAGR": f"{m['cagr']:.2%}",
            "Sharpe": f"{m['sharpe_ratio']:.2f}",
            "Max DD": f"{m['max_drawdown']:.2%}",
            "Trades": m["total_trades"],
            "Win Rate": f"{m['win_rate']:.2%}",
        })

    df = pd.DataFrame(rows).set_index("Oversold Level")
    print(df.to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description="Run backtesting demo")
    parser.add_argument("--ticker", default="SPY", help="Stock ticker symbol")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", default="outputs", help="Directory for chart output")
    parser.add_argument("--sweep-rsi", action="store_true", help="Run RSI parameter sweep")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Backtesting Demo  |  {args.ticker}  |  {args.start} → {args.end}")
    print(f"{'='*60}")

    # Strategy 1: Moving Average Crossover
    ma_strategy = MovingAverageCrossover(fast_window=20, slow_window=50)
    run_strategy(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        strategy=ma_strategy,
        output_dir=args.output_dir,
        chart_func=plot_price_with_signals,
        chart_kwargs={"initial_capital": 10_000.0},
    )

    # Strategy 2: RSI Mean Reversion
    rsi_strategy = RSIMeanReversion(period=14, oversold=30, overbought=70)
    run_strategy(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        strategy=rsi_strategy,
        output_dir=args.output_dir,
        chart_func=plot_rsi_panel,
        chart_kwargs={"initial_capital": 10_000.0, "oversold": 30, "overbought": 70},
    )

    # Optional parameter sweep
    if args.sweep_rsi:
        run_rsi_sweep(args.ticker, args.start, args.end, args.output_dir)

    print(f"\nAll done! Charts saved to '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
