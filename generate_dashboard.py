"""
Generate an interactive HTML dashboard for backtest results.

Runs all strategies on the specified ticker, then produces a single
self-contained HTML file you can open in any browser.

Usage:
    python generate_dashboard.py
    python generate_dashboard.py --ticker MSFT --start 2021-01-01 --end 2025-01-01
    python generate_dashboard.py --ticker SPY --output outputs/spy_dashboard.html
"""

from __future__ import annotations

import argparse
import os

from backtester.data import load_or_fetch
from backtester.engine import Backtester
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from visualization.dashboard import generate_html_dashboard


def main():
    parser = argparse.ArgumentParser(description="Generate HTML backtest dashboard")
    parser.add_argument("--ticker",  default="AAPL",           help="Stock ticker symbol")
    parser.add_argument("--start",   default="2020-01-01",     help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2024-01-01",     help="End date YYYY-MM-DD")
    parser.add_argument("--capital", default=10_000.0, type=float, help="Initial capital")
    parser.add_argument("--output",  default="outputs/dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    print(f"\nFetching data for {args.ticker} ({args.start} → {args.end}) ...")
    data = load_or_fetch(args.ticker, args.start, args.end)
    print(f"  Loaded {len(data)} bars.\n")

    strategies = [
        ("MA Crossover (20/50)",       MovingAverageCrossover(fast_window=20, slow_window=50)),
        ("RSI Mean Reversion (14)",    RSIMeanReversion(period=14, oversold=30, overbought=70)),
    ]

    results = []
    for label, strategy in strategies:
        print(f"Running: {label} ...")
        bt = Backtester(strategy=strategy, initial_capital=args.capital)
        result = bt.run(data)
        m = result.metrics
        print(f"  CAGR: {m['cagr']:.2%}  |  Sharpe: {m['sharpe_ratio']:.2f}  "
              f"|  Max DD: {m['max_drawdown']:.2%}  |  Trades: {m['total_trades']}")
        results.append((label, result))

    print(f"\nGenerating dashboard ...")
    path = generate_html_dashboard(
        results=results,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        output_path=args.output,
    )
    print(f"  Done! Open in your browser:\n\n    {path}\n")


if __name__ == "__main__":
    main()
