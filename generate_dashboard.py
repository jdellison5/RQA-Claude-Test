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
from backtester.portfolio_engine import PortfolioBacktester
from strategies.breakout_trend import BreakoutTrendFollowing
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from visualization.dashboard import generate_html_dashboard


def main():
    parser = argparse.ArgumentParser(description="Generate HTML backtest dashboard")
    parser.add_argument("--ticker",  default="SPY",  help="Stock ticker symbol")
    parser.add_argument("--start",   default=None,   help="Start date YYYY-MM-DD (default: full history)")
    parser.add_argument("--end",     default=None,   help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", default=10_000.0, type=float, help="Initial capital")
    parser.add_argument("--output",  default="outputs/dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    date_range = f"{args.start or 'max'} → {args.end or 'today'}"
    print(f"\nFetching data for {args.ticker} ({date_range}) ...")
    data = load_or_fetch(args.ticker, args.start, args.end)
    print(f"  Loaded {len(data)} bars.\n")

    individual_strategies = [
        ("MA Crossover (20/50)",            MovingAverageCrossover(fast_window=20, slow_window=50)),
        ("RSI Mean Reversion (2)",          RSIMeanReversion(period=2, oversold=25, overbought=75)),
        ("252-Day Breakout + 5% Trail",     BreakoutTrendFollowing(breakout_period=252, trailing_stop=0.05)),
    ]

    results = []
    for label, strategy in individual_strategies:
        print(f"Running: {label} ...")
        bt = Backtester(strategy=strategy, initial_capital=args.capital)
        result = bt.run(data)
        m = result.metrics
        print(f"  CAGR: {m['cagr']:.2%}  |  Sharpe: {m['sharpe_ratio']:.2f}  "
              f"|  Max DD: {m['max_drawdown']:.2%}  |  Trades: {m['total_trades']}")
        results.append((label, result))

    # Combined equal-weight portfolio
    portfolio_label = "Equal-Weight Portfolio (Monthly Rebal)"
    print(f"Running: {portfolio_label} ...")
    pb = PortfolioBacktester(
        strategies=[s for _, s in individual_strategies],
        strategy_labels=[lbl for lbl, _ in individual_strategies],
        initial_capital=args.capital,
    )
    portfolio_result = pb.run(data)
    m = portfolio_result.metrics
    print(f"  CAGR: {m['cagr']:.2%}  |  Sharpe: {m['sharpe_ratio']:.2f}  "
          f"|  Max DD: {m['max_drawdown']:.2%}  |  Trades: {m['total_trades']}")
    results.append((portfolio_label, portfolio_result))

    print(f"\nGenerating dashboard ...")
    path = generate_html_dashboard(
        results=results,
        ticker=args.ticker,
        start=args.start or "max",
        end=args.end or "today",
        initial_capital=args.capital,
        output_path=args.output,
    )
    print(f"  Done! Open in your browser:\n\n    {path}\n")


if __name__ == "__main__":
    main()
